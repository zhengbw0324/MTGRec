import copy
import logging

import numpy as np
import torch
from time import time

from accelerate import Accelerator
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from utils import log, delete_file, ensure_dir
import torch.nn.functional as F

import os


class Trainer(object):

    def __init__(self, config, model, data_num):
        self.config = config
        self.model = model
        self.logger = logging.getLogger()

        self.use_ddp = config['use_ddp']

        self.lr = config["lr"]
        self.learner = config["learner"]
        self.scheduler_type = config["scheduler_type"]

        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.warmup_steps = config["warmup_steps"] * data_num
        self.max_steps = self.epochs * data_num

        self.save_limit = config["save_limit"]
        self.ckpt_queue = []
        self.verbose_step = min(config["verbose_step"], self.epochs)
        self.verbose_delay = min(config["verbose_delay"], self.epochs)
        self.device = config['device']

        self.data_dir = config['data_dir']


        self.best_loss = np.inf
        self.ckpt_name = config['ckpt_name']
        self.ckpt_dir = os.path.join(self.data_dir, self.ckpt_name)
        ensure_dir(self.ckpt_dir)
        self.best_loss_ckpt = f"{self.ckpt_name}.pth"


        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._get_scheduler()

        self.accelerator = Accelerator()

        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)


    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler


    def _train_epoch(self, train_data):

        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_quant_loss = 0.0
        total_count = 0.0

        for batch in train_data:
            x_batch = batch[0].to(self.config['device'])
            self.optimizer.zero_grad()
            recon_x, quant_loss, count = self.model(x_batch)
            reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
            loss = reconstruction_mse_loss + quant_loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            total_loss += loss.detach().cpu().item()
            total_recon_loss += reconstruction_mse_loss.detach().cpu().item()
            total_quant_loss += quant_loss.detach().cpu().item()
            total_count += count

        return total_loss, total_recon_loss, total_quant_loss, total_count

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()
        # valid_data = copy.deepcopy(valid_data)

        indices_set = set()
        num_sample = 0
        for batch in valid_data:

            x_batch = batch[0].to(self.config['device'])
            num_sample += len(x_batch)
            if self.use_ddp:
                indices = self.model.module.encode(x_batch)
            else:
                indices = self.model.encode(x_batch)
            # print(indices.shape)
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        if self.use_ddp:
            # gather_for_metrics
            # self.log(indices_set)
            indices_set_list = self.accelerator.gather_for_metrics(indices_set)
            num_sample_list = self.accelerator.gather_for_metrics((num_sample,))
            # self.log(num_sample_list)
            # for s in indices_set_list:
            #     indices_set = indices_set.union(s)
            indices_set.update(indices_set_list)
            # self.log(indices_set)
            num_sample = sum(num_sample_list)
        # self.log(indices_set)
        # self.log(num_sample)
        collision_rate = (num_sample - len(list(indices_set))) / num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, ckpt_file=None):



        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, f'{self.ckpt_name}-{epoch+1}.pth')

        if self.accelerator.is_main_process:
            if self.config['use_ddp']:  # unwrap model for saving
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                model_state = unwrapped_model.state_dict()
            else:
                model_state = self.model.state_dict()

            state = {
                "config": self.config,
                "epoch": epoch + 1,
                "best_loss": self.best_loss,
                "state_dict": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.lr_scheduler.state_dict(),
            }
            torch.save(state, ckpt_path, pickle_protocol=4)

        return ckpt_path


    def fit(self, dataloader):

        dataloader = self.accelerator.prepare(dataloader)
        self.model.train()

        for epoch_idx in tqdm(range(self.epochs),disable=not self.accelerator.is_main_process):
            # train
            total_loss, total_recon_loss, total_quant_loss, total_count = self._train_epoch(dataloader)

            self.accelerator.wait_for_everyone()
            # eval
            if (epoch_idx + 1) >= self.verbose_delay and (epoch_idx + 1) % self.verbose_step == 0:

                collision_rate = self._valid_epoch(dataloader)

                self.log(
                    f"[TOKENIZER] training\n"
                    f"\tEpoch [{epoch_idx + 1}/{self.epochs}]\n"
                    f"\t  Training lr: {self.lr_scheduler.get_last_lr()}\n"
                    f"\t  Training loss: {total_loss / len(dataloader)}\n"
                    f"\t  Unused codebook:{total_count / len(dataloader)}\n"
                    f"\t  Recosntruction loss: {total_recon_loss / len(dataloader)}\n"
                    f"\t  Quantization loss: {total_quant_loss / len(dataloader)}\n"
                    f"\t  Collision Rate: {collision_rate}\n")

                if total_loss < self.best_loss:
                    self.best_loss = total_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                ckpt_path = self._save_checkpoint(epoch_idx)
                self.ckpt_queue.append(ckpt_path)
                if len(self.ckpt_queue) > self.save_limit:
                    del_ckpt = self.ckpt_queue.pop(0)
                    if self.accelerator.is_main_process:
                        delete_file(del_ckpt)

        self.accelerator.wait_for_everyone()


        return self.model

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)




