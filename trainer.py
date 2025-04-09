import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler
from torch.utils.data import DataLoader


from model import AbstractModel
from tokenizer import Tokenizer
from evaluator import Evaluator
from utils import *


class Trainer:

    def __init__(self, config: dict, model: AbstractModel, tokenizer: Tokenizer, train_dataloader: DataLoader):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.logger = getLogger()

        self.stage = config['stage']
        if self.stage == 'finetune' and config['pretrained_model']!="" and os.path.exists(config['pretrained_model']):
            self.log(f'Loading pretrained model from {config["pretrained_model"]}')
            model_states = torch.load(config['pretrained_model'], map_location='cpu')['model']
            missing_keys, unexpected_keys = self.model.load_state_dict(model_states)
            self.log(f"Missing keys: {missing_keys}")
            self.log(f"Unexpected keys: {unexpected_keys}")

        self.evaluator = Evaluator(config, tokenizer)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)

        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )


        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            f'{self.config["ckpt_name"]}.pth'
        )

        self.results_dir = self.config['results_dir'] if self.config['results_dir'] else self.config['ckpt_dir']
        ensure_dir(self.results_dir)
        
        self.best_epoch = 0
        self.best_val_score = -1
        self.val_delay = self.config['val_delay']
        n = max(1, len(self.config['sem_id_epochs']))
        self.all_scores = [0]* n

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)


    def save_states(self, epoch=0, path=None):
        path = path if path is not None else self.saved_model_ckpt
        if self.accelerator.is_main_process:
            if self.config['use_ddp']:  # unwrap model for saving
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_optimizer = self.accelerator.unwrap_model(self.optimizer)
                unwrapped_scheduler = self.accelerator.unwrap_model(self.scheduler)
                states = {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': unwrapped_optimizer.state_dict(),
                    'scheduler': unwrapped_scheduler.state_dict()
                }
                torch.save(states, path)
            else:
                states = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()
                }
                torch.save(states, path)
            self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {path}')

    def load_states(self, ckpt_path=None):
        ckpt_path = self.saved_model_ckpt if ckpt_path is None else ckpt_path
        ckpt = torch.load(ckpt_path, map_location=self.model.device)
        self.log(f'Loading model checkpoint from {ckpt_path}')
        if self.config['use_ddp']:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_optimizer = self.accelerator.unwrap_model(self.optimizer)
            unwrapped_scheduler = self.accelerator.unwrap_model(self.scheduler)
            unwrapped_model.load_state_dict(ckpt['model'])
            unwrapped_optimizer.load_state_dict(ckpt['optimizer'])
            unwrapped_scheduler.load_state_dict(ckpt['scheduler'])
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(unwrapped_model, unwrapped_optimizer, unwrapped_scheduler)
        else:
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])


    def fit(self, train_dataloader, val_dataloader, epochs, epoch_bias=0):

        train_dataloader, val_dataloader = self.accelerator.prepare(
            train_dataloader, val_dataloader
        )

        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        early_stopping = False

        for epoch in range(epoch_bias, epochs+epoch_bias):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
                disable=not self.accelerator.is_main_process,
            )
            for batch in train_progress_bar:

                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                
                train_progress_bar.set_postfix(lr=self.scheduler.get_last_lr(), loss=loss.item())
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')


            if (self.config['save_interval'] is not None
                and (epoch + 1) % self.config['save_interval'] == 0):
                epoch_ckpt_path = os.path.join(
                    self.config['ckpt_dir'],
                    f'{self.config["ckpt_name"]}_{epoch+1}.pth'
                )
                self.save_states(epoch=epoch, path=epoch_ckpt_path)

            # Evaluation
            if (epoch + 1) > self.val_delay and (epoch + 1) % self.config['eval_interval'] == 0:
                mean_results, all_results = self.evaluate_all_tokenizer(val_dataloader, split='val')

                for i, results in enumerate(all_results):
                    score = results[self.config['val_metric']]
                    self.all_scores[i] = max(self.all_scores[i], score)
                
                self.log_results(mean_results, epoch, prefix='Val')
                for i, results in enumerate(all_results):
                    self.log_results(results, epoch, prefix=f'Val_{i}')

                val_score = mean_results[self.config['val_metric']]
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_epoch = epoch + 1
                    
                    self.save_states(epoch=epoch)

                if self.config['patience'] is not None and epoch + 1 - self.best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    early_stopping = True
                    break

            self.accelerator.wait_for_everyone()


        self.log(f'Best epoch: {self.best_epoch}, Best val score: {self.best_val_score}')
        if self.best_val_score==-1:
            self.save_states(epoch=epochs+epoch_bias)

        return early_stopping

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
            disable=not self.accelerator.is_main_process,
        )
        all_results_info = {"preds": [], "scores": [], "labels": []}
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']:  # ddp, gather data from all devices for evaluation
                    preds, scores = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_scores, all_labels = self.accelerator.gather_for_metrics(
                        (preds, scores, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                    all_results_info["preds"].append(all_preds.detach().cpu())
                    all_results_info["scores"].append(all_scores.detach().cpu())
                    all_results_info["labels"].append(all_labels.detach().cpu())
                else:
                    preds, scores = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])
                    all_results_info["preds"].append(preds.detach().cpu())
                    all_results_info["scores"].append(scores.detach().cpu())
                    all_results_info["labels"].append(batch['labels'].detach().cpu())

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        for key in all_results_info:
            all_results_info[key] = torch.cat(all_results_info[key], dim=0).tolist()

        return output_results, all_results_info


    def store_results(self, results_info, collate_fn, split='test'):
        """
        Store the results in a file.

        Args:
            results_info (dict): The results info to store.
            collate_fn (Collator): The collate function used for data loading.
        """
        preds = results_info['preds']
        pred_ids = []
        for i in range(len(preds)):
            item_list = []
            for j in range(len(preds[i])):
                item = collate_fn.tokens2item(preds[i][j])
                item_list.append(item)
            pred_ids.append(item_list)
        results_info['pred_ids'] = pred_ids

        labels = results_info['labels']
        label_ids = []
        eos_token = collate_fn.tokenizers[0].eos_token
        for i in range(len(labels)):
            cur_label = labels[i]
            if eos_token in cur_label:
                eos_pos = cur_label.index(eos_token)
                cur_label = cur_label[:eos_pos]

            target_item = collate_fn.tokens2item(cur_label)
            label_ids.append(target_item)
        results_info['label_ids'] = label_ids

        if self.accelerator.is_main_process:
            if len(self.config['sem_id_epochs']) == 1:
                tokenizer_id = self.config['sem_id_epochs'][0]
            else:
                tokenizer_id = collate_fn.tokenizer_id

            results_info_path = os.path.join(self.results_dir, f"{split}_results_{tokenizer_id}.json")

            with open(results_info_path, 'w') as f:
                json.dump(results_info, f)
            self.log(f'Stored results to {results_info_path}')


    def evaluate_all_tokenizer(self, dataloader, split='test', store=False):

        tokenizer_num = dataloader.collate_fn.tokenizer_num

        results_list = []

        for tokenizer_id in range(tokenizer_num):
            dataloader.collate_fn.set_tokenizer(tokenizer_id)
            results, results_info = self.evaluate(dataloader, split)
            results_list.append(results)
            if store:
                self.store_results(results_info, dataloader.collate_fn, split)

        mean_results = OrderedDict()

        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                mean_results[key] = np.mean([result[key] for result in results_list])

        return mean_results, results_list

    def log_results(self, results, epoch, prefix='Val'):

        if self.accelerator.is_main_process:
            for key in results:
                self.accelerator.log({f"{prefix}_Metric/{key}": results[key]}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] {prefix} Results: {results}')
    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)