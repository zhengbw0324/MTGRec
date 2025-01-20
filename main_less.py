import argparse
import os
from logging import getLogger

import torch
import math
import numpy as np
import yaml
from accelerate import Accelerator

from collator import Collator
from model import STIGER
from trainer import Trainer
from utils import *
from data_utils import *
from grad_utils import *
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments', help='Dataset name')
    parser.add_argument('--config_file', type=str, default='./config/ptconfig.yaml', help='Config file')
    return parser.parse_known_args()


def accumulate_inf_score(stored_less_scores, checkpoint_weights):
    if sum(checkpoint_weights) != 1:
        s = sum(checkpoint_weights)
        weights = [i / s for i in checkpoint_weights]
    else:
        weights = checkpoint_weights

    final_score = 0
    for i in range(len(stored_less_scores)):
        final_score += stored_less_scores[i] * weights[i]

    return final_score



def main(config):
    
    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    accelerator = config['accelerator']
    log(f'Device: {config["device"]}', accelerator, logger)
    log(f'Config: {str(config)}', accelerator, logger)

    # Tokenizer and Dataset
    tokenizers = get_tokenizers(config)
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    less_train_dataset, less_valid_dataset = get_less_datasets(config)
    train_collate_fn = Collator(config, tokenizers)
    test_collate_fn = Collator(config, tokenizers)

    with accelerator.main_process_first():
        model = STIGER(config, train_dataset, tokenizers[-1])

    log(model, accelerator, logger)
    log(model.n_parameters, accelerator, logger)

    train_data = get_dataloader(config, train_dataset, train_collate_fn, 'train')
    valid_data = get_dataloader(config, valid_dataset, test_collate_fn, 'valid')
    test_data = get_dataloader(config, test_dataset, test_collate_fn, 'test')

    epoch_per_stage = config['epoch_per_stage']
    if config['load_best_for_next_stage'] and config['val_delay'] >= epoch_per_stage[0]:
        config['val_delay'] = epoch_per_stage[0] - 1

    while sum(epoch_per_stage) > config['epochs']:
        epoch_per_stage = epoch_per_stage[:-1]

    trainer = Trainer(config, model, tokenizers[-1], train_data)

    tau = config['tau']
    n_tokenizers = len(tokenizers)

    checkpoint_weights = []
    stored_less_scores = []

    for stage_count, n_epoch in enumerate(epoch_per_stage):
        epoch_bias = 0 if stage_count == 0 else np.cumsum(epoch_per_stage).tolist()[stage_count - 1]
        early_stopping = trainer.fit(train_data, valid_data, n_epoch, epoch_bias)
        if early_stopping:
            break

        if config['load_best_for_next_stage']:
            accelerator.wait_for_everyone()
            trainer.load_states(trainer.saved_model_ckpt)

        model = trainer.model
        adam_optimizer_state = accelerator.unwrap_model(trainer.optimizer).state_dict()['state']
        checkpoint_weights.append(trainer.scheduler.get_last_lr()[0])

        valid_scores = np.array(trainer.all_scores)
        log(f'Validation scores for all tokenizers: {str(valid_scores)}', accelerator, logger)

        # calculate influence scores
        train_grads = []
        for i in range(n_tokenizers):
            train_grad_data = DataLoader(less_train_dataset, batch_size=config['train_batch_size'],
                                         collate_fn=train_data.collate_fn,
                                         num_workers=config['num_proc'], shuffle=False)
            train_grad_data.collate_fn.set_tokenizer(i)
            train_grad_data = accelerator.prepare(train_grad_data)
            train_grad = collect_train_grads(train_grad_data, model, accelerator, proj_dim=8192,
                                             adam_optimizer_state=adam_optimizer_state, )
            train_grads.append(train_grad)

        valid_grads = []
        for i in range(n_tokenizers):
            valid_grad_data = DataLoader(less_valid_dataset, batch_size=config['train_batch_size'],
                                         collate_fn=valid_data.collate_fn,
                                         num_workers=config['num_proc'], shuffle=False)
            valid_grad_data.collate_fn.set_tokenizer(i)
            valid_grad_data = accelerator.prepare(valid_grad_data)
            valid_grad = collect_valid_grads(valid_grad_data, model, accelerator, proj_dim=8192)
            valid_grads.append(valid_grad)


        train_grads = torch.cat(train_grads, dim=0)
        valid_grads = torch.cat(valid_grads, dim=0)

        inf_scores = calculate_influence_score(train_grads, valid_grads)
        log(f'Influence scores: {inf_scores}', accelerator, logger)
        stored_less_scores.append(inf_scores)

        inf_scores = accumulate_inf_score(stored_less_scores, checkpoint_weights)

        inf_scores = inf_scores.mean(-1)
        log(f'Mean influence score: {inf_scores.tolist()}', accelerator, logger)
        inf_scores = inf_scores.cpu().numpy()

        # inf_scores = center_score(inf_scores)
        select_prob = inf_scores / tau
        # log(select_prob, accelerator, logger)
        select_prob = np.exp(select_prob)
        select_prob = select_prob / np.sum(select_prob)
        train_data.collate_fn.set_select_prob(select_prob)
        train_data.collate_fn.set_tokenizer(None)
        valid_data.collate_fn.set_tokenizer(None)


        log(f'Stage {stage_count} selected prob: {select_prob.tolist()}', accelerator, logger)

    if not early_stopping:
        epoch_bias = sum(epoch_per_stage)
        trainer.fit(train_data, valid_data, config['epochs']-epoch_bias, epoch_bias)

    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    model_states = torch.load(trainer.saved_model_ckpt, map_location=trainer.model.device)['model']
    model.load_state_dict(model_states)
    log(f'Loaded best model checkpoint from {trainer.saved_model_ckpt}', accelerator, logger)


    trainer.model, test_data = accelerator.prepare(
        model, test_data
    )
    test_results, all_results = trainer.evaluate_all_tokenizer(test_data)

    if accelerator.is_main_process:
        for key in test_results:
            accelerator.log({f'Test_Metric/{key}': test_results[key]})

        for i, results in enumerate(all_results):
            for key in results:
                accelerator.log({f'Test_{i}_Metric/{key}': results[key]})

    log(f'Test Results: {test_results}', accelerator, logger)
    for i, results in enumerate(all_results):
        log(f'Test Results {i}: {results}', accelerator, logger)

    trainer.end()
    
    
    
if __name__ == '__main__':    
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    # Config
    config = {}
    config.update(yaml.safe_load(open(args.config_file, 'r')))
    config.update(command_line_configs)

    config['run_local_time'] = get_local_time()

    ckpt_name = get_file_name(config)
    config['ckpt_name'] = ckpt_name
    config['dataset'] = args.dataset
    config['data_dir'] = os.path.join(config['data_dir'], config['dataset'])
    config['ckpt_dir'] = os.path.join(config['ckpt_dir'], config['dataset'], ckpt_name)

    config = convert_config_dict(config)

    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()
    torch.distributed.barrier(device_ids=[int(os.environ['LOCAL_RANK'])])
    
    main(config)


    

