import argparse
import os
from logging import getLogger

import torch
import numpy as np
import yaml
from accelerate import Accelerator
from torch.utils.data import TensorDataset, DataLoader

from layers import RQVAEModel
from trainer import Trainer
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments', help='Dataset name')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Config file')
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)


    config = {}


    config.update(yaml.safe_load(open(args.config_file, 'r')))
    config.update(command_line_configs)

    config['run_local_time'] = get_local_time()
    config['dataset'] = args.dataset
    config['data_dir'] = os.path.join(config['data_dir'], config['dataset'])

    config = convert_config_dict(config)

    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()


    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    log(f'Device: {config["device"]}', config['accelerator'], logger)
    log(f'Config: {str(config)}', config['accelerator'], logger)

    id_mapping_file = os.path.join(config['data_dir'], 'id_mapping.json')
    id_mapping = load_json(id_mapping_file)

    sent_emb_path = os.path.join(
		config['data_dir'], f'{os.path.basename(config["sent_emb_model"])}.sent_emb'
	)
    if os.path.exists(sent_emb_path):
        log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...', config['accelerator'], logger)
        sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, config['sent_emb_dim'])
    else:
        if config['use_ddp']:
            raise RuntimeError('Sentence embeddings must be generated in a single process.')

        log(f'[TOKENIZER] Encoding sentence embeddings...', config['accelerator'], logger)

        meta_file = os.path.join(config['data_dir'], 'metadata.sentence.json')
        metadata = load_json(meta_file)
        sent_embs = encode_sent_emb(config, metadata, id_mapping, sent_emb_path)

    # PCA
    if config['sent_emb_pca'] > 0:
        print(f'[TOKENIZER] Applying PCA to sentence embeddings...')
        from sklearn.decomposition import PCA

        pca = PCA(n_components=config['sent_emb_pca'], whiten=True)
        sent_embs = pca.fit_transform(sent_embs)

    log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}', config['accelerator'], logger)


    seq_file = os.path.join(config['data_dir'], 'all_item_seqs.json')
    all_item_seqs = load_json(seq_file)
    train_item = set()
    for user in all_item_seqs:
        items = all_item_seqs[user]
        items = items[:-2]
        train_item.update(items)
    train_item = sorted(list(train_item))

    train_item_list = []
    for item in train_item:
        item_index = id_mapping['item2id'][item] - 1
        train_item_list.append(item_index)
    train_item_list = sorted(train_item_list)
    sent_embs = sent_embs[train_item_list]
    log(f'[TOKENIZER] Sentence embeddings shape after filtering: {sent_embs.shape}', config['accelerator'], logger)

    # Generate semantic IDs
    sent_embs = torch.FloatTensor(sent_embs)


    all_hidden_sizes = [sent_embs.shape[1]] + config['hidden_sizes']
    rqvae_model = RQVAEModel(
		hidden_sizes=all_hidden_sizes,
		n_codebooks=config['n_codebooks'],
		codebook_size=config['codebook_size'],
		dropout=config['dropout'],
		vq_type=config['vq_type'],
        beta=config['beta'],
	).to(config['device'])

    log(rqvae_model, config['accelerator'], logger)

    # print(rqvae_model.quantization_layer.quantization_layers[0].embed[0])
    rqvae_model.generate_codebook(sent_embs, config['device'])
    # print(rqvae_model.quantization_layer.quantization_layers[0].embed[0])

    dataset = TensorDataset(sent_embs)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], pin_memory=True, shuffle=True)
    # for d in dataloader:
    #     print(d[0][0])
    #     break

    trainer = Trainer(config, rqvae_model, len(dataloader))
    rqvae_model = trainer.fit(dataloader)

