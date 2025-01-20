import argparse
import os
from collections import defaultdict
from logging import getLogger

import torch
import numpy as np
import yaml
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from layers import RQVAEModel
from utils import *


def extend_tokens(all_item_tokens, id2item, codebook_size):
   
    tokens2item = defaultdict(list)
    item2tokens = {}
    max_conflict = 0
    for i in range(all_item_tokens.shape[0]):
        str_id = ' '.join(map(str, all_item_tokens[i].tolist()))
        tokens2item[str_id].append(i + 1)
        item = id2item[i + 1]
        item2tokens[item] = (*tuple(all_item_tokens[i].tolist()), len(tokens2item[str_id]))
        max_conflict = max(max_conflict, len(tokens2item[str_id]))
    print(f'[TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
    print(f'Conflict rate: {( all_item_tokens.shape[0] - len(tokens2item) )/ all_item_tokens.shape[0]}')
    if max_conflict > codebook_size:
        raise ValueError(
            f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
            f'{max_conflict} > {codebook_size}. Please increase the codebook size.'
        )
    return item2tokens


def generate_tokens(config, rqvae_model, dataloader, id2item, epoch_ckpt=None):
    rqvae_model.eval()

    all_item_tokens = []
    for batch in tqdm(dataloader):
        item_tokens = rqvae_model.encode(batch[0].to(config['device']))
        all_item_tokens.append(item_tokens)
    all_item_tokens = np.concatenate(all_item_tokens, axis=0)
    print(f'[TOKENIZER] Item tokens shape: {all_item_tokens.shape}')

    item2tokens = extend_tokens(all_item_tokens, id2item, config['codebook_size'])
    codebook = [config['codebook_size']] * (config['n_codebooks'] + 1)
    if epoch_ckpt is not None:
        tokens_path = os.path.join(
            config['data_dir'], config["ckpt_name"],
            f'{os.path.basename(config["sent_emb_model"])}_{list_to_str(codebook, remove_blank=True)}_{epoch_ckpt}.sem_ids'
        )
    else:
        tokens_path = os.path.join(
            config['data_dir'], config["ckpt_name"],
            f'{os.path.basename(config["sent_emb_model"])}_{list_to_str(codebook, remove_blank=True)}.sem_ids'
        )

    print(f'[TOKENIZER] Saving item tokens to {tokens_path}...')
    with open(tokens_path, 'w') as f:
        json.dump(item2tokens, f)
    return item2tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments', help='Dataset name')
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--epoch_ckpts', nargs='+',
                        default=[9950,9951,9952,9953,9954,9955,9956,9957,9958,9959,9960,9961,9962,9963,9964,9965,9966,9967,9968,9969,
                                 9970,9971,9972,9973,9974,9975,9976,9977,9978,9979,9980,9981,9982,9983,9984,9985,9986,9987,9988,9989,
                                 9990,9991,9992,9993,9994,9995,9996,9997,9998,9999,10000],
                        type=int, help='ckpt list')
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

    config['device'], _ = init_device()
    init_seed(config['rand_seed'], config['reproducibility'])

    sent_emb_path = os.path.join(
		config['data_dir'], f'{os.path.basename(config["sent_emb_model"])}.sent_emb'
	)

    if os.path.exists(sent_emb_path):
        print(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
        sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, config['sent_emb_dim'])
    else:
        raise RuntimeError(f'[TOKENIZER] Sentence embeddings not found.')

    # PCA
    if config['sent_emb_pca'] > 0:
        print(f'[TOKENIZER] Applying PCA to sentence embeddings...')
        from sklearn.decomposition import PCA

        pca = PCA(n_components=config['sent_emb_pca'], whiten=True)
        sent_embs = pca.fit_transform(sent_embs)

    print(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

    id_mapping_file = os.path.join(config['data_dir'], 'id_mapping.json')
    id_mapping = load_json(id_mapping_file)
    id2item = id_mapping['id2item']

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

    print(rqvae_model)
    dataset = TensorDataset(sent_embs)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    # args.epoch_ckpts = list(range(9501, 10001))


    for epoch_ckpt in args.epoch_ckpts:
        ckpt_path = os.path.join(config['data_dir'], config["ckpt_name"], f'{config["ckpt_name"]}-{epoch_ckpt}.pth')
        print(f'[TOKENIZER] Loading RQ-VAE model checkpoint from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=config['device'])
        rqvae_model.load_state_dict( ckpt['state_dict'] )

        generate_tokens(config, rqvae_model, dataloader, id2item, epoch_ckpt)
        print("=====================================================")

    ckpt_path = os.path.join(config['data_dir'], config["ckpt_name"], f'{config["ckpt_name"]}.pth')
    print(f'[TOKENIZER] Loading RQ-VAE model checkpoint from {ckpt_path}...')
    ckpt = torch.load(ckpt_path, map_location=config['device'])
    print("Bset epoch:", ckpt["epoch"])
    rqvae_model.load_state_dict(ckpt['state_dict'])

    generate_tokens(config, rqvae_model, dataloader, id2item)