import copy

from torch.utils.data import ConcatDataset, DataLoader

from dataset import SeqRecDataset
from tokenizer import STIGERTokenizer, Tokenizer


def get_datasets(config):
    train_dataset = SeqRecDataset(config, split='train')
    valid_dataset = SeqRecDataset(config, split='valid', sample_ratio=config['val_ratio'])
    test_dataset = SeqRecDataset(config, split='test')

    return train_dataset, valid_dataset, test_dataset



def get_less_datasets(config):
    train_dataset = SeqRecDataset(config, split='train')
    valid_dataset = SeqRecDataset(config, split='valid')

    return train_dataset, valid_dataset



def get_abl_tokenizers(config):
    tokenizers = []
    # "rqvae_seed_{}/sentence-t5-base_256,256,256,256"
    token_prefix = config["token_prefix"]
    for sem_id_epoch in config["sem_id_epochs"]:
        config["token_prefix"] = token_prefix.format(sem_id_epoch)
        tokenizers.append(Tokenizer(config))

    return tokenizers


def get_tokenizers(config):
    tokenizers = []

    for sem_id_epoch in config["sem_id_epochs"]:
        tokenizer = STIGERTokenizer(config, sem_id_epoch)
        tokenizers.append(tokenizer)
    if len(tokenizers) ==0:
        tokenizers.append(Tokenizer(config))

    return tokenizers


def get_dataloader(config, dataset, collate_fn, split):

    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], collate_fn=collate_fn,
                                num_workers=config['num_proc'], shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=config['eval_batch_size'], collate_fn=collate_fn,
                                num_workers=config['num_proc'], shuffle=False)


    return dataloader
