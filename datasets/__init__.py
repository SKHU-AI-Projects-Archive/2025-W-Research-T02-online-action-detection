import torch
from torch.utils.data import DataLoader
from .oad import OAD
from .pkummd import PKUMMD

def build_dataset(config, split):
    datasets = {
        'OAD': OAD,
        'PKUMMD': PKUMMD
    }
    if config.dataset not in datasets:
        raise ValueError(f"Unsupported dataset: {config.dataset}. Supported datasets are: {list(datasets.keys())}")

    return datasets[config.dataset](config, split)

def build_dataloader(config, split):
    dataset = build_dataset(config, split)
    
    is_train = True if split == 'train' else False
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        drop_last=is_train,
        pin_memory=True,
    )
    return loader