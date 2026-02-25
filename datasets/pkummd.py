import os
import torch
from torch.utils.data import Dataset

class PKUMMD(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = config.data_dir
        self.num_frames = config.num_frames
        self.num_nodes = config.num_nodes
        self.num_classes = config.num_classes
        self.in_channels = config.in_channels

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        dummy_input = torch.zeros((self.num_frames, self.in_channels, self.num_nodes), dtype=torch.float32)
        dummy_target = torch.zeros(self.num_frames, dtype=torch.long)
        return dummy_input, dummy_target