import torch
import torch.nn as nn
import torch.nn.functional as F

class GATv2(nn.Module):
    def __init__(self, config):
        super(GATv2, self).__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.out_channels = config.out_channels
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate
        self.num_classes = config.num_classes

    def forward(self, x):
        N, T, C, E = x.shape
        x = x.reshape(N * T, C, E)
        out = torch.zeros(N * T, self.num_classes, dtype=torch.float32)
        out = out.reshape(N, T, self.num_classes)

        return out