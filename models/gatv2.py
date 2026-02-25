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
        self.dropout = config.dropout

        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, T, C, E = x.shape
        x = x.reshape(N * T, C, E)
        out = torch.zeros(N * T, self.out_channels, dtype=torch.float32)
        out = out + self.dummy_param
        out = out.reshape(N, T, self.out_channels)

        return out