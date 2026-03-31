# models/gatv2_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

KINECT_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (5, 6), (6, 7), (7, 8),
    (2, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (7, 21), (7, 22), (11, 23), (11, 24),
]

def build_edge_index(edges, num_nodes, bidirectional=True):
    src, dst = zip(*edges)
    src, dst = list(src), list(dst)
    if bidirectional:
        src, dst = src + dst, dst + src
    return torch.tensor([src, dst], dtype=torch.long)


class GATv2Baseline(nn.Module):
    def __init__(self, config):
        super(GATv2Baseline, self).__init__()
        self.in_channels = int(config.in_channels)       # 3
        self.hidden_channels = int(config.hidden_channels)
        self.out_channels = int(config.out_channels)
        self.num_heads = int(config.num_heads)
        self.dropout = float(config.dropout)
        self.num_nodes = int(config.num_nodes)

        assert self.hidden_channels % self.num_heads == 0
        self.head_dim = self.hidden_channels // self.num_heads

        # No input_proj, no TSM — raw in_channels(3) → GATv2Conv directly
        self.gat = GATv2Conv(
            in_channels=self.in_channels,       # 3 (not 128)
            out_channels=self.head_dim,
            heads=self.num_heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )

        edge_index = build_edge_index(KINECT_EDGES, self.num_nodes, bidirectional=True)
        self.register_buffer('edge_index', edge_index)

        self.classifier = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x):
        # x: (B, T, C, N)
        B, T, C, N = x.shape

        x = x.permute(0, 1, 3, 2)          # (B, T, N, C=3)
        x = x.reshape(B * T * N, C)        # (B*T*N, 3)

        num_graphs = B * T
        E = self.edge_index.size(1)
        offset = torch.arange(num_graphs, device=x.device) * N
        offset = offset.unsqueeze(1).expand(-1, E)
        edge_index_batch = self.edge_index.unsqueeze(0).expand(num_graphs, -1, -1)
        edge_index_batch = edge_index_batch + offset.unsqueeze(1)
        edge_index_batch = edge_index_batch.transpose(0, 1).reshape(2, -1)

        out = self.gat(x, edge_index_batch)
        out = F.elu(out)                    # (B*T*N, hidden_channels)

        out = out.reshape(B * T, N, self.hidden_channels)
        out = out.mean(dim=1)               # mean pooling: (B*T, hidden)
        out = self.classifier(out)          # (B*T, num_classes)

        return out.reshape(B, T, self.out_channels)