import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

KINECT_EDGES = [
    (0, 1),   # SpineBase - SpineMid
    (1, 2),   # SpineMid - SpineShoulder
    (2, 3),   # SpineShoulder - Neck
    (3, 4),   # Neck - Head
    (2, 5),   # SpineShoulder - ShoulderLeft
    (5, 6),   # ShoulderLeft - ElbowLeft
    (6, 7),   # ElbowLeft - WristLeft
    (7, 8),   # WristLeft - HandLeft
    (2, 9),   # SpineShoulder - ShoulderRight
    (9, 10),  # ShoulderRight - ElbowRight
    (10, 11), # ElbowRight - WristRight
    (11, 12), # WristRight - HandRight
    (0, 13),  # SpineBase - HipLeft
    (13, 14), # HipLeft - KneeLeft
    (14, 15), # KneeLeft - AnkleLeft
    (15, 16), # AnkleLeft - FootLeft
    (0, 17),  # SpineBase - HipRight
    (17, 18), # HipRight - KneeRight
    (18, 19), # KneeRight - AnkleRight
    (19, 20), # AnkleRight - FootRight
    (7, 21),  # WristLeft - HandTipLeft
    (7, 22),  # WristLeft - ThumbLeft
    (11, 23), # WristRight - HandTipRight
    (11, 24), # WristRight - ThumbRight
]


def build_edge_index(edges, num_nodes, bidirectional=True):
    src, dst = zip(*edges)
    src, dst = list(src), list(dst)
    if bidirectional:
        src, dst = src + dst, dst + src
    return torch.tensor([src, dst], dtype=torch.long)


def tsm(x, fold_div=8):
    """
    Temporal Shift Module (no parameters).
    x: (B, T, N, C)
    Shifts a fraction of channels forward and backward in time.
    """
    B, T, N, C = x.shape
    fold = C // fold_div
    out = torch.zeros_like(x)
    out[:, 1:,  :, :fold]       = x[:, :-1, :, :fold]        # forward shift
    out[:, :-1, :, fold:2*fold] = x[:, 1:,  :, fold:2*fold]  # backward shift
    out[:, :,   :, 2*fold:]     = x[:, :,   :, 2*fold:]       # unchanged
    return out


class GATv2(nn.Module):
    def __init__(self, config):
        super(GATv2, self).__init__()
        self.in_channels = int(config.in_channels)
        self.hidden_channels = int(config.hidden_channels)
        self.out_channels = int(config.out_channels)
        self.num_heads = int(config.num_heads)
        self.dropout = float(config.dropout)
        self.num_nodes = int(config.num_nodes)

        assert self.hidden_channels % self.num_heads == 0
        self.head_dim = self.hidden_channels // self.num_heads

        # Linear projection: in_channels(3) → hidden_channels(128)
        # Required to ensure sufficient channels for TSM (fold = hidden // fold_div = 16)
        self.input_proj = nn.Linear(self.in_channels, self.hidden_channels)

        # GATv2Conv operates on projected hidden_channels(128)
        self.gat = GATv2Conv(
            in_channels=self.hidden_channels,
            out_channels=self.head_dim,
            heads=self.num_heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )

        edge_index = build_edge_index(KINECT_EDGES, self.num_nodes, bidirectional=True)
        self.register_buffer('edge_index', edge_index)

        self.classifier = nn.Linear(self.hidden_channels, self.out_channels)

        # Save original GAT forward once at init for monkey-patching
        self._original_gat_forward = self.gat.forward

    def forward(self, x):
        # x: (B, T, C, N)
        B, T, C, N = x.shape

        # Monkey-patch GAT forward to inject TSM before each GAT call
        def patched_forward(x_flat, edge_index, **kwargs):
            # x_flat: (B*T*N, hidden=128)
            H = x_flat.size(-1)
            x_4d = x_flat.reshape(B, T, N, H)          # restore temporal axis: (B, T, N, 128)
            x_shifted = tsm(x_4d)                       # apply TSM (fold = 128 // 8 = 16)
            x_flat2 = x_shifted.reshape(B * T * N, H)  # flatten back
            return self._original_gat_forward(x_flat2, edge_index, **kwargs)

        self.gat.forward = patched_forward

        # Linear projection: (B,T,C,N) → (B,T,N,C) → (B*T*N, C) → (B*T*N, 128)
        x = x.permute(0, 1, 3, 2)              # (B, T, N, C=3)
        x = x.reshape(B * T * N, C)            # (B*T*N, C=3)
        x = self.input_proj(x)                 # (B*T*N, 128)

        # Expand edge_index for all graphs in the batch
        num_graphs = B * T
        E = self.edge_index.size(1)
        offset = torch.arange(num_graphs, device=x.device) * N
        offset = offset.unsqueeze(1).expand(-1, E)
        edge_index_batch = self.edge_index.unsqueeze(0).expand(num_graphs, -1, -1)
        edge_index_batch = edge_index_batch + offset.unsqueeze(1)
        edge_index_batch = edge_index_batch.transpose(0, 1).reshape(2, -1)

        out = self.gat(x, edge_index_batch)    # patched_forward: TSM → GAT
        out = F.elu(out)                        # (B*T*N, hidden_channels)

        out = out.reshape(B * T, N, self.hidden_channels)
        out = out.mean(dim=1)                   # mean pooling: (B*T, hidden)
        out = self.classifier(out)              # (B*T, num_classes)

        return out.reshape(B, T, self.out_channels)