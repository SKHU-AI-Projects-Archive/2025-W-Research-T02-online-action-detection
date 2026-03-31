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


def compute_edge_attr(x_4d_raw, edge_index_base):
    x_4d_raw = x_4d_raw[..., :3]
    B, T, N, C = x_4d_raw.shape
    src = edge_index_base[0]
    dst = edge_index_base[1]
    E = src.size(0)

    x_flat = x_4d_raw.reshape(B * T, N, C)
    pos_i = x_flat[:, src, :]
    pos_j = x_flat[:, dst, :]
    diff  = pos_j - pos_i

    d_ij     = diff.norm(dim=-1, keepdim=True)
    theta_ij = torch.atan2(diff[..., 1], diff[..., 0]).unsqueeze(-1)

    d_4d     = d_ij.reshape(B, T, E, 1)
    theta_4d = theta_ij.reshape(B, T, E, 1)

    delta_d     = torch.zeros_like(d_4d)
    delta_theta = torch.zeros_like(theta_4d)
    delta_d[:, 1:] = d_4d[:, 1:] - d_4d[:, :-1]

    dtheta = theta_4d[:, 1:] - theta_4d[:, :-1]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    delta_theta[:, 1:] = dtheta

    d_ij        = d_4d.reshape(B * T, E, 1)
    theta_ij    = theta_4d.reshape(B * T, E, 1)
    delta_d     = delta_d.reshape(B * T, E, 1)
    delta_theta = delta_theta.reshape(B * T, E, 1)

    edge_attr = torch.cat([d_ij, theta_ij, delta_d, delta_theta], dim=-1)
    return edge_attr.reshape(B * T * E, 4)


def _dilated_shift(x, fold_div=8, mode='uni', dilations=(1, 2, 4, 8)):
    B, T, N, C = x.shape
    fold_total = C // fold_div
    num_groups = len(dilations)
    group_size = fold_total // num_groups

    out = x.clone()

    if mode == 'uni':
        for i, d in enumerate(dilations):
            ch_s, ch_e = i * group_size, (i + 1) * group_size
            if d < T:
                out[:, d:, :, ch_s:ch_e] = x[:, :-d, :, ch_s:ch_e]
    elif mode == 'bi':
        half = num_groups // 2
        for i in range(half):
            d = dilations[i]
            ch_s, ch_e = i * group_size, (i + 1) * group_size
            if d < T:
                out[:, d:, :, ch_s:ch_e] = x[:, :-d, :, ch_s:ch_e]
        for i in range(half, num_groups):
            d = dilations[i - half]
            ch_s, ch_e = i * group_size, (i + 1) * group_size
            if d < T:
                out[:, :-d, :, ch_s:ch_e] = x[:, d:, :, ch_s:ch_e]
    else:
        raise ValueError(f"tsm mode must be 'uni' or 'bi', got '{mode}'")

    return out


class GatedTSM(nn.Module):
    def __init__(self, channels, num_nodes, fold_div=8, mode='uni', dilations=(1, 2, 4, 8)):
        super().__init__()
        self.channels  = channels
        self.num_nodes = num_nodes
        self.fold_div  = fold_div
        self.mode      = mode
        self.dilations = dilations

        assert channels // fold_div > 0, \
            f"fold_total = channels({channels}) // fold_div({fold_div}) must be > 0"

        self.gate_proj = nn.Linear(channels, channels, bias=True)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, x_4d):
        B, T, N, C = x_4d.shape
        assert N == self.num_nodes, \
            f"GatedTSM expected num_nodes={self.num_nodes}, got {N}"

        x_flat  = x_4d.reshape(B * T * N, C)
        gate    = torch.sigmoid(self.gate_proj(x_flat))
        gate_4d = gate.reshape(B, T, N, C)

        shifted = _dilated_shift(x_4d, self.fold_div, self.mode, self.dilations)
        delta   = shifted - x_4d

        return x_4d + gate_4d * delta


class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x_4d):
        B, T, N, C = x_4d.shape
        x = x_4d.permute(0, 2, 3, 1)
        x = x.reshape(B * N, C, T)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = x.reshape(B, N, C, T)
        x = x.permute(0, 3, 1, 2)
        return self.norm(x_4d + x)


def expand_edge_index(edge_index, num_graphs, num_nodes, device):
    E = edge_index.size(1)
    offset = torch.arange(num_graphs, device=device) * num_nodes
    offset = offset.view(num_graphs, 1, 1)
    edge_index_exp = edge_index.unsqueeze(0).expand(num_graphs, -1, -1)
    edge_index_exp = edge_index_exp + offset
    edge_index_batch = edge_index_exp.permute(1, 0, 2).reshape(2, -1)
    return edge_index_batch


class GATv2(nn.Module):
    def __init__(self, config):
        super(GATv2, self).__init__()

        self.in_channels     = int(config.in_channels)
        self.hidden_channels = int(config.hidden_channels)
        self.out_channels    = int(config.out_channels)
        self.num_heads       = int(config.num_heads)
        self.dropout         = float(config.dropout)
        self.num_nodes       = int(config.num_nodes)

        assert self.hidden_channels % self.num_heads == 0, \
            "hidden_channels must be divisible by num_heads"
        self.head_dim = self.hidden_channels // self.num_heads

        self.num_gat_layers = int(getattr(config, 'num_gat_layers', 4))  # 4-hop coverage
        self.use_tsm        = bool(getattr(config, 'use_tsm', True))
        self.tsm_mode       = str(getattr(config, 'tsm_mode', 'uni'))
        self.tsm_fold_div   = int(getattr(config, 'tsm_fold_div', 8))
        self.tsm_dilations  = tuple(getattr(config, 'tsm_dilations', (1, 2, 4, 8)))

        fold_total = self.hidden_channels // self.tsm_fold_div
        assert fold_total % len(self.tsm_dilations) == 0, (
            f"fold_total ({fold_total}) must be divisible by "
            f"len(tsm_dilations) ({len(self.tsm_dilations)})"
        )

        self.use_causal_conv = bool(getattr(config, 'use_causal_conv', False))
        self.causal_kernel   = int(getattr(config, 'causal_kernel', 3))
        self.causal_dilation = int(getattr(config, 'causal_dilation', 1))

        self.edge_attr_dim = 4

        self.input_proj = nn.Linear(self.in_channels, self.hidden_channels)

        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=self.hidden_channels,
                out_channels=self.head_dim,
                heads=self.num_heads,
                concat=True,
                dropout=self.dropout,
                add_self_loops=False,
                edge_dim=self.edge_attr_dim,
            )
            for _ in range(self.num_gat_layers)
        ])

        if self.use_tsm:
            self.gated_tsm_layers = nn.ModuleList([
                GatedTSM(
                    channels=self.hidden_channels,
                    num_nodes=self.num_nodes,
                    fold_div=self.tsm_fold_div,
                    mode=self.tsm_mode,
                    dilations=self.tsm_dilations,
                )
                for _ in range(self.num_gat_layers)
            ])

        if self.use_causal_conv:
            self.causal_conv_layers = nn.ModuleList([
                CausalConv1d(
                    channels=self.hidden_channels,
                    kernel_size=self.causal_kernel,
                    dilation=self.causal_dilation,
                )
                for _ in range(self.num_gat_layers)
            ])

        edge_index = build_edge_index(KINECT_EDGES, self.num_nodes, bidirectional=True)
        self.register_buffer('edge_index', edge_index)

        self.joint_att = nn.Linear(self.hidden_channels, 1)
        self.classifier = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x):
        """
        Args:
            x : (B, T, C, N)
        Returns:
            logits : (B, T, num_classes)
        """
        B, T, C, N = x.shape

        x_4d_raw = x.permute(0, 1, 3, 2)           # (B, T, N, C)
        x = x_4d_raw.reshape(B * T * N, C)
        x = self.input_proj(x)                      # (B*T*N, hidden)

        edge_index_batch = expand_edge_index(
            self.edge_index, B * T, N, x.device
        )
        edge_attr = compute_edge_attr(x_4d_raw, self.edge_index)  # (B*T*E, 4)

        for i, gat in enumerate(self.gat_layers):
            x_res = x                               # ✅ residual

            if self.use_tsm:
                x_4d = x.reshape(B, T, N, self.hidden_channels)
                x_4d = self.gated_tsm_layers[i](x_4d)
                x    = x_4d.reshape(B * T * N, self.hidden_channels)

            x = gat(x, edge_index_batch, edge_attr=edge_attr)
            x = F.elu(x)
            x = x + x_res                           # ✅ residual add

            if self.use_causal_conv:
                x_4d = x.reshape(B, T, N, self.hidden_channels)
                x_4d = self.causal_conv_layers[i](x_4d)
                x    = x_4d.reshape(B * T * N, self.hidden_channels)

        # Joint-Adaptive Attention Pooling
        x   = x.reshape(B * T, N, self.hidden_channels)
        att = torch.softmax(self.joint_att(x), dim=1)
        x   = (x * att).sum(dim=1)                 # (B*T, hidden)

        x = self.classifier(x)

        return x.reshape(B, T, self.out_channels)