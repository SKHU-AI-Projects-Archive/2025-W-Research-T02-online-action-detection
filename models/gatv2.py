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


def _dilated_shift(x, fold_div=8, mode='uni', dilations=(1, 2, 4, 8)):
    """
    Dilated Temporal Shift (no parameters, internal helper).

    Args:
        x         : (B, T, N, C)
        fold_div  : total fold = C // fold_div channels will be shifted
        mode      : 'uni' → past→present only  (online-safe)
                    'bi'  → past↔future        (offline only)
        dilations : dilation values; fold channels split equally across groups

    Returns:
        shifted : (B, T, N, C)
    """
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
    """
    Node-wise Gated Temporal Shift Module.

        gate   = sigmoid(W_g @ x + b)
        output = x + gate ⊙ (shift(x) - x)

    Also returns the temporal delta (shifted - x) for use in MSE.

    bias=-2.0 → initial gate ≈ 0.12 → near-identity at start
    """

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
        """
        Args:
            x_4d : (B, T, N, C)
        Returns:
            out   : (B, T, N, C)  — gated output
            delta : (B, T, N, C)  — temporal delta (shifted - x), used by MSE
        """
        B, T, N, C = x_4d.shape
        assert N == self.num_nodes, \
            f"GatedTSM expected num_nodes={self.num_nodes}, got {N}"

        x_flat  = x_4d.reshape(B * T * N, C)
        gate    = torch.sigmoid(self.gate_proj(x_flat))
        gate_4d = gate.reshape(B, T, N, C)

        shifted = _dilated_shift(x_4d, self.fold_div, self.mode, self.dilations)
        delta   = shifted - x_4d                          # temporal difference

        return x_4d + gate_4d * delta, delta


class MotionSqueezedExcitation(nn.Module):
    """
    Motion-Squeezed Excitation (MSE).

    Reuses the temporal delta from GatedTSM to compute per-node motion scores,
    then uses these scores to re-weight GAT output features.

    Intuition:
      - GatedTSM decides "how much past info to absorb" per channel
      - MSE asks "which nodes are currently moving?" and amplifies them
      - Nodes with large motion (e.g. wrists, ankles) get higher weight
      - Static nodes (e.g. spine) are naturally suppressed

    Flow:
        delta   : (B, T, N, C)  — |shifted - x| from GatedTSM
        score   : (B, T, N, 1)  — per-node motion magnitude (mean over C)
        weight  : (B, T, N, 1)  — sigmoid-gated score → [0, 1]
        output  : x_gat * (1 + weight)  — residual excitation

    Using (1 + weight) instead of weight alone ensures:
      - weight=0 → output = x_gat  (identity, no suppression)
      - weight=1 → output = 2 * x_gat (double excitation for active nodes)

    Args:
        channels : int — feature dim C (= hidden_channels)
    """

    def __init__(self, channels):
        super().__init__()
        # Lightweight gate: scalar per node from motion score
        # bias=-1.0 → initial weight ≈ sigmoid(-1) ≈ 0.27 → gentle start
        self.motion_gate = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.motion_gate.weight, 1.0)
        nn.init.constant_(self.motion_gate.bias, -1.0)

    def forward(self, x_gat, delta):
        """
        Args:
            x_gat : (B, T, N, C)  — GAT output features
            delta : (B, T, N, C)  — temporal delta from GatedTSM

        Returns:
            out   : (B, T, N, C)  — motion-excited features
        """
        # Per-node motion magnitude: mean of |delta| over channels
        motion_score = delta.abs().mean(dim=-1, keepdim=True)  # (B, T, N, 1)

        # Learnable gate on motion score
        weight = torch.sigmoid(self.motion_gate(motion_score))  # (B, T, N, 1)

        # Residual excitation: active nodes amplified, static nodes unchanged
        return x_gat * (1.0 + weight)


class CausalConv1d(nn.Module):
    """
    Node-wise Causal Temporal Convolution (online-safe).

    Applied after each GATv2Conv+MSE along the T dimension.
    Uses left-only padding → strictly causal, no future leakage.
    Depthwise conv for efficiency. LayerNorm + residual for stability.

    Receptive field = 1 + (kernel_size - 1) * dilation frames into the past.

    Args:
        channels    : int — feature dim C
        kernel_size : int — temporal kernel size (default 3)
        dilation    : int — temporal dilation (default 1)
    """

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
        """
        Args:
            x_4d : (B, T, N, C)
        Returns:
            out  : (B, T, N, C)
        """
        B, T, N, C = x_4d.shape

        x = x_4d.permute(0, 2, 3, 1)       # (B, N, C, T)
        x = x.reshape(B * N, C, T)         # (B*N, C, T)
        x = F.pad(x, (self.pad, 0))        # causal: left only
        x = self.conv(x)                   # (B*N, C, T)
        x = x.reshape(B, N, C, T)         # (B, N, C, T)
        x = x.permute(0, 3, 1, 2)         # (B, T, N, C)

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
    """
    GATv2 with GatedTSM, MotionSqueezedExcitation, and optional CausalConv1d.

    Forward flow per layer:
        GatedTSM → GATv2Conv → ELU → MSE → (CausalConv1d)

    GatedTSM produces temporal delta which is passed to MSE,
    allowing motion-aware node excitation without extra input.

    Config keys
    -----------
    in_channels      : int   — input joint feature dim (e.g. 3 for xyz)
    hidden_channels  : int   — internal feature dim (must be divisible by num_heads)
    out_channels     : int   — number of action classes
    num_heads        : int   — GAT attention heads
    dropout          : float
    num_nodes        : int   — skeleton joints (25 for Kinect)
    num_gat_layers   : int   — number of stacked GATv2Conv layers (default 2)
    use_tsm          : bool  — whether to apply GatedTSM (default True)
    tsm_mode         : str   — 'uni' or 'bi' (default 'uni')
    tsm_fold_div     : int   — fold = hidden // tsm_fold_div (default 8)
    tsm_dilations    : tuple — dilation list e.g. (1,2,4,16)
    use_mse          : bool  — whether to apply MSE after GAT (default True)
    use_causal_conv  : bool  — whether to apply CausalConv1d after MSE (default False)
    causal_kernel    : int   — kernel size for CausalConv1d (default 3)
    causal_dilation  : int   — dilation for CausalConv1d (default 1)
    """

    def __init__(self, config):
        super(GATv2, self).__init__()

        # ── basic dims ──────────────────────────────────────────────────────────
        self.in_channels     = int(config.in_channels)
        self.hidden_channels = int(config.hidden_channels)
        self.out_channels    = int(config.out_channels)
        self.num_heads       = int(config.num_heads)
        self.dropout         = float(config.dropout)
        self.num_nodes       = int(config.num_nodes)

        assert self.hidden_channels % self.num_heads == 0, \
            "hidden_channels must be divisible by num_heads"
        self.head_dim = self.hidden_channels // self.num_heads

        # ── TSM config ──────────────────────────────────────────────────────────
        self.num_gat_layers = int(getattr(config, 'num_gat_layers', 2))
        self.use_tsm        = bool(getattr(config, 'use_tsm', True))
        self.tsm_mode       = str(getattr(config, 'tsm_mode', 'uni'))
        self.tsm_fold_div   = int(getattr(config, 'tsm_fold_div', 8))
        self.tsm_dilations  = tuple(getattr(config, 'tsm_dilations', (1, 2, 4, 8)))

        fold_total = self.hidden_channels // self.tsm_fold_div
        assert fold_total % len(self.tsm_dilations) == 0, (
            f"fold_total ({fold_total}) must be divisible by "
            f"len(tsm_dilations) ({len(self.tsm_dilations)})"
        )

        # ── MSE config ──────────────────────────────────────────────────────────
        self.use_mse = bool(getattr(config, 'use_mse', True))

        # ── CausalConv config ───────────────────────────────────────────────────
        self.use_causal_conv = bool(getattr(config, 'use_causal_conv', False))
        self.causal_kernel   = int(getattr(config, 'causal_kernel', 3))
        self.causal_dilation = int(getattr(config, 'causal_dilation', 1))

        # ── layers ──────────────────────────────────────────────────────────────
        self.input_proj = nn.Linear(self.in_channels, self.hidden_channels)

        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=self.hidden_channels,
                out_channels=self.head_dim,
                heads=self.num_heads,
                concat=True,
                dropout=self.dropout,
                add_self_loops=True,
            )
            for _ in range(self.num_gat_layers)
        ])

        # GatedTSM: one per GAT layer
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

        # MSE: one per GAT layer (applied after GAT+ELU)
        if self.use_mse:
            self.mse_layers = nn.ModuleList([
                MotionSqueezedExcitation(channels=self.hidden_channels)
                for _ in range(self.num_gat_layers)
            ])

        # CausalConv1d: one per GAT layer (applied after MSE)
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

        self.classifier = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x):
        """
        Args:
            x : (B, T, C, N)
        Returns:
            logits : (B, T, num_classes)
        """
        B, T, C, N = x.shape

        x = x.permute(0, 1, 3, 2)              # (B, T, N, C)
        x = x.reshape(B * T * N, C)
        x = self.input_proj(x)                 # (B*T*N, hidden)

        edge_index_batch = expand_edge_index(
            self.edge_index, B * T, N, x.device
        )

        delta = None  # temporal delta from GatedTSM, passed to MSE

        for i, gat in enumerate(self.gat_layers):

            # 1) GatedTSM: temporal shift → also produces delta for MSE
            if self.use_tsm:
                x_4d = x.reshape(B, T, N, self.hidden_channels)
                x_4d, delta = self.gated_tsm_layers[i](x_4d)
                x = x_4d.reshape(B * T * N, self.hidden_channels)

            # 2) GATv2Conv: spatial message passing
            x = gat(x, edge_index_batch)        # (B*T*N, hidden)
            x = F.elu(x)

            # 3) MSE: motion-aware node excitation
            if self.use_mse and delta is not None:
                x_4d = x.reshape(B, T, N, self.hidden_channels)
                x_4d = self.mse_layers[i](x_4d, delta)
                x = x_4d.reshape(B * T * N, self.hidden_channels)

            # 4) CausalConv1d: temporal conv (optional)
            if self.use_causal_conv:
                x_4d = x.reshape(B, T, N, self.hidden_channels)
                x_4d = self.causal_conv_layers[i](x_4d)
                x = x_4d.reshape(B * T * N, self.hidden_channels)

        # pooling & classification
        x = x.reshape(B * T, N, self.hidden_channels)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x.reshape(B, T, self.out_channels)