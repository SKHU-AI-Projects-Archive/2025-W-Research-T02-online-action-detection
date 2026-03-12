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
                    'bi'  → past↔present       (offline only)
        dilations : dilation values; fold channels split equally across groups

    Returns:
        shifted: (B, T, N, C)
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
                out[:, d:,  :, ch_s:ch_e] = x[:, :-d, :, ch_s:ch_e]
    elif mode == 'bi':
        half = num_groups // 2
        for i in range(half):
            d = dilations[i]
            ch_s, ch_e = i * group_size, (i + 1) * group_size
            if d < T:
                out[:, d:,  :, ch_s:ch_e] = x[:, :-d, :, ch_s:ch_e]   # past→present
        for i in range(half, num_groups):
            d = dilations[i - half]
            ch_s, ch_e = i * group_size, (i + 1) * group_size
            if d < T:
                out[:, :-d, :, ch_s:ch_e] = x[:, d:,  :, ch_s:ch_e]   # future→present
    else:
        raise ValueError(f"tsm mode must be 'uni' or 'bi', got '{mode}'")

    return out


class GatedTSM(nn.Module):
    """
    Node-wise Gated Temporal Shift Module.

    Each node independently learns how much temporal context to absorb:

        gate   = sigmoid(W_g @ x + b)      W_g: (C, C), b: (C,)
        output = x + gate ⊙ (shift(x) - x)

    This is a clean delta-residual form:
      - gate = 0  →  output = x          (ignore temporal shift, keep current)
      - gate = 1  →  output = shift(x)   (fully replace with shifted features)

    Unlike `x + gate * shift(x)`, this avoids double-counting x in unshifted
    channels (where shift(x) == x, causing x*(1+gate) scaling artifacts).

    Gate initialization:
      bias = -2.0  →  sigmoid(-2) ≈ 0.12 at init
      → model starts near identity, gradually opens gates during training

    Args:
        channels  : int   — feature dim C (= hidden_channels)
        num_nodes : int   — expected N; checked in forward for safety
        fold_div  : int   — fold = channels // fold_div shifted channels
        mode      : str   — 'uni' (online-safe) or 'bi' (offline only)
        dilations : tuple — dilation distances for each fold group
    """

    def __init__(self, channels, num_nodes, fold_div=8, mode='uni', dilations=(1, 2, 4, 8)):
        super().__init__()
        self.channels  = channels
        self.num_nodes = num_nodes
        self.fold_div  = fold_div
        self.mode      = mode
        self.dilations = dilations

        # Sanity check: shifted channels must be > 0
        assert channels // fold_div > 0, \
            f"fold_total = channels({channels}) // fold_div({fold_div}) must be > 0"

        # Gate projection: (C → C) with bias
        # bias=-2.0 → initial gate ≈ sigmoid(-2) ≈ 0.12 → near-identity at start
        self.gate_proj = nn.Linear(channels, channels, bias=True)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, x_4d):
        """
        Args:
            x_4d : (B, T, N, C)

        Returns:
            out  : (B, T, N, C)
        """
        B, T, N, C = x_4d.shape
        assert N == self.num_nodes, \
            f"GatedTSM expected num_nodes={self.num_nodes}, got {N}"

        # ── gate: per-node (each node sees its own feature vector) ─────────────
        x_flat  = x_4d.reshape(B * T * N, C)
        gate    = torch.sigmoid(self.gate_proj(x_flat))     # (B*T*N, C)
        gate_4d = gate.reshape(B, T, N, C)                  # (B, T, N, C)

        # ── dilated shift (parameter-free) ────────────────────────────────────
        shifted = _dilated_shift(x_4d, self.fold_div, self.mode, self.dilations)

        # ── delta-residual: x + gate ⊙ (shift - x) ───────────────────────────
        # Avoids double-counting x in unshifted channels
        # gate=0 → keep x, gate=1 → fully adopt shifted features
        return x_4d + gate_4d * (shifted - x_4d)


def expand_edge_index(edge_index, num_graphs, num_nodes, device):
    """
    Tile a single graph's edge_index across a batch of num_graphs graphs.

    Args:
        edge_index : (2, E)  — single graph edge index
        num_graphs : int     — B*T
        num_nodes  : int     — N (nodes per graph)
        device     : torch.device

    Returns:
        edge_index_batch : (2, num_graphs * E)
    """
    E = edge_index.size(1)
    # offset[i] = i * N, shape (num_graphs, 1, 1) for broadcasting
    offset = torch.arange(num_graphs, device=device) * num_nodes  # (num_graphs,)
    offset = offset.view(num_graphs, 1, 1)                         # (num_graphs, 1, 1)
    edge_index_exp = edge_index.unsqueeze(0).expand(num_graphs, -1, -1)  # (num_graphs, 2, E)
    edge_index_exp = edge_index_exp + offset                              # (num_graphs, 2, E)
    edge_index_batch = edge_index_exp.permute(1, 0, 2).reshape(2, -1)    # (2, num_graphs*E)
    return edge_index_batch


class GATv2(nn.Module):
    """
    GATv2 with optional Dilated Temporal Shift Module (TSM).

    Config keys
    -----------
    in_channels     : int   — input joint feature dim (e.g. 3 for xyz)
    hidden_channels : int   — internal feature dim (must be divisible by num_heads)
    out_channels    : int   — number of action classes
    num_heads       : int   — GAT attention heads
    dropout         : float
    num_nodes       : int   — skeleton joints (25 for Kinect)
    num_gat_layers  : int   — number of stacked GATv2Conv layers (default 1)
    use_tsm         : bool  — whether to apply TSM before each GAT layer (default True)
    tsm_mode        : str   — 'uni' (online-safe) or 'bi' (offline) (default 'uni')
    tsm_fold_div    : int   — fold = hidden // tsm_fold_div (default 8)
    tsm_dilations   : str   — comma-separated dilation list e.g. '1,2,4,8' (default '1,2,4,8')
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
        # Configuration is parsed via ast.literal_eval (see utils/env.py),
        # so ini values are already Python types:
        #   num_gat_layers = 2       → int
        #   use_tsm = True           → bool
        #   tsm_mode = 'uni'         → str  (quotes required in ini!)
        #   tsm_fold_div = 8         → int
        #   tsm_dilations = 1,2,4,8  → tuple (ast parses comma-expr as tuple)
        self.num_gat_layers = int(getattr(config, 'num_gat_layers', 1))
        self.use_tsm        = bool(getattr(config, 'use_tsm', True))
        self.tsm_mode       = str(getattr(config, 'tsm_mode', 'uni'))
        self.tsm_fold_div   = int(getattr(config, 'tsm_fold_div', 8))
        self.tsm_dilations  = tuple(getattr(config, 'tsm_dilations', (1, 2, 4, 8)))

        # sanity check: fold channels must be divisible by number of dilation groups
        fold_total = self.hidden_channels // self.tsm_fold_div
        assert fold_total % len(self.tsm_dilations) == 0, (
            f"fold_total ({fold_total}) must be divisible by "
            f"len(tsm_dilations) ({len(self.tsm_dilations)})"
        )

        # ── layers ──────────────────────────────────────────────────────────────
        # Linear projection: in_channels → hidden_channels
        # Needed to have enough channels for TSM fold groups
        self.input_proj = nn.Linear(self.in_channels, self.hidden_channels)

        # Stack of GATv2Conv layers
        # First layer: hidden → hidden (concat=True, so out = head_dim * num_heads = hidden)
        # Subsequent layers: same hidden → hidden
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=self.hidden_channels,
                out_channels=self.head_dim,
                heads=self.num_heads,
                concat=True,            # output dim = head_dim * num_heads = hidden_channels
                dropout=self.dropout,
                add_self_loops=True,
            )
            for _ in range(self.num_gat_layers)
        ])

        # ── Gated TSM modules (one per GAT layer) ───────────────────────────────
        # Each GAT layer gets its own GatedTSM so gates can specialize per depth
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

        # Edge index (single graph, registered as buffer → moved with .to(device))
        edge_index = build_edge_index(KINECT_EDGES, self.num_nodes, bidirectional=True)
        self.register_buffer('edge_index', edge_index)

        # Classifier: mean-pooled node features → class logits
        self.classifier = nn.Linear(self.hidden_channels, self.out_channels)

    # ── forward ─────────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x : (B, T, C, N)   — batch of skeleton sequences

        Returns:
            logits : (B, T, num_classes)
        """
        B, T, C, N = x.shape

        # (B, T, C, N) → (B, T, N, C) → (B*T*N, C)
        x = x.permute(0, 1, 3, 2)              # (B, T, N, C)
        x = x.reshape(B * T * N, C)            # (B*T*N, C)

        # Linear projection: C → hidden_channels
        x = self.input_proj(x)                 # (B*T*N, hidden)

        # Build batched edge index once (shared across all GAT layers)
        edge_index_batch = expand_edge_index(
            self.edge_index, B * T, N, x.device
        )                                       # (2, B*T*E)

        # ── GAT layers with Gated TSM ────────────────────────────────────────
        for i, gat in enumerate(self.gat_layers):
            if self.use_tsm:
                # Restore temporal axis for GatedTSM, then flatten back
                # IMPORTANT: T must NOT be merged with B or N before TSM
                x_4d = x.reshape(B, T, N, self.hidden_channels)    # (B, T, N, hidden)
                x_4d = self.gated_tsm_layers[i](x_4d)              # (B, T, N, hidden)
                x    = x_4d.reshape(B * T * N, self.hidden_channels)   # (B*T*N, hidden)

            x = gat(x, edge_index_batch)        # (B*T*N, hidden)
            x = F.elu(x)                        # (B*T*N, hidden)

        # ── pooling & classification ────────────────────────────────────────────
        x = x.reshape(B * T, N, self.hidden_channels)
        x = x.mean(dim=1)                       # mean pool over nodes: (B*T, hidden)
        x = self.classifier(x)                  # (B*T, num_classes)

        return x.reshape(B, T, self.out_channels)