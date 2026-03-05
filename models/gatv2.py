import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# Kinect v2 25개 관절 스켈레톤 edge 정의
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
    """
    스켈레톤 edge 리스트를 torch_geometric 형식의 edge_index (2, E)로 변환.
    bidirectional=True이면 양방향 edge 추가.
    """
    src, dst = zip(*edges)
    src = list(src)
    dst = list(dst)

    if bidirectional:
        src, dst = src + dst, dst + src  # 양방향

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


class GATv2(nn.Module):
    def __init__(self, config):
        super(GATv2, self).__init__()
        self.config = config
        self.in_channels = int(config.in_channels)
        self.hidden_channels = int(config.hidden_channels)
        self.out_channels = int(config.out_channels)
        self.num_heads = int(config.num_heads)
        self.dropout = float(config.dropout)
        self.num_nodes = int(config.num_nodes)

        assert self.hidden_channels % self.num_heads == 0
        self.head_dim = self.hidden_channels // self.num_heads

        # torch_geometric GATv2Conv 사용
        # concat=True → 출력 크기: num_heads * head_dim = hidden_channels
        self.gat = GATv2Conv(
            in_channels=self.in_channels,
            out_channels=self.head_dim,
            heads=self.num_heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )

        # 스켈레톤 edge_index 등록 (고정, 학습 X)
        edge_index = build_edge_index(KINECT_EDGES, self.num_nodes, bidirectional=True)
        self.register_buffer('edge_index', edge_index)

        # classifier: mean pooling 후 분류
        self.classifier = nn.Linear(
            self.hidden_channels,
            self.out_channels
        )

    def forward(self, x):
        # x: (B, T, C, N)
        B, T, C, N = x.shape

        # GATv2Conv는 (num_nodes, in_channels) 입력
        x = x.permute(0, 1, 3, 2)   # (B, T, N, C)
        x = x.reshape(B * T * N, C)  # (B*T*N, C)

        # edge_index를 B*T개 그래프 배치에 맞게 확장
        # torch_geometric 배치 방식: 각 그래프마다 노드 offset 추가
        num_graphs = B * T
        edge_index = self.edge_index  # (2, E)
        E = edge_index.size(1)

        # offset: [0, N, 2N, ..., (B*T-1)*N]
        offset = torch.arange(num_graphs, device=x.device) * N  # (B*T,)
        offset = offset.unsqueeze(1).expand(-1, E)               # (B*T, E)
        edge_index_batch = edge_index.unsqueeze(0).expand(num_graphs, -1, -1)  # (B*T, 2, E)
        edge_index_batch = edge_index_batch + offset.unsqueeze(1)              # (B*T, 2, E)
        edge_index_batch = edge_index_batch.transpose(0, 1).reshape(2, -1)  # (2, B*T*E)

        # GATv2Conv forward
        out = self.gat(x, edge_index_batch)  # (B*T*N, hidden_channels)
        out = F.elu(out)                      # ELU 활성화

        # reshape back
        out = out.reshape(B * T, N, self.hidden_channels)  # (B*T, N, hidden)

        # mean pooling over nodes
        out = out.mean(dim=1)                               # (B*T, hidden)

        # classification
        out = self.classifier(out)                          # (B*T, num_classes)

        return out.reshape(B, T, self.out_channels)