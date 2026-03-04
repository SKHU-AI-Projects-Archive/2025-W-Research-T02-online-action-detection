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
        self.num_nodes = config.num_nodes

        assert self.hidden_channels % self.num_heads == 0
        self.head_dim = self.hidden_channels // self.num_heads

        # GATv2 projection
        self.W = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.W2 = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.a = nn.Linear(self.head_dim, 1, bias=False)

        self.dropout_layer = nn.Dropout(self.dropout)

        # classifier (node flatten)
        self.classifier = nn.Linear(self.hidden_channels * self.num_nodes,
                                    self.out_channels)

    def forward(self, x):
        # x: (B, T, C, N)
        B, T, C, N = x.shape

        x = x.reshape(B * T, C, N)        # (B*T, C, N)
        x = x.permute(0, 2, 1)            # (B*T, N, C)

        # --- GATv2 ---
        Wx  = self.W(x).view(B*T, N, self.num_heads, self.head_dim)
        Wx2 = self.W2(x).view(B*T, N, self.num_heads, self.head_dim)

        Wx_i = Wx.unsqueeze(2)            # (B*T, N, 1, H, D)
        Wx_j = Wx2.unsqueeze(1)           # (B*T, 1, N, H, D)

        e = self.a(F.leaky_relu(Wx_i + Wx_j, 0.2)).squeeze(-1)
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout_layer(alpha)

        out = torch.einsum('bijn,bjnh->binh', alpha, Wx2)
        out = out.reshape(B*T, N, self.hidden_channels)

        # flatten nodes
        out = out.reshape(B*T, -1)

        # classification
        out = self.classifier(out)

        return out.reshape(B, T, self.out_channels)