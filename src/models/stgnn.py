"""
Spatiotemporal GNN model for crime prediction.

Architecture:
  For each time step t in [0, T):
      H_t = GNN(X_t, A)                →  (B, N, H_gnn)
  Stack over T:
      H_seq = stack(H_0 .. H_{T-1})    →  (B, T, N, H_gnn)
  Reshape to per-node sequences:
      (B*N, T, H_gnn)
  Temporal encoder (LSTM / GRU / MHA):
      → (B*N, H_temporal)
  Reshape + Linear:
      → (B, N)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    DROPOUT,
    GNN_HIDDEN_DIM,
    MHA_NUM_HEADS,
    NUM_FEATURES,
    NUM_GNN_LAYERS,
    NUM_REGIONS,
    TEMPORAL_HIDDEN_DIM,
)
from src.models.temporal_modules import build_temporal_encoder


class GCNLayer(nn.Module):
    """Single GCN layer: H' = A_norm @ H @ W + b."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(A, H)
        return self.linear(support)


class GNNBlock(nn.Module):
    """Multi-layer GCN with ReLU + dropout between layers."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.append(GCNLayer(d_in, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = X
        for i, layer in enumerate(self.layers):
            H = layer(H, A)
            if i < len(self.layers) - 1:
                H = F.relu(H)
                H = self.dropout(H)
        return H


class STGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network with configurable temporal encoder.

    Parameters
    ----------
    temporal_type : 'lstm' | 'gru' | 'mha'
    """

    def __init__(
        self,
        temporal_type: str = "lstm",
        num_features: int = NUM_FEATURES,
        num_regions: int = NUM_REGIONS,
        gnn_hidden: int = GNN_HIDDEN_DIM,
        temporal_hidden: int = TEMPORAL_HIDDEN_DIM,
        num_gnn_layers: int = NUM_GNN_LAYERS,
        dropout: float = DROPOUT,
        mha_num_heads: int = MHA_NUM_HEADS,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.gnn_hidden = gnn_hidden
        self.temporal_type = temporal_type

        self.gnn = GNNBlock(num_features, gnn_hidden, num_gnn_layers, dropout)

        # Build temporal encoder
        extra_kwargs = {}
        if temporal_type == "mha":
            extra_kwargs["num_heads"] = mha_num_heads
        self.temporal = build_temporal_encoder(
            temporal_type, input_dim=gnn_hidden,
            hidden_dim=temporal_hidden, dropout=dropout, **extra_kwargs,
        )
        self.fc = nn.Linear(self.temporal.output_dim, 1)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X : (B, T, N, F)
        A : (N, N)
        Returns: (B, N)
        """
        B, T, N, F = X.shape

        # Per-timestep GNN
        h_list = []
        for t in range(T):
            h_list.append(self.gnn(X[:, t, :, :], A))
        H = torch.stack(h_list, dim=1)  # (B, T, N, H_gnn)

        # Reshape to per-node sequences → (B*N, T, H_gnn)
        H = H.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.gnn_hidden)

        # Temporal encoding → (B*N, H_temporal)
        h_temporal = self.temporal(H)

        # Predict → (B, N)
        out = self.fc(h_temporal).squeeze(-1).view(B, N)
        return out