"""
Spatiotemporal GNN model for crime prediction.

Architecture:
  For each time step t in [0, T):
      H_t = GNN(X_t, A)          →  (B, N, H)
  Stack over T:
      H_seq = stack(H_0 .. H_{T-1})  →  (B, T, N, H)
  Reshape to per-node sequences:
      (B*N, T, H)
  LSTM:
      → (B*N, H_lstm)
  Reshape + Linear:
      → (B, N, 1) → squeeze → (B, N)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    DROPOUT,
    GNN_HIDDEN_DIM,
    LSTM_HIDDEN_DIM,
    NUM_FEATURES,
    NUM_GNN_LAYERS,
    NUM_REGIONS,
)


class GCNLayer(nn.Module):
    """Single Graph Convolutional layer: H' = σ(A_norm · H · W)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        H : (B, N, F_in)
        A : (N, N) — pre-normalized adjacency
        """
        # A @ H → spatial aggregation, then linear projection
        support = torch.matmul(A, H)  # (B, N, F_in)
        out = self.linear(support)     # (B, N, F_out)
        return out


class GNNBlock(nn.Module):
    """Multi-layer GCN with ReLU + dropout between layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            layers.append(GCNLayer(d_in, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X : (B, N, F)
        A : (N, N)
        Returns: (B, N, hidden_dim)
        """
        H = X
        for i, layer in enumerate(self.layers):
            H = layer(H, A)
            if i < len(self.layers) - 1:
                H = F.relu(H)
                H = self.dropout(H)
        return H


class STGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network.

    Per-timestep GCN → temporal LSTM → linear prediction.
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_regions: int = NUM_REGIONS,
        gnn_hidden: int = GNN_HIDDEN_DIM,
        lstm_hidden: int = LSTM_HIDDEN_DIM,
        num_gnn_layers: int = NUM_GNN_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.gnn_hidden = gnn_hidden

        self.gnn = GNNBlock(num_features, gnn_hidden, num_gnn_layers, dropout)
        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (B, T, N, F) — input feature sequences
        A : (N, N) — normalized adjacency matrix

        Returns
        -------
        Y_hat : (B, N) — predicted crime_count for next day
        """
        B, T, N, F = X.shape

        # 1. Per-timestep GNN
        h_list = []
        for t in range(T):
            h_t = self.gnn(X[:, t, :, :], A)  # (B, N, H)
            h_list.append(h_t)
        H = torch.stack(h_list, dim=1)  # (B, T, N, H)

        # 2. Reshape to per-node sequences: (B*N, T, H)
        H = H.permute(0, 2, 1, 3).contiguous()  # (B, N, T, H)
        H = H.view(B * N, T, self.gnn_hidden)

        # 3. LSTM — take last hidden state
        lstm_out, _ = self.lstm(H)         # (B*N, T, H_lstm)
        lstm_last = lstm_out[:, -1, :]     # (B*N, H_lstm)
        lstm_last = self.dropout(lstm_last)

        # 4. Predict
        out = self.fc(lstm_last).squeeze(-1)  # (B*N,)
        out = out.view(B, N)                  # (B, N)
        return out