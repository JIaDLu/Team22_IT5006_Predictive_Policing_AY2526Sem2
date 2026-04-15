"""
Shared temporal sequence encoders for both baseline and ST-GNN models.

Each module follows a unified contract:
    Input:  (B, T, D)   — batch of sequences
    Output: (B, D_out)   — one vector per sequence

Supported: LSTM, GRU, MultiHeadAttention (with mean/attention pooling).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """LSTM encoder — returns last time-step hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, hidden_dim)"""
        out, _ = self.lstm(x)        # (B, T, H)
        return self.dropout(out[:, -1, :])  # last step


class GRUEncoder(nn.Module):
    """GRU encoder — returns last time-step hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, hidden_dim)"""
        out, _ = self.gru(x)
        return self.dropout(out[:, -1, :])


class MHAEncoder(nn.Module):
    """
    Multi-Head Self-Attention encoder with positional encoding.

    Pooling modes:
      - 'mean':      average over all time steps
      - 'attention': learnable attention pooling over time
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "attention",
        max_len: int = 64,
    ):
        super().__init__()
        self.output_dim = hidden_dim
        self.pooling = pooling

        # Project input to hidden_dim (required for multi-head attention)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (fixed sinusoidal)
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Attention pooling
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_in) → (B, hidden_dim)"""
        B, T, _ = x.shape

        # Project + positional encoding
        h = self.input_proj(x) + self.pe[:, :T, :]  # (B, T, H)

        # Self-attention
        attn_out, _ = self.attn(h, h, h)  # (B, T, H)
        h = self.norm(h + attn_out)        # residual + layer norm

        # Pooling
        if self.pooling == "mean":
            out = h.mean(dim=1)  # (B, H)
        else:
            # Learnable attention pooling
            query = self.pool_query.expand(B, -1, -1)      # (B, 1, H)
            pooled, _ = self.attn(query, h, h)              # (B, 1, H)
            out = pooled.squeeze(1)                         # (B, H)

        return self.dropout(out)


# ============================================================
# Factory
# ============================================================
TEMPORAL_REGISTRY = {
    "lstm": LSTMEncoder,
    "gru": GRUEncoder,
    "mha": MHAEncoder,
}


def build_temporal_encoder(
    name: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """
    Build a temporal encoder by name.

    Parameters
    ----------
    name : 'lstm' | 'gru' | 'mha'
    """
    name = name.lower()
    if name not in TEMPORAL_REGISTRY:
        raise ValueError(
            f"Unknown temporal encoder '{name}'. "
            f"Choose from {list(TEMPORAL_REGISTRY.keys())}"
        )
    cls = TEMPORAL_REGISTRY[name]
    return cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)