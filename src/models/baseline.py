"""
Baseline model for crime prediction — temporal-only, no spatial structure.

Architecture:
  X (B, T, N, F) → flatten per-step → (B, T, N*F) → temporal encoder → (B, H) → Linear → (B, N)

No adjacency matrix is used. The model learns purely temporal dynamics
over flattened region features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import (
    DROPOUT,
    MHA_NUM_HEADS,
    NUM_FEATURES,
    NUM_REGIONS,
    TEMPORAL_HIDDEN_DIM,
)
from src.models.temporal_modules import build_temporal_encoder


class BaselineModel(nn.Module):
    """
    Temporal-only baseline with configurable encoder.

    Parameters
    ----------
    temporal_type : 'lstm' | 'gru' | 'mha'
    """

    def __init__(
        self,
        temporal_type: str = "lstm",
        num_features: int = NUM_FEATURES,
        num_regions: int = NUM_REGIONS,
        temporal_hidden: int = TEMPORAL_HIDDEN_DIM,
        dropout: float = DROPOUT,
        mha_num_heads: int = MHA_NUM_HEADS,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.temporal_type = temporal_type
        self.flat_dim = num_regions * num_features  # N * F

        extra_kwargs = {}
        if temporal_type == "mha":
            extra_kwargs["num_heads"] = mha_num_heads

        self.temporal = build_temporal_encoder(
            temporal_type,
            input_dim=self.flat_dim,
            hidden_dim=temporal_hidden,
            dropout=dropout,
            **extra_kwargs,
        )
        self.fc = nn.Linear(self.temporal.output_dim, num_regions)

    def forward(self, X: torch.Tensor, A: torch.Tensor = None) -> torch.Tensor:
        """
        X : (B, T, N, F)
        A : ignored (kept for interface compatibility)
        Returns: (B, N)
        """
        B, T, N, F = X.shape

        # Flatten spatial dimension: (B, T, N*F)
        X_flat = X.reshape(B, T, N * F)

        # Temporal encoding → (B, H)
        h = self.temporal(X_flat)

        # Predict all regions → (B, N)
        return self.fc(h)