"""
Training loop with validation-based early stopping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, LEARNING_RATE, NUM_EPOCHS, PATIENCE, WEIGHT_DECAY
from src.models.stgnn import STGNN


def train_model(
    model: STGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj: torch.Tensor,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    device: str = DEVICE,
    save_path: str = "checkpoints/best_model.pt",
) -> dict:
    """
    Train the STGNN model.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss' (per-epoch lists)
    """
    model = model.to(device)
    adj = adj.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_hat = model(X_batch, adj)
            loss = criterion(Y_hat, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_hat = model(X_batch, adj)
                loss = criterion(Y_hat, Y_batch)
                val_losses.append(loss.item())

        epoch_train = np.mean(train_losses)
        epoch_val = np.mean(val_losses)
        history["train_loss"].append(epoch_train)
        history["val_loss"].append(epoch_val)

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"train_loss={epoch_train:.4f}  val_loss={epoch_val:.4f}",
            end="",
        )

        # ---- Early stopping ----
        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print("  ★ saved", end="")
        else:
            epochs_no_improve += 1

        print()

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"Best val_loss = {best_val_loss:.4f}")
    return history