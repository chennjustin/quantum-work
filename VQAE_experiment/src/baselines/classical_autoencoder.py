"""Classical autoencoder baseline (requires PyTorch)."""

from __future__ import annotations

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the classical autoencoder baseline. "
            "Install with: python -m pip install torch"
        ) from exc


def train_classical_autoencoder(
    X_train: np.ndarray,
    latent_dim: int = 2,
    epochs: int = 100,
    lr: float = 1e-2,
    seed: int = 7,
):
    """Train a lightweight PyTorch autoencoder on normal data."""

    torch, nn = _require_torch()
    torch.manual_seed(seed)

    n_features = X_train.shape[1]
    hidden = max(latent_dim * 2, 4)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_features),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = AE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(X_tensor)
        loss = loss_fn(recon, X_tensor)
        loss.backward()
        optimizer.step()

    return model


def reconstruction_anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Return MSE reconstruction error as anomaly score."""

    torch, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        recon = model(X_tensor)
        mse = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
    return mse.astype(float)
