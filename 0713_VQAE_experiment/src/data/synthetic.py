"""Synthetic toy datasets (ideal baseline)."""

from __future__ import annotations

import numpy as np

from src.config import ToyDataConfig


def make_normal_angles(count: int, config: ToyDataConfig, rng: np.random.Generator) -> np.ndarray:
    """Normal samples: shared rotation angle per qubit."""

    x = rng.normal(config.normal_mu, config.normal_sigma, size=count)
    n_features = 2  # toy default; extended in make_toy_datasets
    return np.column_stack([x] * n_features)


def make_anomaly_angles(count: int, config: ToyDataConfig, rng: np.random.Generator) -> np.ndarray:
    """Anomaly samples: correlation-breaking near the normal mean."""

    center = rng.normal(config.normal_mu, config.normal_sigma, size=count)
    delta = rng.uniform(config.anomaly_delta_min, config.anomaly_delta_max, size=count)
    signs = rng.choice([-1.0, 1.0], size=count)
    x1 = center + signs * delta
    x2 = center - signs * delta
    return np.column_stack([x1, x2])


def make_toy_datasets(config: ToyDataConfig, n_features: int = 2) -> dict[str, np.ndarray]:
    """Build train/validation/test splits for the toy experiment."""

    if n_features != 2:
        raise NotImplementedError("Toy synthetic generator currently supports n_features=2 only")

    rng = np.random.default_rng(config.seed)
    return {
        "train_normal": make_normal_angles(config.train_size, config, rng),
        "validation_normal": make_normal_angles(config.validation_size, config, rng),
        "test_normal": make_normal_angles(config.test_size, config, rng),
        "test_anomaly": make_anomaly_angles(config.test_size, config, rng),
    }
