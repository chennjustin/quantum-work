"""Synthetic datasets for the minimal VQAE experiment."""

from __future__ import annotations

import numpy as np

from .config import DataConfig


def make_normal_angles(count: int, config: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Create normal samples where both qubits share the same rotation angle."""

    x = rng.normal(config.normal_mu, config.normal_sigma, size=count)
    return np.column_stack([x, x])


def make_anomaly_angles(count: int, config: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Create anomalous samples that break correlation near the normal mean."""

    center = rng.normal(config.normal_mu, config.normal_sigma, size=count)
    delta = rng.uniform(config.anomaly_delta_min, config.anomaly_delta_max, size=count)
    signs = rng.choice([-1.0, 1.0], size=count)
    x1 = center + signs * delta
    x2 = center - signs * delta
    return np.column_stack([x1, x2])


def make_datasets(config: DataConfig) -> dict[str, np.ndarray]:
    """Build train, validation, and test splits."""

    rng = np.random.default_rng(config.seed)
    return {
        "train_normal": make_normal_angles(config.train_size, config, rng),
        "validation_normal": make_normal_angles(config.validation_size, config, rng),
        "test_normal": make_normal_angles(config.test_size, config, rng),
        "test_anomaly": make_anomaly_angles(config.test_size, config, rng),
    }
