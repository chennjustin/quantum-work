"""Evaluation metrics for compression and anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import EvaluationConfig, SimulationConfig
from .quantum import (
    actual_swap_test_losses_sampled,
    reconstruction_fidelity,
    trash_z_expectation_exact,
)


def evaluate_samples(
    theta: np.ndarray,
    samples: np.ndarray,
    simulation: SimulationConfig,
) -> dict[str, np.ndarray]:
    """Compute Qiskit-backed VQAE scores for a set of angle samples.

    The compression anomaly score is the actual sampled SWAP-test loss. The
    exact trash-Z expectation and reconstruction fidelity are kept only as
    diagnostics of how well the trash qubit is compressed toward |0>.
    """

    reconstruction_scores = []
    fidelities = []
    z_values = []

    for x1, x2 in samples:
        fidelity = reconstruction_fidelity(x1, x2, theta)
        reconstruction_scores.append(1.0 - fidelity)
        fidelities.append(fidelity)
        z_values.append(trash_z_expectation_exact(x1, x2, theta))

    swap_losses = actual_swap_test_losses_sampled(
        [(float(x1), float(x2)) for x1, x2 in samples],
        theta,
        shots=simulation.shots,
        seed=simulation.sampler_seed,
    )

    return {
        "reconstruction_score": np.array(reconstruction_scores),
        "reconstruction_fidelity": np.array(fidelities),
        "trash_z_expectation": np.array(z_values),
        "actual_swap_test_loss": np.array(swap_losses),
    }


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    """Return simple descriptive statistics for a score vector."""

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def classification_metrics(
    validation_normal_scores: np.ndarray,
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    config: EvaluationConfig,
) -> dict[str, float]:
    """Evaluate threshold metrics using validation normals and ROC-AUC on test data."""

    y_true = np.concatenate([np.zeros_like(normal_scores), np.ones_like(anomaly_scores)])
    scores = np.concatenate([normal_scores, anomaly_scores])
    threshold = float(np.quantile(validation_normal_scores, config.validation_quantile))
    y_pred = (scores > threshold).astype(int)

    return {
        "threshold": threshold,
        "threshold_normal_quantile": float(config.validation_quantile),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
