"""Plot helpers for the minimal VQAE experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay


def plot_loss_curve(loss_history: list[float], output_path: Path) -> None:
    """Save the optimizer loss curve."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, linewidth=2)
    ax.set_title("SWAP-Test Loss During Optimization")
    ax.set_xlabel("Objective Evaluation")
    ax.set_ylabel("Average SWAP-Test Loss P(aux=1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_histogram(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    """Save a histogram comparing normal and anomalous anomaly scores."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_scores, bins=30, alpha=0.7, label="Normal")
    ax.hist(anomaly_scores, bins=30, alpha=0.7, label="Anomaly")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_curve(normal_scores: np.ndarray, anomaly_scores: np.ndarray, output_path: Path) -> None:
    """Save the ROC curve for an anomaly score."""

    y_true = np.concatenate([np.zeros_like(normal_scores), np.ones_like(anomaly_scores)])
    scores = np.concatenate([normal_scores, anomaly_scores])
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax)
    ax.set_title("ROC Curve: SWAP-Test Score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
