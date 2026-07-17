"""Evaluation metrics for anomaly detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_from_validation(
    validation_normal_scores: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """Select anomaly threshold from validation normal scores."""

    return float(np.quantile(validation_normal_scores, quantile))


def classification_metrics(
    validation_normal_scores: np.ndarray,
    test_normal_scores: np.ndarray,
    test_anomaly_scores: np.ndarray,
    quantile: float = 0.95,
) -> dict[str, float]:
    """Compute threshold-based metrics and ROC-AUC / PR-AUC."""

    threshold = threshold_from_validation(validation_normal_scores, quantile)
    y_true = np.concatenate([
        np.zeros(len(test_normal_scores)),
        np.ones(len(test_anomaly_scores)),
    ])
    scores = np.concatenate([test_normal_scores, test_anomaly_scores])
    y_pred = (scores > threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, scores)
    # recall at 5% FPR
    recall_at_5fpr = 0.0
    mask = fpr <= 0.05
    if mask.any():
        recall_at_5fpr = float(np.max(tpr[mask]))

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fpr_rate = fp / max(1, tn + fp)

    return {
        "threshold": threshold,
        "threshold_quantile": float(quantile),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "false_positive_rate": float(fpr_rate),
        "recall_at_5pct_fpr": recall_at_5fpr,
    }
