"""Isolation Forest baseline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.05,
    seed: int = 7,
) -> IsolationForest:
    """Train Isolation Forest on normal training data only."""

    model = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
    )
    model.fit(X_train)
    return model


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous)."""

    # decision_function: higher = more normal; invert for anomaly score
    return -model.decision_function(X)
