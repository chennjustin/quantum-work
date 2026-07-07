"""Data preprocessing: fit on train normal only, map to [0, pi]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PROHIBITED_FEATURE_COLUMNS = {
    "pass_fail_status",
    "return_code",
    "exception_name",
    "exception_message",
    "buggy_fixed_flag",
    "test_failure_text",
    "test_result_label",
    "label",
}


def validate_no_label_leakage(feature_columns: list[str]) -> None:
    """Fail if prohibited label-leaking columns appear in X."""

    leaked = set(feature_columns) & PROHIBITED_FEATURE_COLUMNS
    if leaked:
        raise ValueError(f"Label leakage detected in features: {sorted(leaked)}")


@dataclass
class FeaturePreprocessor:
    """Robust scaling + clipping + angle mapping."""

    feature_columns: list[str]
    clip_min: float = 0.0
    clip_max: float = 1.0
    _scaler: RobustScaler | None = None

    def fit(self, df: pd.DataFrame) -> FeaturePreprocessor:
        """Fit scaler on training normal data only."""

        validate_no_label_leakage(self.feature_columns)
        values = df[self.feature_columns].astype(float).values
        self._scaler = RobustScaler()
        self._scaler.fit(values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features to Ry encoding angles in [0, pi]."""

        if self._scaler is None:
            raise RuntimeError("Preprocessor must be fit before transform")
        values = df[self.feature_columns].astype(float).values
        scaled = self._scaler.transform(values)
        clipped = np.clip(scaled, self.clip_min, self.clip_max)
        return clipped * np.pi

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
