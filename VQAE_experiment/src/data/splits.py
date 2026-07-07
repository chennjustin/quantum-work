"""Group-aware data splitting by project + bug_id."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_bug_group(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by project+bug_id groups. No bug appears in more than one split."""

    if "project" not in df.columns or "bug_id" not in df.columns:
        raise ValueError("DataFrame must contain 'project' and 'bug_id' columns")

    groups = df[["project", "bug_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    groups = groups.iloc[perm].reset_index(drop=True)

    n = len(groups)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train_groups = groups.iloc[:n_train]
    val_groups = groups.iloc[n_train : n_train + n_val]
    test_groups = groups.iloc[n_train + n_val :]

    def _filter(selected: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(selected, on=["project", "bug_id"], how="inner")
        return merged.reset_index(drop=True)

    return _filter(train_groups), _filter(val_groups), _filter(test_groups)


def build_bugsinpy_splits(
    df: pd.DataFrame,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Build train (normal fixed), validation (normal fixed), test (normal+anomaly)."""

    normal_fixed = df[(df["revision"] == "fixed") & (df["label"] == 0)].copy()
    buggy = df[(df["revision"] == "buggy") & (df["label"] == 1)].copy()

    train, val, test_normal = split_by_bug_group(normal_fixed, seed=seed)

    # Test anomalies: held-out buggy groups not in train/val
    used = set(zip(train["project"], train["bug_id"])) | set(zip(val["project"], val["bug_id"]))
    test_anomaly = buggy[
        buggy.apply(lambda r: (r["project"], r["bug_id"]) not in used, axis=1)
    ].copy()

    test = pd.concat([test_normal, test_anomaly], ignore_index=True)
    return {"train": train, "validation": val, "test": test}
