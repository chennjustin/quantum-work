"""Test-pair-aware data splitting for BugsInPy features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_test_pair(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete rows by `(project, bug_id, test_id)` groups at train:val:test.

    Each pair is assigned to exactly one split. Callers decide which revisions
    to keep after this assignment.
    """

    required = {"project", "bug_id", "test_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame must contain columns {sorted(required)}; missing {sorted(missing)}")

    groups = df[["project", "bug_id", "test_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    groups = groups.iloc[perm].reset_index(drop=True)

    n = len(groups)
    if n < 3:
        raise ValueError(f"Need at least 3 test pairs to form train/val/test; got {n}")

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError(f"Split fractions leave no test pairs for n={n}")

    train_groups = groups.iloc[:n_train]
    val_groups = groups.iloc[n_train : n_train + n_val]
    test_groups = groups.iloc[n_train + n_val :]

    def _filter(selected: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(selected, on=["project", "bug_id", "test_id"], how="inner")
        return merged.reset_index(drop=True)

    return _filter(train_groups), _filter(val_groups), _filter(test_groups)


def build_bugsinpy_splits(
    df: pd.DataFrame,
    seed: int = 7,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """Build splits at test-pair granularity (default 7:2:1).

    - train: fixed rows only (label=0)
    - validation / test: both fixed and buggy rows from their pairs
    """

    train_pairs, val_pairs, test_pairs = split_by_test_pair(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    train = train_pairs[
        (train_pairs["revision"] == "fixed") & (train_pairs["label"] == 0)
    ].copy().reset_index(drop=True)
    validation = val_pairs.copy().reset_index(drop=True)
    test = test_pairs.copy().reset_index(drop=True)

    if train.empty:
        raise ValueError("Train split has no fixed rows after filtering")
    if validation.empty or test.empty:
        raise ValueError("Validation/test split is empty after pair assignment")

    return {"train": train, "validation": validation, "test": test}
