"""Feature extraction from BugsInPy execution artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_SCHEMA_VERSION = "1.0"

DEFAULT_FEATURES = [
    "test_runtime_seconds",
    "coverage_ratio",
    "covered_line_count",
    "changed_line_coverage_ratio",
]

OPTIONAL_FEATURES = [
    "executed_file_count",
    "executed_function_count",
    "changed_file_coverage_ratio",
]


def parse_coverage_artifact(coverage_path: Path) -> dict[str, Any]:
    """Parse a simplified coverage JSON artifact."""

    if not coverage_path.exists():
        return {"available": False}
    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    return {"available": True, **data}


def extract_features_from_run(
    runtime_seconds: float,
    coverage_data: dict[str, Any],
) -> dict[str, float | None]:
    """Extract feature vector from runtime and coverage data."""

    if not coverage_data.get("available", False):
        return {name: None for name in DEFAULT_FEATURES + OPTIONAL_FEATURES}

    total_lines = coverage_data.get("total_lines", 0) or 0
    covered = coverage_data.get("covered_lines", 0) or 0
    changed_total = coverage_data.get("changed_lines_total", 0) or 0
    changed_covered = coverage_data.get("changed_lines_covered", 0) or 0
    changed_files_total = coverage_data.get("changed_files_total", 0) or 0
    changed_files_covered = coverage_data.get("changed_files_covered", 0) or 0

    return {
        "test_runtime_seconds": float(runtime_seconds),
        "coverage_ratio": float(covered / total_lines) if total_lines else None,
        "covered_line_count": float(covered),
        "changed_line_coverage_ratio": float(changed_covered / changed_total) if changed_total else None,
        "executed_file_count": float(coverage_data.get("executed_files", 0) or 0),
        "executed_function_count": float(coverage_data.get("executed_functions", 0) or 0),
        "changed_file_coverage_ratio": float(changed_files_covered / changed_files_total)
        if changed_files_total
        else None,
    }


def build_execution_row(
    project: str,
    bug_id: str,
    revision: str,
    test_id: str,
    label: int,
    features: dict[str, float | None],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one execution record."""

    row: dict[str, Any] = {
        "project": project,
        "bug_id": bug_id,
        "revision": revision,
        "test_id": test_id,
        "label": label,
    }
    row.update(features)
    if metadata:
        row.update({f"meta_{k}": v for k, v in metadata.items()})
    return row


def load_processed_features(path: Path) -> pd.DataFrame:
    """Load processed feature CSV."""

    return pd.read_csv(path)

def save_processed_features(df: pd.DataFrame, path: Path) -> None:
    """Save processed feature CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
