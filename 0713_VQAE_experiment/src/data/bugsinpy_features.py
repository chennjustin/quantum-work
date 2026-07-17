"""Feature extraction from BugsInPy execution artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_SCHEMA_VERSION = "1.1"

PROCESSED_TEST_LEVEL_FILENAME = "bugsinpy_features.csv"
PROCESSED_REVISION_AGGREGATE_FILENAME = "bugsinpy_features_revision_aggregate.csv"

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


def parse_unittest_runtime(output: str) -> float | None:
    """Parse `Ran N test(s) in X.XXXs` from unittest output."""

    match = re.search(r"Ran \d+ tests? in ([\d.]+)s", output)
    if not match:
        return None
    return float(match.group(1))


def parse_coverage_report_text(report_text: str) -> dict[str, Any]:
    """Parse `coverage report -m` output into aggregate and per-file stats."""

    per_file: dict[str, dict[str, int]] = {}
    total_lines = covered_lines = None

    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Name"):
            continue
        parts = stripped.split()
        if stripped.startswith("TOTAL") and len(parts) >= 4:
            total_lines = int(parts[1])
            miss = int(parts[2])
            covered_lines = total_lines - miss
            continue
        if len(parts) < 4:
            continue
        try:
            stmts = int(parts[-4])
            miss = int(parts[-3])
        except ValueError:
            continue
        filename = " ".join(parts[:-4])
        per_file[filename] = {
            "stmts": stmts,
            "miss": miss,
            "covered": stmts - miss,
        }

    if total_lines is None:
        return {"available": False}

    return {
        "available": True,
        "total_lines": total_lines,
        "covered_lines": covered_lines,
        "per_file": per_file,
    }


def parse_patch_metadata(patch_text: str) -> dict[str, Any]:
    """Parse a unified diff for changed files and approximate changed lines."""

    changed_files: list[str] = []
    changed_lines = 0
    current_file: str | None = None

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            match = re.search(r"b/(.+)$", line)
            current_file = match.group(1) if match else None
            if current_file and current_file not in changed_files:
                changed_files.append(current_file)
            continue
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            changed_lines += 1

    return {
        "changed_files": changed_files,
        "changed_lines_total": changed_lines,
    }


def build_coverage_data(
    report_text: str,
    patch_text: str | None = None,
) -> dict[str, Any]:
    """Combine coverage report and optional patch metadata."""

    parsed = parse_coverage_report_text(report_text)
    if not parsed.get("available", False):
        return {"available": False}

    coverage_data: dict[str, Any] = {
        "available": True,
        "total_lines": parsed["total_lines"],
        "covered_lines": parsed["covered_lines"],
        "executed_files": len(parsed.get("per_file", {})),
        "executed_functions": 0,
        "changed_lines_total": 0,
        "changed_lines_covered": 0,
        "changed_files_total": 0,
        "changed_files_covered": 0,
    }

    if not patch_text:
        return coverage_data

    patch_meta = parse_patch_metadata(patch_text)
    changed_files = patch_meta["changed_files"]
    coverage_data["changed_lines_total"] = patch_meta["changed_lines_total"]
    coverage_data["changed_files_total"] = len(changed_files)

    per_file = parsed.get("per_file", {})
    changed_covered = 0
    changed_files_covered = 0
    for filename in changed_files:
        normalized = filename.replace("\\", "/")
        file_stats = None
        for key, stats in per_file.items():
            key_norm = key.replace("\\", "/")
            if key_norm.endswith(normalized) or normalized.endswith(key_norm):
                file_stats = stats
                break
        if file_stats is None:
            continue
        changed_covered += file_stats["covered"]
        if file_stats["covered"] > 0:
            changed_files_covered += 1

    coverage_data["changed_lines_covered"] = min(
        changed_covered,
        coverage_data["changed_lines_total"],
    )
    coverage_data["changed_files_covered"] = changed_files_covered
    return coverage_data


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
    granularity: str = "test",
) -> dict[str, Any]:
    """Build one execution record."""

    row: dict[str, Any] = {
        "project": project,
        "bug_id": bug_id,
        "revision": revision,
        "test_id": test_id,
        "label": label,
        "granularity": granularity,
    }
    row.update(features)
    if metadata:
        row.update({f"meta_{k}": v for k, v in metadata.items()})
    return row


def aggregate_revision_features(test_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate per-test execution features into one revision-level vector."""

    if not test_rows:
        raise ValueError("Cannot aggregate an empty test row list")

    feature_names = DEFAULT_FEATURES + OPTIONAL_FEATURES
    aggregated: dict[str, float | None] = {}

    for name in feature_names:
        values = [float(row[name]) for row in test_rows if row.get(name) is not None]
        if not values:
            aggregated[f"mean_{name}"] = None
            aggregated[f"std_{name}"] = None
            aggregated[f"min_{name}"] = None
            aggregated[f"max_{name}"] = None
            continue
        mean_value = float(sum(values) / len(values))
        aggregated[f"mean_{name}"] = mean_value
        if len(values) == 1:
            aggregated[f"std_{name}"] = 0.0
        else:
            variance = sum((value - mean_value) ** 2 for value in values) / len(values)
            aggregated[f"std_{name}"] = float(variance**0.5)
        aggregated[f"min_{name}"] = float(min(values))
        aggregated[f"max_{name}"] = float(max(values))

    pass_codes = [
        int(row["meta_test_returncode"])
        for row in test_rows
        if row.get("meta_test_returncode") is not None
    ]
    if pass_codes:
        aggregated["pass_rate"] = float(sum(code == 0 for code in pass_codes) / len(pass_codes))
    else:
        aggregated["pass_rate"] = None
    aggregated["n_triggering_tests"] = float(len(test_rows))
    return aggregated


def build_revision_aggregate_row(
    project: str,
    bug_id: str,
    revision: str,
    label: int,
    test_rows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one revision-level aggregate record from test-level rows."""

    aggregate_features = aggregate_revision_features(test_rows)
    aggregate_metadata = dict(metadata or {})
    aggregate_metadata["source_test_ids"] = ",".join(row["test_id"] for row in test_rows)
    return build_execution_row(
        project=project,
        bug_id=bug_id,
        revision=revision,
        test_id="__aggregate__",
        label=label,
        features=aggregate_features,
        metadata=aggregate_metadata,
        granularity="revision_aggregate",
    )


def load_processed_features(path: Path) -> pd.DataFrame:
    """Load processed feature CSV (test-level rows for VQAE)."""

    df = pd.read_csv(path)
    if "granularity" in df.columns:
        df = df[df["granularity"] == "test"].copy()
    return df.reset_index(drop=True)


def filter_complete_feature_pairs(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Keep rows whose (project, bug_id, test_id) pair has all features on both revisions."""

    complete_keys: set[tuple[str, str, str]] = set()
    for key, group in df.groupby(["project", "bug_id", "test_id"], dropna=False):
        if len(group) != 2:
            continue
        if group[feature_columns].notna().all().all():
            complete_keys.add((str(key[0]), str(key[1]), str(key[2])))

    if not complete_keys:
        raise ValueError("No complete fixed/buggy feature pairs remain after filtering")

    mask = df.apply(
        lambda row: (str(row["project"]), str(row["bug_id"]), str(row["test_id"])) in complete_keys,
        axis=1,
    )
    return df.loc[mask].reset_index(drop=True)

def save_processed_features(df: pd.DataFrame, path: Path) -> None:
    """Save processed feature CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
