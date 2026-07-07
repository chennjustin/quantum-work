"""Collect BugsInPy execution artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.config import load_config
from src.data.bugsinpy_runner import checkout_revision, compile_revision, run_test, save_command_artifacts
from src.data.bugsinpy_features import build_execution_row, extract_features_from_run, save_processed_features
import pandas as pd


def _sample_manifest() -> list[dict]:
    """Return a sample manifest with six bug groups when BugsInPy is unavailable."""

    rows: list[dict] = []
    for i in range(1, 7):
        rows.extend([
            {"project": "sampleproj", "bug_id": f"bug{i}", "revision": "fixed", "test_id": f"test_{i}", "label": 0},
            {"project": "sampleproj", "bug_id": f"bug{i}", "revision": "buggy", "test_id": f"test_{i}", "label": 1},
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BugsInPy execution features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    bug_cfg = config.data.bugsinpy
    raw_root = EXPERIMENT_ROOT / "data" / "raw" / "bugsinpy"
    processed_path = EXPERIMENT_ROOT / "data" / "processed" / "bugsinpy_features.csv"

    if not bug_cfg.bugsinpy_root:
        print("BUGSINPY_ROOT not set. Generating sample processed features for pipeline testing.")
        rows = []
        for item in _sample_manifest():
            coverage = {
                "available": True,
                "total_lines": 100,
                "covered_lines": 80 if item["label"] == 0 else 40,
                "changed_lines_total": 20,
                "changed_lines_covered": 18 if item["label"] == 0 else 5,
            }
            features = extract_features_from_run(1.5 if item["label"] == 0 else 3.0, coverage)
            rows.append(build_execution_row(**item, features=features))
        df = pd.DataFrame(rows)
        save_processed_features(df, processed_path)
        print(f"Saved sample features to {processed_path}")
        return

    bugsinpy_root = Path(bug_cfg.bugsinpy_root)
    if not bugsinpy_root.exists():
        raise FileNotFoundError(f"BugsInPy root not found: {bugsinpy_root}")

    # Real collection would iterate selected projects/bugs
    print(f"BugsInPy collection from {bugsinpy_root} (command_prefix={bug_cfg.command_prefix})")
    print("Configure selected_projects and selected_bug_ids in config to collect real data.")


if __name__ == "__main__":
    main()
