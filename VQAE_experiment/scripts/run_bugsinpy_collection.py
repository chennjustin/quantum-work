"""Collect BugsInPy execution artifacts into processed feature CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.config import load_config
from src.data.bugsinpy_features import (
    build_coverage_data,
    build_execution_row,
    extract_features_from_run,
    parse_unittest_runtime,
    save_processed_features,
)
from src.data.bugsinpy_runner import (
    checkout_revision,
    compile_revision,
    extract_test_id,
    project_checkout_dir,
    read_test_command,
    run_coverage,
    run_test,
    save_command_artifacts,
)


def _sample_manifest() -> list[dict]:
    """Return a sample manifest with six bug groups when BugsInPy is unavailable."""

    rows: list[dict] = []
    for i in range(1, 7):
        rows.extend([
            {"project": "sampleproj", "bug_id": f"bug{i}", "revision": "fixed", "test_id": f"test_{i}", "label": 0},
            {"project": "sampleproj", "bug_id": f"bug{i}", "revision": "buggy", "test_id": f"test_{i}", "label": 1},
        ])
    return rows


def list_bug_ids(bugsinpy_root: Path, project: str) -> list[str]:
    """List bug ids available for a project."""

    bugs_dir = bugsinpy_root / "projects" / project / "bugs"
    if not bugs_dir.exists():
        raise FileNotFoundError(f"Project bugs directory not found: {bugs_dir}")
    return sorted(
        [path.name for path in bugs_dir.iterdir() if path.is_dir()],
        key=lambda value: int(value),
    )


def resolve_workspace_root(bugsinpy_root: Path, configured_root: str) -> Path:
    if configured_root:
        return Path(configured_root)
    return bugsinpy_root / "workspace"


def _read_patch_text(project: str, bug_id: str, bugsinpy_root: Path) -> str | None:
    patch_path = bugsinpy_root / "projects" / project / "bugs" / bug_id / "bug_patch.txt"
    if not patch_path.exists():
        return None
    return patch_path.read_text(encoding="utf-8")


def collect_revision(
    *,
    project: str,
    bug_id: str,
    revision: str,
    label: int,
    bugsinpy_root: Path,
    workspace_root: Path,
    raw_root: Path,
    command_prefix: list[str],
    compile_timeout: int,
    test_timeout: int,
    skip_existing: bool,
) -> dict:
    """Checkout, run, and extract features for one revision."""

    work_dir = workspace_root / f"{project}_{bug_id}_{revision}"
    project_dir = project_checkout_dir(work_dir, project)
    artifact_dir = raw_root / "runs" / f"{project}_{bug_id}_{revision}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (artifact_dir / "features.json").exists() and project_dir.exists():
        return json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))

    checkout_result = checkout_revision(
        bugsinpy_root=bugsinpy_root,
        project=project,
        bug_id=bug_id,
        revision=revision,
        work_dir=work_dir,
        command_prefix=command_prefix,
        output_dir=artifact_dir / "checkout",
    )
    if checkout_result.returncode != 0:
        raise RuntimeError(
            f"Checkout failed for {project} bug {bug_id} ({revision}):\n"
            f"{checkout_result.stdout}\n{checkout_result.stderr}"
        )

    if not project_dir.exists():
        raise FileNotFoundError(f"Checkout project directory missing: {project_dir}")

    compile_result = compile_revision(
        project_dir=project_dir,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        output_dir=artifact_dir / "compile",
        timeout=compile_timeout,
    )
    if compile_result.returncode != 0:
        raise RuntimeError(
            f"Compile failed for {project} bug {bug_id} ({revision}):\n"
            f"{compile_result.stdout}\n{compile_result.stderr}"
        )

    test_result = run_test(
        project_dir=project_dir,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        output_dir=artifact_dir / "test",
        timeout=test_timeout,
    )
    test_output = "\n".join(
        part for part in [test_result.stdout, test_result.stderr] if part
    )
    runtime_seconds = parse_unittest_runtime(test_output)
    if runtime_seconds is None:
        runtime_seconds = test_result.elapsed_seconds

    coverage_result = run_coverage(
        project_dir=project_dir,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        output_dir=artifact_dir / "coverage",
        timeout=test_timeout,
    )
    coverage_output = "\n".join(
        part for part in [coverage_result.stdout, coverage_result.stderr] if part
    )
    coverage_file = project_dir / "coverage_bugsinpy.txt"
    if coverage_file.exists():
        coverage_output = coverage_file.read_text(encoding="utf-8")

    patch_text = _read_patch_text(project, bug_id, bugsinpy_root)
    coverage_data = build_coverage_data(coverage_output, patch_text)
    features = extract_features_from_run(runtime_seconds, coverage_data)

    test_command = read_test_command(project_dir)
    test_id = extract_test_id(test_command)
    row = build_execution_row(
        project=project,
        bug_id=str(bug_id),
        revision=revision,
        test_id=test_id,
        label=label,
        features=features,
        metadata={
            "test_returncode": test_result.returncode,
            "coverage_returncode": coverage_result.returncode,
            "workspace": str(work_dir),
        },
    )

    (artifact_dir / "features.json").write_text(
        json.dumps(row, indent=2),
        encoding="utf-8",
    )
    save_command_artifacts(test_result, artifact_dir / "test")
    save_command_artifacts(coverage_result, artifact_dir / "coverage")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BugsInPy execution features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse cached per-revision features when available",
    )
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

    workspace_root = resolve_workspace_root(bugsinpy_root, bug_cfg.workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    projects = bug_cfg.selected_projects
    if not projects:
        raise ValueError("Configure data.bugsinpy.selected_projects in the YAML config.")

    rows: list[dict] = []
    for project in projects:
        bug_ids = bug_cfg.selected_bug_ids or list_bug_ids(bugsinpy_root, project)
        print(f"Collecting {project}: bugs {', '.join(bug_ids)}")
        for bug_id in bug_ids:
            for revision, label in (("fixed", 0), ("buggy", 1)):
                print(f"  -> {project} bug {bug_id} ({revision})")
                row = collect_revision(
                    project=project,
                    bug_id=str(bug_id),
                    revision=revision,
                    label=label,
                    bugsinpy_root=bugsinpy_root,
                    workspace_root=workspace_root,
                    raw_root=raw_root,
                    command_prefix=bug_cfg.command_prefix,
                    compile_timeout=bug_cfg.compile_timeout_seconds,
                    test_timeout=bug_cfg.test_timeout_seconds,
                    skip_existing=args.skip_existing,
                )
                rows.append(row)

    df = pd.DataFrame(rows)
    save_processed_features(df, processed_path)
    print(f"Saved {len(df)} rows to {processed_path}")
    print(f"Raw artifacts: {raw_root / 'runs'}")


if __name__ == "__main__":
    main()
