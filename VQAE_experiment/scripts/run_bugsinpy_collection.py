"""Collect BugsInPy execution artifacts into processed feature CSVs."""

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
    PROCESSED_REVISION_AGGREGATE_FILENAME,
    PROCESSED_TEST_LEVEL_FILENAME,
    build_coverage_data,
    build_execution_row,
    build_revision_aggregate_row,
    extract_features_from_run,
    parse_unittest_runtime,
    save_processed_features,
)
from src.data.bugsinpy_runner import (
    checkout_revision,
    compile_revision,
    extract_test_id,
    parse_triggering_test_commands,
    project_checkout_dir,
    read_coverage_output,
    run_single_coverage,
    run_single_test,
    save_command_artifacts,
    single_test_target,
)


def _sample_manifest() -> list[dict]:
    """Return a sample manifest with six bug groups when BugsInPy is unavailable."""

    rows: list[dict] = []
    for i in range(1, 7):
        for test_suffix in ("A", "B"):
            rows.extend([
                {
                    "project": "sampleproj",
                    "bug_id": f"bug{i}",
                    "revision": "fixed",
                    "test_id": f"test_{i}_{test_suffix}",
                    "label": 0,
                    "granularity": "test",
                },
                {
                    "project": "sampleproj",
                    "bug_id": f"bug{i}",
                    "revision": "buggy",
                    "test_id": f"test_{i}_{test_suffix}",
                    "label": 1,
                    "granularity": "test",
                },
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


def resolve_tests_per_bug(tests_per_bug: str | int) -> int | None:
    """Return a per-bug test cap, or None for all triggering tests."""

    if isinstance(tests_per_bug, int):
        if tests_per_bug <= 0:
            raise ValueError("tests_per_bug must be positive or 'all_triggering'")
        return tests_per_bug
    if tests_per_bug == "all_triggering":
        return None
    raise ValueError("tests_per_bug must be 'all_triggering' or a positive integer")


def _row_key(row: dict) -> tuple[str, str, str, str]:
    return (row["project"], str(row["bug_id"]), row["revision"], row["test_id"])


def _aggregate_key(row: dict) -> tuple[str, str, str]:
    return (row["project"], str(row["bug_id"]), row["revision"])


def merge_test_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Merge test-level rows without duplicating executions."""

    merged = {_row_key(row): row for row in existing}
    merged.update({_row_key(row): row for row in new_rows})
    return list(merged.values())


def merge_aggregate_rows(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    """Merge revision aggregate rows."""

    merged = {_aggregate_key(row): row for row in existing}
    merged.update({_aggregate_key(row): row for row in new_rows})
    return list(merged.values())


def load_existing_test_rows(path: Path) -> list[dict]:
    """Load prior test-level rows if a processed CSV already exists."""

    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "granularity" in df.columns:
        df = df[df["granularity"] == "test"]
    return df.to_dict("records")


def load_existing_aggregate_rows(path: Path) -> list[dict]:
    """Load prior revision aggregate rows if present."""

    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "granularity" in df.columns:
        df = df[df["granularity"] == "revision_aggregate"]
    return df.to_dict("records")


def plan_bug_targets(
    *,
    bugsinpy_root: Path,
    projects: list[str],
    selected_bug_ids: list[str],
    num_bugs: int,
) -> list[tuple[str, str]]:
    """Build an ordered list of (project, bug_id) pairs up to num_bugs."""

    if num_bugs <= 0:
        raise ValueError("collection.num_bugs must be positive")

    planned: list[tuple[str, str]] = []
    for project in projects:
        bug_ids = selected_bug_ids or list_bug_ids(bugsinpy_root, project)
        for bug_id in bug_ids:
            planned.append((project, str(bug_id)))
            if len(planned) >= num_bugs:
                return planned
    if len(planned) < num_bugs:
        print(
            f"Warning: only {len(planned)} bugs available across selected projects, "
            f"less than collection.num_bugs={num_bugs}."
        )
    return planned


def _read_patch_text(project: str, bug_id: str, bugsinpy_root: Path) -> str | None:
    patch_path = bugsinpy_root / "projects" / project / "bugs" / bug_id / "bug_patch.txt"
    if not patch_path.exists():
        return None
    return patch_path.read_text(encoding="utf-8")


def _collect_single_test_execution(
    *,
    project: str,
    bug_id: str,
    revision: str,
    label: int,
    test_command: str,
    project_dir: Path,
    bugsinpy_root: Path,
    artifact_dir: Path,
    command_prefix: list[str],
    test_timeout: int,
    patch_text: str | None,
    workspace: str,
) -> dict:
    """Run one triggering test on the current checkout and extract features."""

    test_id = extract_test_id(test_command)
    single_target = single_test_target(test_command)
    test_artifact_dir = artifact_dir / f"test_{test_id}"
    test_artifact_dir.mkdir(parents=True, exist_ok=True)

    test_result = run_single_test(
        project_dir=project_dir,
        bugsinpy_root=bugsinpy_root,
        single_test=single_target,
        command_prefix=command_prefix,
        output_dir=test_artifact_dir / "run",
        timeout=test_timeout,
    )
    test_output = "\n".join(part for part in [test_result.stdout, test_result.stderr] if part)
    runtime_seconds = parse_unittest_runtime(test_output)
    if runtime_seconds is None:
        runtime_seconds = test_result.elapsed_seconds

    coverage_result = run_single_coverage(
        project_dir=project_dir,
        bugsinpy_root=bugsinpy_root,
        single_test=single_target,
        command_prefix=command_prefix,
        output_dir=test_artifact_dir / "coverage",
        timeout=test_timeout,
    )
    coverage_output = read_coverage_output(project_dir, coverage_result)
    coverage_data = build_coverage_data(coverage_output, patch_text)
    features = extract_features_from_run(runtime_seconds, coverage_data)

    row = build_execution_row(
        project=project,
        bug_id=str(bug_id),
        revision=revision,
        test_id=test_id,
        label=label,
        features=features,
        metadata={
            "test_command": test_command,
            "single_test_target": single_target,
            "test_returncode": test_result.returncode,
            "coverage_returncode": coverage_result.returncode,
            "workspace": workspace,
        },
        granularity="test",
    )
    (test_artifact_dir / "features.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    save_command_artifacts(test_result, test_artifact_dir / "run")
    save_command_artifacts(coverage_result, test_artifact_dir / "coverage")
    return row


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
    tests_per_bug: int | None,
) -> tuple[list[dict], dict]:
    """Checkout one revision, run each triggering test, and build aggregate row."""

    work_dir = workspace_root / f"{project}_{bug_id}_{revision}"
    project_dir = project_checkout_dir(work_dir, project)
    artifact_dir = raw_root / "runs" / f"{project}_{bug_id}_{revision}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = artifact_dir / "aggregate" / "features.json"

    test_commands = None
    if skip_existing and aggregate_path.exists():
        aggregate_row = json.loads(aggregate_path.read_text(encoding="utf-8"))
        test_rows = []
        for test_artifact in sorted(artifact_dir.glob("test_*/features.json")):
            test_rows.append(json.loads(test_artifact.read_text(encoding="utf-8")))
        if test_rows:
            return test_rows, aggregate_row

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

    test_commands = parse_triggering_test_commands(project_dir / "bugsinpy_run_test.sh")
    if tests_per_bug is not None:
        test_commands = test_commands[:tests_per_bug]
    patch_text = _read_patch_text(project, str(bug_id), bugsinpy_root)
    workspace = str(work_dir)

    test_rows: list[dict] = []
    for test_command in test_commands:
        test_id = extract_test_id(test_command)
        cached_path = artifact_dir / f"test_{test_id}" / "features.json"
        if skip_existing and cached_path.exists():
            test_rows.append(json.loads(cached_path.read_text(encoding="utf-8")))
            continue

        row = _collect_single_test_execution(
            project=project,
            bug_id=bug_id,
            revision=revision,
            label=label,
            test_command=test_command,
            project_dir=project_dir,
            bugsinpy_root=bugsinpy_root,
            artifact_dir=artifact_dir,
            command_prefix=command_prefix,
            test_timeout=test_timeout,
            patch_text=patch_text,
            workspace=workspace,
        )
        test_rows.append(row)

    aggregate_row = build_revision_aggregate_row(
        project=project,
        bug_id=str(bug_id),
        revision=revision,
        label=label,
        test_rows=test_rows,
        metadata={"workspace": workspace, "n_triggering_tests": len(test_rows)},
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate_row, indent=2), encoding="utf-8")
    return test_rows, aggregate_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BugsInPy execution features")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse cached per-test and aggregate features when available",
    )
    parser.add_argument(
        "--num-bugs",
        type=int,
        default=None,
        help="Override collection.num_bugs from the YAML config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bug_cfg = config.data.bugsinpy
    raw_root = EXPERIMENT_ROOT / "data" / "raw" / "bugsinpy"
    processed_dir = EXPERIMENT_ROOT / "data" / "processed"
    test_level_path = processed_dir / PROCESSED_TEST_LEVEL_FILENAME
    aggregate_path = processed_dir / PROCESSED_REVISION_AGGREGATE_FILENAME

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
        test_df = pd.DataFrame(rows)
        save_processed_features(test_df, test_level_path)
        aggregate_rows = []
        for (project, bug_id, revision), group in test_df.groupby(["project", "bug_id", "revision"]):
            aggregate_rows.append(
                build_revision_aggregate_row(
                    project=project,
                    bug_id=str(bug_id),
                    revision=revision,
                    label=int(group["label"].iloc[0]),
                    test_rows=group.to_dict("records"),
                )
            )
        save_processed_features(pd.DataFrame(aggregate_rows), aggregate_path)
        print(f"Saved sample test-level features to {test_level_path}")
        print(f"Saved sample revision aggregates to {aggregate_path}")
        return

    bugsinpy_root = Path(bug_cfg.bugsinpy_root)
    if not bugsinpy_root.exists():
        raise FileNotFoundError(f"BugsInPy root not found: {bugsinpy_root}")

    workspace_root = resolve_workspace_root(bugsinpy_root, bug_cfg.workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    projects = bug_cfg.selected_projects
    if not projects:
        raise ValueError("Configure data.bugsinpy.selected_projects in the YAML config.")

    collection = bug_cfg.collection
    num_bugs = args.num_bugs or collection.num_bugs
    tests_per_bug = resolve_tests_per_bug(collection.tests_per_bug)
    bug_targets = plan_bug_targets(
        bugsinpy_root=bugsinpy_root,
        projects=projects,
        selected_bug_ids=bug_cfg.selected_bug_ids,
        num_bugs=num_bugs,
    )
    tests_label = (
        "all triggering tests in run_test.sh"
        if tests_per_bug is None
        else f"first {tests_per_bug} triggering test(s) per bug"
    )
    print(
        f"Collection plan: {num_bugs} bugs, {tests_label}, "
        f"2 revisions each -> up to {num_bugs * (tests_per_bug or 1) * 2} test-level rows "
        f"(more if bugs have multiple triggering tests)."
    )
    print(f"Planned bugs ({len(bug_targets)}): {bug_targets[:5]}{'...' if len(bug_targets) > 5 else ''}")

    test_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    failures: list[dict] = []
    for project, bug_id in bug_targets:
        print(f"Collecting {project} bug {bug_id}")
        bug_failed = False
        for revision, label in (("fixed", 0), ("buggy", 1)):
            print(f"  -> {project} bug {bug_id} ({revision})")
            try:
                revision_tests, aggregate_row = collect_revision(
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
                    tests_per_bug=tests_per_bug,
                )
            except Exception as exc:
                print(f"     FAILED: {exc}")
                failures.append(
                    {
                        "project": project,
                        "bug_id": str(bug_id),
                        "revision": revision,
                        "error": str(exc),
                    }
                )
                bug_failed = True
                break
            print(f"     collected {len(revision_tests)} test execution(s)")
            test_rows.extend(revision_tests)
            aggregate_rows.append(aggregate_row)
        if bug_failed:
            print(f"  Skipping remaining revisions for {project} bug {bug_id}")

    if failures:
        failures_path = processed_dir / "collection_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Recorded {len(failures)} failure(s) to {failures_path}")

    test_rows = merge_test_rows(load_existing_test_rows(test_level_path), test_rows)
    aggregate_rows = merge_aggregate_rows(load_existing_aggregate_rows(aggregate_path), aggregate_rows)

    test_df = pd.DataFrame(test_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    save_processed_features(test_df, test_level_path)
    save_processed_features(aggregate_df, aggregate_path)
    print(f"Saved {len(test_df)} test-level rows to {test_level_path}")
    print(f"Saved {len(aggregate_df)} revision-aggregate rows to {aggregate_path}")
    print(f"Raw artifacts: {raw_root / 'runs'}")


if __name__ == "__main__":
    main()
