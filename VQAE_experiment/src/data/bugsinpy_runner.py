"""BugsInPy command runner with configurable command prefix."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class CommandResult:
    """Captured subprocess result."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    cwd: str


def run_command(
    command: Sequence[str],
    cwd: Path,
    command_prefix: list[str] | None = None,
    timeout: int | None = None,
) -> CommandResult:
    """Run a command, optionally prefixed for WSL/Docker."""

    prefix = list(command_prefix or [])
    full_cmd = prefix + list(command)
    start = time.perf_counter()
    proc = subprocess.run(
        full_cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    return CommandResult(
        command=full_cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_seconds=elapsed,
        cwd=str(cwd),
    )


def save_command_artifacts(result: CommandResult, output_dir: Path) -> None:
    """Save stdout/stderr/metadata for a BugsInPy command."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    metadata = {
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "cwd": result.cwd,
    }
    (output_dir / "command_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "runtime.json").write_text(
        json.dumps({"elapsed_seconds": result.elapsed_seconds}, indent=2),
        encoding="utf-8",
    )


def checkout_revision(
    bugsinpy_root: Path,
    project: str,
    bug_id: str,
    revision: str,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
) -> CommandResult:
    """Checkout a BugsInPy revision."""

    cmd = ["bugsinpy-checkout", project, bug_id, revision]
    result = run_command(cmd, cwd=bugsinpy_root, command_prefix=command_prefix)
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def compile_revision(
    bugsinpy_root: Path,
    project: str,
    bug_id: str,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
) -> CommandResult:
    """Compile the checked-out revision."""

    cmd = ["bugsinpy-compile"]
    result = run_command(cmd, cwd=bugsinpy_root, command_prefix=command_prefix)
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def run_test(
    bugsinpy_root: Path,
    test_id: str,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
) -> CommandResult:
    """Run a single test in the current BugsInPy checkout."""

    cmd = ["bugsinpy-test", test_id]
    result = run_command(cmd, cwd=bugsinpy_root, command_prefix=command_prefix)
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def run_coverage(
    bugsinpy_root: Path,
    test_id: str,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
) -> CommandResult:
    """Run coverage for a test (project-specific; wrapper command)."""

    cmd = ["bugsinpy-coverage", test_id]
    result = run_command(cmd, cwd=bugsinpy_root, command_prefix=command_prefix)
    if output_dir:
        save_command_artifacts(result, output_dir)
        (output_dir / "coverage_raw").mkdir(exist_ok=True)
    return result
