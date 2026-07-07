"""BugsInPy command runner with WSL support and official CLI flags."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
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


def to_wsl_path(path: Path) -> str:
    """Convert a Windows path to a WSL mount path."""

    resolved = str(path.resolve()).replace("\\", "/")
    if len(resolved) >= 2 and resolved[1] == ":":
        drive = resolved[0].lower()
        return f"/mnt/{drive}{resolved[2:]}"
    return resolved


def should_use_wsl(command_prefix: list[str] | None) -> bool:
    """Return True when commands should be routed through WSL."""

    if command_prefix:
        return command_prefix[0].lower() == "wsl"
    return os.name == "nt"


def bugsinpy_bin_dir(bugsinpy_root: Path) -> Path:
    return bugsinpy_root / "framework" / "bin"


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _build_bash_command(
    command: Sequence[str],
    cwd: Path,
    bugsinpy_root: Path,
) -> str:
    """Build a single bash command with PATH and working directory set."""

    bin_dir = to_wsl_path(bugsinpy_bin_dir(bugsinpy_root))
    wsl_cwd = to_wsl_path(cwd)
    quoted = " ".join(_shell_quote(part) for part in command)
    return (
        f'export PATH="$PATH:{bin_dir}" && '
        f"cd {_shell_quote(wsl_cwd)} && "
        f"{quoted}"
    )


def run_command(
    command: Sequence[str],
    cwd: Path,
    bugsinpy_root: Path,
    command_prefix: list[str] | None = None,
    timeout: int | None = None,
) -> CommandResult:
    """Run a command locally or through WSL on Windows."""

    prefix = list(command_prefix or [])
    start = time.perf_counter()

    if should_use_wsl(prefix):
        bash_cmd = _build_bash_command(command, cwd, bugsinpy_root)
        if prefix and prefix[0].lower() == "wsl":
            shell_parts = prefix + ["bash", "-lc", bash_cmd]
        else:
            shell_parts = ["wsl", "bash", "-lc", bash_cmd]
        proc = subprocess.run(
            shell_parts,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        full_cmd = shell_parts
        run_cwd = str(cwd)
    else:
        env = os.environ.copy()
        bin_dir = str(bugsinpy_bin_dir(bugsinpy_root))
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
        full_cmd = prefix + list(command)
        proc = subprocess.run(
            full_cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        run_cwd = str(cwd)

    elapsed = time.perf_counter() - start
    return CommandResult(
        command=full_cmd,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        elapsed_seconds=elapsed,
        cwd=run_cwd,
    )


def save_command_artifacts(result: CommandResult, output_dir: Path) -> None:
    """Save stdout/stderr/metadata for a BugsInPy command."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    (output_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    metadata = {
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "cwd": result.cwd,
    }
    (output_dir / "command_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def remove_workspace_dir(
    work_dir: Path,
    bugsinpy_root: Path,
    command_prefix: list[str] | None = None,
) -> None:
    """Remove a workspace directory, using WSL when Windows cannot unlink WSL files."""

    if not work_dir.exists():
        return

    if should_use_wsl(command_prefix):
        run_command(
            ["rm", "-rf", to_wsl_path(work_dir)],
            cwd=bugsinpy_root,
            bugsinpy_root=bugsinpy_root,
            command_prefix=command_prefix,
        )
        return

    shutil.rmtree(work_dir)


def checkout_revision(
    bugsinpy_root: Path,
    project: str,
    bug_id: str,
    revision: str,
    work_dir: Path,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
    timeout: int | None = 3600,
) -> CommandResult:
    """Checkout a BugsInPy revision into an absolute workspace directory."""

    version_id = "0" if revision == "buggy" else "1"
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    remove_workspace_dir(work_dir, bugsinpy_root, command_prefix)

    cmd = [
        "bugsinpy-checkout",
        "-p",
        project,
        "-i",
        str(bug_id),
        "-v",
        version_id,
        "-w",
        to_wsl_path(work_dir) if should_use_wsl(command_prefix) else str(work_dir.resolve()),
    ]
    result = run_command(
        cmd,
        cwd=bugsinpy_root,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def compile_revision(
    project_dir: Path,
    bugsinpy_root: Path,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
    timeout: int | None = 3600,
) -> CommandResult:
    """Compile the checked-out revision."""

    result = run_command(
        ["bugsinpy-compile"],
        cwd=project_dir,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def run_test(
    project_dir: Path,
    bugsinpy_root: Path,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
    timeout: int | None = 1800,
) -> CommandResult:
    """Run the bug-relevant test in the current checkout."""

    result = run_command(
        ["bugsinpy-test"],
        cwd=project_dir,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def run_coverage(
    project_dir: Path,
    bugsinpy_root: Path,
    command_prefix: list[str] | None = None,
    output_dir: Path | None = None,
    timeout: int | None = 1800,
) -> CommandResult:
    """Run coverage for the bug-relevant test."""

    result = run_command(
        ["bugsinpy-coverage"],
        cwd=project_dir,
        bugsinpy_root=bugsinpy_root,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    if output_dir:
        save_command_artifacts(result, output_dir)
    return result


def read_test_command(project_dir: Path) -> str:
    """Read the unittest/pytest command from bugsinpy_run_test.sh."""

    script_path = project_dir / "bugsinpy_run_test.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing run test script: {script_path}")
    for line in script_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise ValueError(f"No test command found in {script_path}")


def extract_test_id(test_command: str) -> str:
    """Extract a stable test identifier from a BugsInPy test command."""

    match = re.search(r"\.([A-Za-z0-9_]+)\s*$", test_command.strip())
    if match:
        return match.group(1)
    return test_command.strip().replace(" ", "_")


def project_checkout_dir(work_dir: Path, project: str) -> Path:
    """Return the cloned project directory inside a workspace folder."""

    return work_dir / project
