"""Run BugsInPy collection in resumable batches."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENT_ROOT / "scripts" / "run_bugsinpy_collection.py"
LOG_DIR = EXPERIMENT_ROOT / "logs"
MAIN_CONFIG = EXPERIMENT_ROOT / "configs" / "bugsinpy_collect.yaml"
SMOKE_CONFIG = EXPERIMENT_ROOT / "configs" / "bugsinpy_collect_smoke.yaml"


def run_step(label: str, config: Path, extra_args: list[str], log_path: Path) -> int:
    cmd = [sys.executable, str(SCRIPT), "--config", str(config), *extra_args]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 72}\n{datetime.now().isoformat()}  {label}\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=EXPERIMENT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nExit code: {proc.returncode}\n")
    return proc.returncode


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "collection_batches.log"

    steps = [
        ("smoke-8-bugs", SMOKE_CONFIG, ["--skip-existing"]),
        ("batch-25-bugs", MAIN_CONFIG, ["--num-bugs", "25", "--skip-existing"]),
        ("batch-50-bugs", MAIN_CONFIG, ["--num-bugs", "50", "--skip-existing"]),
        ("batch-75-bugs", MAIN_CONFIG, ["--num-bugs", "75", "--skip-existing"]),
        ("batch-100-bugs", MAIN_CONFIG, ["--num-bugs", "100", "--skip-existing"]),
    ]

    for label, config, extra_args in steps:
        code = run_step(label, config, extra_args, log_path)
        if code != 0:
            print(f"{label} failed with exit code {code}. See {log_path}")
            raise SystemExit(code)

    print(f"All collection batches finished. Log: {log_path}")


if __name__ == "__main__":
    main()
