"""Experiment artifact management."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qiskit
import yaml


def create_experiment_dir(base: Path, experiment_name: str) -> Path:
    """Create a timestamped experiment output directory."""

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = base / f"{experiment_name}_{ts}"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)
    return out


def save_config_copy(config_path: Path, output_dir: Path) -> None:
    """Copy the exact config used."""

    shutil.copy2(config_path, output_dir / "config_used.yaml")


def save_reproducibility_metadata(output_dir: Path, extra: dict[str, Any] | None = None) -> None:
    """Save package versions and platform info."""

    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "qiskit_version": qiskit.__version__,
        "packages": {},
    }
    for pkg in ("numpy", "scipy", "pandas", "sklearn", "qiskit_aer", "qiskit_ibm_runtime"):
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            meta["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta["packages"][pkg] = "not installed"

    if extra:
        meta.update(extra)

    (output_dir / "reproducibility.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_training_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
