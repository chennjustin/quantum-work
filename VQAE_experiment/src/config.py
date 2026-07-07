"""YAML configuration loading and validation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_DIR = EXPERIMENT_ROOT / "data"
CONFIG_DIR = EXPERIMENT_ROOT / "configs"


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} placeholders in strings."""

    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return os.environ.get(key, "")

        return re.sub(r"\$\{([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class BackendConfig:
    mode: str = "ideal"
    fake_backend_name: str = "fake_manila"
    shots: int = 4096
    seed_simulator: int = 123
    optimization_level: int = 1


@dataclass
class VQAEConfig:
    n_input_qubits: int = 2
    n_latent_qubits: int = 1
    ansatz_depth: int = 1

    def __post_init__(self) -> None:
        n_trash = self.n_input_qubits - self.n_latent_qubits
        if n_trash <= 0:
            raise ValueError("n_latent_qubits must be smaller than n_input_qubits")


@dataclass
class TrainingConfig:
    objective: str = "swap_test_loss"
    optimizer: str = "COBYLA"
    maxiter: int = 200
    optimizer_seed: int = 11
    initial_parameter_scale: float = 0.15
    cobyla_rhobeg: float = 0.15
    tol: float = 1e-6


@dataclass
class EvaluationConfig:
    validation_quantile: float = 0.95
    noise_seeds: list[int] = field(default_factory=lambda: [101, 202, 303])
    optimizer_seeds: list[int] = field(default_factory=lambda: [11])


@dataclass
class ToyDataConfig:
    normal_mu: float = 0.5
    normal_sigma: float = 0.05
    anomaly_delta_min: float = 0.15
    anomaly_delta_max: float = 0.30
    train_size: int = 80
    validation_size: int = 80
    test_size: int = 200
    seed: int = 7


@dataclass
class BugsInPyDataConfig:
    bugsinpy_root: str = ""
    workspace_root: str = ""
    command_prefix: list[str] = field(default_factory=list)
    selected_projects: list[str] = field(default_factory=list)
    selected_bug_ids: list[str] = field(default_factory=list)
    compile_timeout_seconds: int = 3600
    test_timeout_seconds: int = 1800
    selected_features: list[str] = field(
        default_factory=lambda: [
            "test_runtime_seconds",
            "coverage_ratio",
            "covered_line_count",
            "changed_line_coverage_ratio",
        ]
    )


@dataclass
class DataConfig:
    dataset: str = "toy"
    toy: ToyDataConfig = field(default_factory=ToyDataConfig)
    bugsinpy: BugsInPyDataConfig = field(default_factory=BugsInPyDataConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str = "experiment"
    backend: BackendConfig = field(default_factory=BackendConfig)
    vqae: VQAEConfig = field(default_factory=VQAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training_mode: str = "ideal_training"  # ideal_training | noisy_training | ideal_train_noisy_eval

    @property
    def n_trash_qubits(self) -> int:
        return self.vqae.n_input_qubits - self.vqae.n_latent_qubits

    @property
    def swap_test_qubits(self) -> int:
        return self.vqae.n_input_qubits + self.n_trash_qubits + 1


def _merge_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""

    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in fields}
    nested = {}
    for name in fields:
        if name in filtered and hasattr(cls, "__dataclass_fields__"):
            field_type = cls.__dataclass_fields__[name].type  # type: ignore[attr-defined]
            if isinstance(filtered[name], dict) and field_type in (
                BackendConfig,
                VQAEConfig,
                TrainingConfig,
                EvaluationConfig,
                DataConfig,
                ToyDataConfig,
                BugsInPyDataConfig,
            ):
                nested[name] = _merge_dataclass(field_type, filtered.pop(name))
    filtered.update(nested)
    return cls(**filtered)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = _expand_env(raw or {})

    backend = _merge_dataclass(BackendConfig, raw.get("backend", {}))
    vqae = _merge_dataclass(VQAEConfig, raw.get("vqae", {}))
    training = _merge_dataclass(TrainingConfig, raw.get("training", {}))
    evaluation = _merge_dataclass(EvaluationConfig, raw.get("evaluation", {}))

    data_raw = raw.get("data", {})
    dataset = data_raw.get("dataset", "toy")
    toy = _merge_dataclass(ToyDataConfig, data_raw.get("toy", data_raw if dataset == "toy" else {}))
    bugsinpy = _merge_dataclass(BugsInPyDataConfig, data_raw.get("bugsinpy", data_raw if dataset == "bugsinpy" else {}))
    data = DataConfig(dataset=dataset, toy=toy, bugsinpy=bugsinpy)

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", config_path.stem),
        backend=backend,
        vqae=vqae,
        training=training,
        evaluation=evaluation,
        data=data,
        training_mode=raw.get("training_mode", "ideal_training"),
    )
