"""Experiment configuration for the minimal VQAE proof of concept."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


@dataclass(frozen=True)
class DataConfig:
    """Synthetic normal and anomaly data settings."""

    normal_mu: float = 0.5
    normal_sigma: float = 0.05
    anomaly_delta_min: float = 0.15
    anomaly_delta_max: float = 0.30
    train_size: int = 80
    validation_size: int = 80
    test_size: int = 200
    seed: int = 7


@dataclass(frozen=True)
class OptimizerConfig:
    """Classical optimizer settings.

    The SWAP-test objective is sampled (shot-based), so each evaluation is more
    expensive and inherently noisy. maxiter is kept modest to keep the sampled
    training run tractable; raising it mostly buys diminishing returns against
    the shot-noise floor.
    """

    maxiter: int = 200
    initial_parameter_scale: float = 0.15
    cobyla_rhobeg: float = 0.15
    tol: float = 1e-6
    seed: int = 11


@dataclass(frozen=True)
class SimulationConfig:
    """Qiskit sampling settings for the SWAP-test objective."""

    shots: int = 4096
    sampler_seed: int = 123


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings for anomaly thresholds."""

    validation_quantile: float = 0.95


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level settings for the full experiment run."""

    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
