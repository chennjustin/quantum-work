"""VQA training loop minimizing sampled SWAP-test loss."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from src.config import ExperimentConfig
from src.quantum.backend import BackendRunner, create_backend_runner
from src.quantum.circuits import n_encoder_parameters
from src.quantum.losses import mean_swap_test_loss


@dataclass
class TrainingResult:
    """Training artifacts."""

    theta: np.ndarray
    loss_history: list[float]
    nfev: int
    success: bool
    message: str
    training_time_seconds: float
    backend_metadata: list[dict] = field(default_factory=list)


def _resolve_training_backend(config: ExperimentConfig, seed: int) -> BackendRunner:
    """Select backend for training. No silent fallback to ideal."""

    if config.training_mode in ("ideal_training", "ideal_train_noisy_eval"):
        return create_backend_runner(
            mode="ideal",
            shots=config.backend.shots,
            seed_simulator=seed,
            optimization_level=config.backend.optimization_level,
        )
    if config.training_mode == "noisy_training":
        return create_backend_runner(
            mode="ibm_fake_noisy",
            fake_backend_name=config.backend.fake_backend_name,
            shots=config.backend.shots,
            seed_simulator=seed,
            optimization_level=config.backend.optimization_level,
        )
    raise ValueError(f"Unknown training_mode: {config.training_mode}")


def train_vqae(
    train_angles: np.ndarray,
    config: ExperimentConfig,
    seed: int | None = None,
) -> TrainingResult:
    """Train VQAE encoder parameters with COBYLA on sampled SWAP-test loss."""

    import time

    opt_seed = seed if seed is not None else config.training.optimizer_seed
    backend = _resolve_training_backend(config, config.backend.seed_simulator)

    if config.training.objective != "swap_test_loss":
        raise ValueError("Only swap_test_loss is supported as the training objective")

    rng = np.random.default_rng(opt_seed)
    n_params = n_encoder_parameters(config.vqae)
    theta0 = rng.uniform(
        -config.training.initial_parameter_scale,
        config.training.initial_parameter_scale,
        size=n_params,
    )

    loss_history: list[float] = []
    metadata_log: list[dict] = []

    def objective(theta: np.ndarray) -> float:
        loss = mean_swap_test_loss(train_angles, theta, config.vqae, backend)
        loss_history.append(loss)
        if backend.last_metadata is not None:
            metadata_log.append(backend.last_metadata.__dict__)
        return loss

    start = time.perf_counter()
    result = minimize(
        objective,
        theta0,
        method=config.training.optimizer,
        options={
            "maxiter": config.training.maxiter,
            "rhobeg": config.training.cobyla_rhobeg,
            "tol": config.training.tol,
        },
    )
    elapsed = time.perf_counter() - start

    return TrainingResult(
        theta=result.x,
        loss_history=loss_history,
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        training_time_seconds=float(elapsed),
        backend_metadata=metadata_log,
    )
