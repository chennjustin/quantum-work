"""Training loop for the sampled SWAP-test VQAE objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import OptimizerConfig, SimulationConfig
from .quantum import actual_swap_test_losses_sampled


@dataclass(frozen=True)
class TrainingResult:
    """Container for trained parameters and optimization history."""

    theta: np.ndarray
    loss_history: list[float]
    nfev: int
    success: bool
    message: str


def batch_swap_loss(
    theta: np.ndarray,
    batch_angles: np.ndarray,
    simulation: SimulationConfig,
) -> float:
    """Compute mean sampled SWAP-test loss over a batch in one sampler job."""

    losses = actual_swap_test_losses_sampled(
        [(float(x1), float(x2)) for x1, x2 in batch_angles],
        theta,
        shots=simulation.shots,
        seed=simulation.sampler_seed,
    )
    return float(np.mean(losses))


def train_swap_test_loss(
    train_angles: np.ndarray,
    config: OptimizerConfig,
    simulation: SimulationConfig,
) -> TrainingResult:
    """Train the encoder by minimizing the average sampled SWAP-test loss.

    The SWAP-test objective is shot-based and therefore inherently noisy. A fixed
    sampler seed keeps the objective reproducible for a given theta, which helps
    COBYLA behave more consistently across runs.
    """

    rng = np.random.default_rng(config.seed)
    theta0 = rng.uniform(-config.initial_parameter_scale, config.initial_parameter_scale, size=4)
    loss_history: list[float] = []

    def objective(theta: np.ndarray) -> float:
        loss = batch_swap_loss(theta, train_angles, simulation)
        loss_history.append(loss)
        return loss

    result = minimize(
        objective,
        theta0,
        method="COBYLA",
        options={"maxiter": config.maxiter, "rhobeg": config.cobyla_rhobeg, "tol": config.tol},
    )

    return TrainingResult(
        theta=result.x,
        loss_history=loss_history,
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )
