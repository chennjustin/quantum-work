"""VQAE evaluation on held-out samples."""

from __future__ import annotations

import numpy as np

from src.config import ExperimentConfig
from src.quantum.backend import BackendRunner, create_backend_runner
from src.quantum.diagnostics import reconstruction_fidelity_ideal, trash_z_expectation_ideal
from src.quantum.losses import swap_test_loss_batch


def _resolve_eval_backend(config: ExperimentConfig, seed: int) -> BackendRunner:
    """Select backend for evaluation."""

    if config.training_mode == "ideal_train_noisy_eval":
        return create_backend_runner(
            mode="ibm_fake_noisy",
            fake_backend_name=config.backend.fake_backend_name,
            shots=config.backend.shots,
            seed_simulator=seed,
            optimization_level=config.backend.optimization_level,
        )
    if config.backend.mode == "ibm_fake_noisy" or config.training_mode == "noisy_training":
        return create_backend_runner(
            mode="ibm_fake_noisy",
            fake_backend_name=config.backend.fake_backend_name,
            shots=config.backend.shots,
            seed_simulator=seed,
            optimization_level=config.backend.optimization_level,
        )
    return create_backend_runner(
        mode="ideal",
        shots=config.backend.shots,
        seed_simulator=seed,
        optimization_level=config.backend.optimization_level,
    )


def evaluate_angle_batch(
    theta: np.ndarray,
    angles: np.ndarray,
    config: ExperimentConfig,
    backend: BackendRunner,
    include_ideal_diagnostics: bool = True,
) -> dict[str, np.ndarray]:
    """Compute SWAP-test scores and optional ideal diagnostics."""

    swap_losses = swap_test_loss_batch(angles, theta, config.vqae, backend)
    result: dict[str, np.ndarray] = {
        "swap_test_loss": np.array(swap_losses),
    }

    if include_ideal_diagnostics and backend.mode == "ideal":
        recon = []
        trash_z = []
        for sample in angles:
            recon.append(reconstruction_fidelity_ideal(sample, theta, config.vqae))
            trash_z.append(trash_z_expectation_ideal(sample, theta, config.vqae))
        result["reconstruction_fidelity_ideal"] = np.array(recon)
        result["reconstruction_score_ideal"] = 1.0 - np.array(recon)
        result["trash_z_expectation_ideal"] = np.array(trash_z)

    return result


def evaluate_with_seeds(
    theta: np.ndarray,
    angles: np.ndarray,
    config: ExperimentConfig,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    """Average SWAP scores across multiple simulator seeds."""

    all_losses = []
    for seed in seeds:
        backend = _resolve_eval_backend(config, seed)
        losses = swap_test_loss_batch(angles, theta, config.vqae, backend)
        all_losses.append(losses)
    stacked = np.stack(all_losses, axis=0)
    return {
        "swap_test_loss_mean": stacked.mean(axis=0),
        "swap_test_loss_std": stacked.std(axis=0),
        "swap_test_loss": stacked.mean(axis=0),
    }
