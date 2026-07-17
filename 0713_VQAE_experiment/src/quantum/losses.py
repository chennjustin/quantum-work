"""SWAP-test and diagnostic loss functions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.config import VQAEConfig
from src.quantum.backend import BackendRunner
from src.quantum.circuits import (
    build_direct_trash_measurement_circuit,
    build_swap_test_circuit,
    trash_qubit_indices,
)


def swap_test_loss_sampled(
    angles: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
    backend: BackendRunner,
) -> float:
    """Official sampled SWAP-test loss: P(auxiliary = 1)."""

    circuit = build_swap_test_circuit(angles, theta, config)
    prob = backend.run_one_bit_measurement(circuit)
    return float(np.clip(prob, 0.0, 1.0))


def swap_test_loss_batch(
    angle_batch: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
    backend: BackendRunner,
) -> list[float]:
    """Compute sampled SWAP-test loss for a batch of angle vectors."""

    circuits = [build_swap_test_circuit(angles, theta, config) for angles in angle_batch]
    return [float(np.clip(p, 0.0, 1.0)) for p in backend.run_batch_one_bit_measurements(circuits)]


def mean_swap_test_loss(
    angle_batch: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
    backend: BackendRunner,
) -> float:
    """Return the mean sampled SWAP-test loss over a batch."""

    losses = swap_test_loss_batch(angle_batch, theta, config, backend)
    return float(np.mean(losses))


def direct_trash_loss_debug(
    angles: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
    backend: BackendRunner,
) -> float:
    """Optional debug loss: mean P(trash qubit = 1) across trash subsystem.

    Not used as the default training objective.
    """

    circuit = build_direct_trash_measurement_circuit(angles, theta, config)
    if backend.mode == "ideal":
        from qiskit.quantum_info import Statevector

        state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        probs = []
        for q in trash_qubit_indices(config):
            marginal = state.probabilities(qargs=[q])
            probs.append(float(marginal[1]))
        return float(np.mean(probs))

    # Noisy: measure trash bits via sampler counts on multi-bit classical register
    from qiskit import transpile

    assert backend._simulator is not None
    transpiled = transpile(
        circuit,
        backend=backend._simulator,
        optimization_level=backend.optimization_level,
        seed_transpiler=backend.seed,
    )
    job = backend._simulator.run(transpiled, shots=backend.shots, seed_simulator=backend.seed)
    counts = job.result().get_counts(transpiled)
    n_trash = len(trash_qubit_indices(config))
    total_ones = 0
    for bitstring, count in counts.items():
        ones = sum(1 for ch in bitstring if ch == "1")
        total_ones += ones * count
    return float(total_ones / (backend.shots * n_trash))
