"""Qiskit-backed quantum utilities for the two-qubit VQAE experiment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from .ansatz import (
    build_encoded_circuit,
    build_encoder_circuit,
    build_input_encoding_circuit,
    build_swap_test_circuit,
)


TRASH_QUBIT = 1
LATENT_QUBIT = 0


def _sample_bit_one_probabilities(
    circuits: Sequence[QuantumCircuit],
    shots: int,
    seed: int,
) -> list[float]:
    """Return P(measured bit = 1) for a batch of one-classical-bit circuits.

    All circuits run in a single StatevectorSampler job. Batching avoids the
    per-sample overhead of rebuilding the sampler, which matters because the
    SWAP-test objective is sampled for every training sample and iteration.
    """

    if shots <= 0:
        raise ValueError("shots must be positive")
    if len(circuits) == 0:
        return []

    sampler = StatevectorSampler(seed=seed)
    result = sampler.run(list(circuits), shots=shots).result()

    probabilities: list[float] = []
    for pub_result in result:
        counts = pub_result.data.c.get_counts()
        probabilities.append(float(counts.get("1", 0) / shots))
    return probabilities


def input_statevector(x1: float, x2: float) -> Statevector:
    """Prepare the input state with a Qiskit circuit and return its Statevector."""

    return Statevector.from_instruction(build_input_encoding_circuit(x1, x2))


def encoded_statevector(x1: float, x2: float, theta: np.ndarray) -> Statevector:
    """Prepare input plus encoder with Qiskit and return the encoded Statevector."""

    return Statevector.from_instruction(build_encoded_circuit(x1, x2, theta))


def trash_z_expectation_exact(x1: float, x2: float, theta: np.ndarray) -> float:
    """Return exact <Z> on q1 from Qiskit marginal probabilities.

    This is an exact diagnostic of how close the trash qubit is to |0>. It is
    used only to sanity-check the sampled SWAP-test loss, not as a training loss.
    """

    state = encoded_statevector(x1, x2, theta)
    probabilities = state.probabilities(qargs=[TRASH_QUBIT])
    return float(probabilities[0] - probabilities[1])


def reconstruction_fidelity(x1: float, x2: float, theta: np.ndarray) -> float:
    """Return Qiskit reconstruction fidelity after discarding q1 trash.

    Qiskit displays bitstrings in big-endian order, but qargs still refer to
    circuit qubit indices. We trace out q1, keep q0 as the latent subsystem,
    append a fresh |0><0| trash subsystem as q1, then apply U(theta)^dagger.
    """

    input_state = input_statevector(x1, x2)
    encoded_density = DensityMatrix(encoded_statevector(x1, x2, theta))
    latent_density = partial_trace(encoded_density, [TRASH_QUBIT])
    fresh_trash_zero = DensityMatrix.from_label("0")

    # For Qiskit's subsystem ordering, latent_density.expand(|0>) yields a
    # two-qubit density matrix where q0 is latent and q1 is the fresh trash.
    compressed_density = latent_density.expand(fresh_trash_zero)
    decoder = build_encoder_circuit(theta).inverse()
    reconstructed_density = compressed_density.evolve(decoder)
    fidelity = state_fidelity(input_state, reconstructed_density)
    return float(np.clip(fidelity, 0.0, 1.0))


def actual_swap_test_losses_sampled(
    samples: Iterable[tuple[float, float]],
    theta: np.ndarray,
    shots: int,
    seed: int,
) -> list[float]:
    """Estimate P(auxiliary = 1) for a batch with the actual Qiskit SWAP test.

    This executes the explicit four-qubit SWAP-test circuit (controlled-SWAP plus
    a final auxiliary measurement) once per sample, all within a single sampler
    job. This is the sole compression loss used for training and anomaly scoring.
    """

    circuits = [build_swap_test_circuit(x1, x2, theta) for x1, x2 in samples]
    return _sample_bit_one_probabilities(circuits, shots=shots, seed=seed)


def actual_swap_test_loss_sampled(
    x1: float,
    x2: float,
    theta: np.ndarray,
    shots: int,
    seed: int,
) -> float:
    """Estimate P(auxiliary = 1) for a single sample with the SWAP test."""

    return actual_swap_test_losses_sampled([(x1, x2)], theta, shots=shots, seed=seed)[0]


def run_smoke_checks() -> None:
    """Assert Qiskit qubit ordering and core behavior for this experiment."""

    ordering_circuit = QuantumCircuit(2)
    ordering_circuit.x(1)
    density = DensityMatrix(Statevector.from_instruction(ordering_circuit))

    q0_density = partial_trace(density, [1])
    q1_density = partial_trace(density, [0])
    assert np.isclose(q0_density.probabilities([0])[0], 1.0), "q0 should remain |0>"
    assert np.isclose(q1_density.probabilities([0])[1], 1.0), "q1 should be |1>"

    latent_one = DensityMatrix.from_label("1")
    trash_zero = DensityMatrix.from_label("0")
    combined = latent_one.expand(trash_zero)
    assert np.isclose(combined.probabilities([0])[1], 1.0), "q0 should be latent |1>"
    assert np.isclose(combined.probabilities([1])[0], 1.0), "q1 should be fresh trash |0>"

    theta_zero = np.zeros(4)
    # Trash |0> gives <Z> = +1, trash |1> gives <Z> = -1.
    assert np.isclose(trash_z_expectation_exact(0.0, 0.0, theta_zero), 1.0), "trash should be |0>"
    assert np.isclose(trash_z_expectation_exact(0.0, np.pi, theta_zero), -1.0), "trash should be |1>"
