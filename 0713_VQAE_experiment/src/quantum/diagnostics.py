"""Ideal-simulator diagnostics (not noisy-hardware measurements)."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

from src.config import VQAEConfig
from src.quantum.circuits import (
    build_encoded_circuit,
    build_encoder_circuit,
    build_input_encoding_circuit,
    trash_qubit_indices,
)


def reconstruction_fidelity_ideal(
    angles: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
) -> float:
    """Ideal statevector reconstruction fidelity (diagnostic only).

    This must not be reported as a noisy-hardware measurement.
    """

    input_state = Statevector.from_instruction(build_input_encoding_circuit(angles, config))
    encoded = Statevector.from_instruction(build_encoded_circuit(angles, theta, config))
    encoded_density = DensityMatrix(encoded)

    trash_indices = trash_qubit_indices(config)
    keep = [q for q in range(config.n_input_qubits) if q not in trash_indices]
    latent_density = partial_trace(encoded_density, trash_indices)

    # Rebuild compressed state: latent subsystem + fresh |0> on each trash qubit
    compressed = latent_density
    for _ in trash_indices:
        compressed = compressed.expand(DensityMatrix.from_label("0"))

    decoder = build_encoder_circuit(theta, config).inverse()
    reconstructed = compressed.evolve(decoder)
    fidelity = state_fidelity(input_state, reconstructed)
    return float(np.clip(fidelity, 0.0, 1.0))


def trash_z_expectation_ideal(
    angles: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
) -> float:
    """Ideal <Z> averaged over trash qubits (diagnostic only)."""

    state = Statevector.from_instruction(build_encoded_circuit(angles, theta, config))
    z_values = []
    for q in trash_qubit_indices(config):
        probs = state.probabilities(qargs=[q])
        z_values.append(float(probs[0] - probs[1]))
    return float(np.mean(z_values))
