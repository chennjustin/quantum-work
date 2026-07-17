"""Generic VQAE circuit builders for configurable qubit counts."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from src.config import VQAEConfig


def n_trash_qubits(config: VQAEConfig) -> int:
    """Return the number of trash qubits."""

    return config.n_input_qubits - config.n_latent_qubits


def n_encoder_parameters(config: VQAEConfig) -> int:
    """Return the number of trainable encoder parameters."""

    return config.n_input_qubits * config.ansatz_depth


def latent_qubit_indices(config: VQAEConfig) -> list[int]:
    """Return circuit indices assigned to the latent subsystem."""

    return list(range(config.n_latent_qubits))


def trash_qubit_indices(config: VQAEConfig) -> list[int]:
    """Return circuit indices assigned to the trash subsystem."""

    n_latent = config.n_latent_qubits
    return list(range(n_latent, config.n_input_qubits))


def reference_qubit_indices(config: VQAEConfig) -> list[int]:
    """Return reference-qubit indices (initialized as |0>)."""

    n_in = config.n_input_qubits
    n_trash = n_trash_qubits(config)
    return list(range(n_in, n_in + n_trash))


def auxiliary_qubit_index(config: VQAEConfig) -> int:
    """Return the auxiliary SWAP-test control qubit index."""

    return config.n_input_qubits + n_trash_qubits(config)


def total_swap_test_qubits(config: VQAEConfig) -> int:
    """Return total qubits in the SWAP-test circuit."""

    return auxiliary_qubit_index(config) + 1


def build_input_encoding_circuit(angles: np.ndarray, config: VQAEConfig) -> QuantumCircuit:
    """Build Ry angle encoding: one rotation per input qubit."""

    if len(angles) != config.n_input_qubits:
        raise ValueError(f"Expected {config.n_input_qubits} angles, got {len(angles)}")

    qc = QuantumCircuit(config.n_input_qubits, name="input_encoding")
    for q, angle in enumerate(angles):
        qc.ry(float(angle), q)
    return qc


def build_encoder_circuit(theta: np.ndarray, config: VQAEConfig) -> QuantumCircuit:
    """Build a shallow hardware-efficient variational encoder.

    Layout per depth layer:
      Ry rotations on all input qubits
      nearest-neighbor CX chain

    Qubit ordering:
      q0..q(L-1)   = latent
      qL..q(n-1)   = trash
    """

    expected = n_encoder_parameters(config)
    if len(theta) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(theta)}")

    n = config.n_input_qubits
    qc = QuantumCircuit(n, name="encoder")
    idx = 0
    for _ in range(config.ansatz_depth):
        for q in range(n):
            qc.ry(float(theta[idx]), q)
            idx += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def build_encoded_circuit(angles: np.ndarray, theta: np.ndarray, config: VQAEConfig) -> QuantumCircuit:
    """Compose input encoding and encoder."""

    qc = QuantumCircuit(config.n_input_qubits, name="encoded_vqae")
    qc.compose(build_input_encoding_circuit(angles, config), inplace=True)
    qc.compose(build_encoder_circuit(theta, config), inplace=True)
    return qc


def build_swap_test_circuit(angles: np.ndarray, theta: np.ndarray, config: VQAEConfig) -> QuantumCircuit:
    """Build the full SWAP-test circuit with multi-trash support.

    Circuit layout:
      q0..q(L-1)              latent (after encoder)
      qL..q(n_in-1)          trash
      qn_in..qn_in+n_tr-1    reference |0...0>
      q_aux                  auxiliary control

    Sequence:
      encode + encoder
      H(aux)
      for each trash/reference pair: CSWAP(aux, trash_i, ref_i)
      H(aux)
      measure(aux)
    """

    n_in = config.n_input_qubits
    n_tr = n_trash_qubits(config)
    aux = auxiliary_qubit_index(config)
    total_qubits = total_swap_test_qubits(config)

    qc = QuantumCircuit(total_qubits, 1, name="swap_test")
    qc.compose(build_encoded_circuit(angles, theta, config), qubits=list(range(n_in)), inplace=True)

    qc.h(aux)
    for offset, trash_q in enumerate(trash_qubit_indices(config)):
        ref_q = reference_qubit_indices(config)[offset]
        qc.cswap(aux, trash_q, ref_q)
    qc.h(aux)
    qc.measure(aux, 0)
    return qc


def build_direct_trash_measurement_circuit(
    angles: np.ndarray,
    theta: np.ndarray,
    config: VQAEConfig,
) -> QuantumCircuit:
    """Build a debug circuit measuring all trash qubits (not used for training)."""

    trash = trash_qubit_indices(config)
    qc = QuantumCircuit(config.n_input_qubits, len(trash), name="direct_trash_debug")
    qc.compose(build_encoded_circuit(angles, theta, config), inplace=True)
    for c_idx, q in enumerate(trash):
        qc.measure(q, c_idx)
    return qc
