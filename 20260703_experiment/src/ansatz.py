"""Qiskit circuit builders for the two-qubit VQAE."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def _validate_theta(theta: np.ndarray) -> None:
    """Validate the four-parameter encoder vector."""

    if len(theta) != 4:
        raise ValueError("theta must contain four parameters")


def build_input_encoding_circuit(x1: float, x2: float) -> QuantumCircuit:
    """Build Ry angle encoding for the two classical input features."""

    qc = QuantumCircuit(2, name="input_encoding")
    qc.ry(float(x1), 0)
    qc.ry(float(x2), 1)
    return qc


def build_encoder_circuit(theta: np.ndarray) -> QuantumCircuit:
    """Build the trainable VQAE encoder circuit.

    Qubit roles are fixed throughout the experiment:
    q0 = latent qubit, q1 = trash qubit.
    """

    _validate_theta(theta)
    qc = QuantumCircuit(2, name="encoder")
    qc.ry(float(theta[0]), 0)
    qc.ry(float(theta[1]), 1)
    qc.cx(0, 1)
    qc.ry(float(theta[2]), 0)
    qc.ry(float(theta[3]), 1)
    return qc


def build_encoded_circuit(x1: float, x2: float, theta: np.ndarray) -> QuantumCircuit:
    """Build input encoding followed by the variational encoder."""

    qc = QuantumCircuit(2, name="encoded_vqae")
    qc.compose(build_input_encoding_circuit(x1, x2), inplace=True)
    qc.compose(build_encoder_circuit(theta), inplace=True)
    return qc


def build_swap_test_circuit(
    x1: float,
    x2: float,
    theta: np.ndarray,
) -> QuantumCircuit:
    """Build the actual four-qubit SWAP-test circuit.

    q0 = latent, q1 = trash, q2 = |0> reference, q3 = auxiliary control.
    Only the auxiliary qubit is measured at the end.
    """

    latent = 0
    trash = 1
    reference = 2
    auxiliary = 3

    qc = QuantumCircuit(4, 1, name="swap_test")
    qc.compose(build_encoded_circuit(x1, x2, theta), qubits=[latent, trash], inplace=True)
    qc.h(auxiliary)
    qc.cswap(auxiliary, trash, reference)
    qc.h(auxiliary)
    qc.measure(auxiliary, 0)
    return qc


def encoding_diagram() -> str:
    """Return a Qiskit text diagram for the data encoding circuit."""

    return str(build_input_encoding_circuit(0.5, 0.5).draw(output="text"))


def ansatz_diagram() -> str:
    """Return a Qiskit text diagram for the variational encoder."""

    theta = np.array([0.1, 0.2, 0.3, 0.4])
    return str(build_encoder_circuit(theta).draw(output="text"))


def swap_test_diagram() -> str:
    """Return a Qiskit text diagram for the sampled SWAP-test circuit."""

    theta = np.array([0.1, 0.2, 0.3, 0.4])
    return str(build_swap_test_circuit(0.5, 0.5, theta).draw(output="text"))
