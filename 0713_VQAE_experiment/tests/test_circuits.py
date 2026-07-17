"""Tests for quantum circuits."""

import numpy as np
import pytest

from src.config import VQAEConfig
from src.quantum.circuits import (
    build_swap_test_circuit,
    n_encoder_parameters,
    total_swap_test_qubits,
)


def test_two_qubit_swap_circuit_qubit_count():
    config = VQAEConfig(n_input_qubits=2, n_latent_qubits=1, ansatz_depth=2)
    assert total_swap_test_qubits(config) == 4
    theta = np.zeros(n_encoder_parameters(config))
    qc = build_swap_test_circuit(np.array([0.5, 0.5]), theta, config)
    assert qc.num_qubits == 4
    assert qc.num_clbits == 1


def test_four_qubit_swap_circuit_qubit_count():
    config = VQAEConfig(n_input_qubits=4, n_latent_qubits=2, ansatz_depth=1)
    assert total_swap_test_qubits(config) == 7
    theta = np.zeros(n_encoder_parameters(config))
    qc = build_swap_test_circuit(np.array([0.1, 0.2, 0.3, 0.4]), theta, config)
    assert qc.num_qubits == 7
