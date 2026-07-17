"""Tests for noisy IBM fake backend."""

import numpy as np
import pytest

from src.config import VQAEConfig
from src.quantum.backend import BackendUnavailableError, create_backend_runner
from src.quantum.circuits import build_swap_test_circuit, n_encoder_parameters


def test_noisy_swap_result_in_unit_interval():
    config = VQAEConfig(n_input_qubits=2, n_latent_qubits=1, ansatz_depth=2)
    theta = np.zeros(n_encoder_parameters(config))
    circuit = build_swap_test_circuit(np.array([0.5, 0.5]), theta, config)

    runner = create_backend_runner(
        mode="ibm_fake_noisy",
        fake_backend_name="fake_manila",
        shots=512,
        seed_simulator=42,
    )
    assert not runner.uses_statevector_sampler
    prob = runner.run_one_bit_measurement(circuit)
    assert 0.0 <= prob <= 1.0
    assert runner.last_metadata is not None
    assert runner.last_metadata.transpiled_depth is not None


def test_noisy_backend_raises_without_aer(monkeypatch):
    import src.quantum.backend as backend_mod

    def _raise_import():
        raise ImportError("no aer")

    monkeypatch.setattr(backend_mod, "AerSimulator", None, raising=False)
    # Patch the import inside __post_init__ by removing qiskit_aer
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "qiskit_aer":
            raise ImportError("no aer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(BackendUnavailableError):
        create_backend_runner(mode="ibm_fake_noisy", fake_backend_name="fake_manila")
