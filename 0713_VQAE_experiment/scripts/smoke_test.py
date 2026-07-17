"""Quick smoke checks for VQAE experiment wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.config import VQAEConfig, load_config
from src.quantum.backend import create_backend_runner
from src.quantum.circuits import (
    build_swap_test_circuit,
    n_encoder_parameters,
    total_swap_test_qubits,
)
from src.quantum.losses import swap_test_loss_sampled
from src.data.preprocessing import validate_no_label_leakage


def main() -> None:
    config_path = EXPERIMENT_ROOT / "configs" / "toy_ideal.yaml"
    config = load_config(config_path)
    vqae = config.vqae

    assert total_swap_test_qubits(vqae) == 4
    theta = np.zeros(n_encoder_parameters(vqae))
    angles = np.array([0.5, 0.5])
    circuit = build_swap_test_circuit(angles, theta, vqae)
    assert circuit.num_qubits == 4
    assert circuit.num_clbits == 1

    backend = create_backend_runner("ideal", shots=256, seed_simulator=123)
    loss = swap_test_loss_sampled(angles, theta, vqae, backend)
    assert 0.0 <= loss <= 1.0

    validate_no_label_leakage(["coverage_ratio", "test_runtime_seconds"])

    print("smoke_test: OK")
    print(f"  config: {config_path.name}")
    print(f"  swap_test_loss: {loss:.4f}")


if __name__ == "__main__":
    main()
