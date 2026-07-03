"""Smoke checks for the Qiskit-backed VQAE implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.quantum import (
    actual_swap_test_loss_sampled,
    reconstruction_fidelity,
    run_smoke_checks,
    trash_z_expectation_exact,
)


def main() -> None:
    """Run executable assertions for the Qiskit quantum layer."""

    run_smoke_checks()

    theta = np.zeros(4)
    trash_zero_z = trash_z_expectation_exact(0.0, 0.0, theta)
    trash_one_z = trash_z_expectation_exact(0.0, np.pi, theta)
    fidelity = reconstruction_fidelity(0.0, 0.0, theta)
    swap_loss_trash_zero = actual_swap_test_loss_sampled(0.0, 0.0, theta, shots=4096, seed=123)
    swap_loss_trash_one = actual_swap_test_loss_sampled(0.0, np.pi, theta, shots=4096, seed=123)

    assert np.isclose(trash_zero_z, 1.0), "trash |0> should give <Z> = 1"
    assert np.isclose(trash_one_z, -1.0), "trash |1> should give <Z> = -1"
    assert np.isclose(fidelity, 1.0), "zero input should reconstruct perfectly"
    # Ideal SWAP-test loss is 0 for trash |0> and 0.5 for orthogonal trash |1>.
    assert swap_loss_trash_zero < 0.05, "SWAP loss should be near 0 for trash |0>"
    assert 0.4 < swap_loss_trash_one < 0.6, "SWAP loss should be near 0.5 for trash |1>"

    print("Qiskit VQAE smoke checks passed.")


if __name__ == "__main__":
    main()
