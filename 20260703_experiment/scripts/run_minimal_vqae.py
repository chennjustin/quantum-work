"""Run the minimal Variational Quantum Autoencoder proof of concept.

This script trains a two-qubit encoder on correlated normal states using the
actual sampled SWAP-test loss, then evaluates the SWAP-test anomaly score and a
reconstruction diagnostic on normal/anomalous samples.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qiskit

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.ansatz import ansatz_diagram, encoding_diagram, swap_test_diagram
from src.config import EXPERIMENT_ROOT as CONFIG_EXPERIMENT_ROOT
from src.config import FIGURE_DIR, OUTPUT_DIR, TABLE_DIR, ExperimentConfig
from src.data import make_datasets
from src.evaluation import classification_metrics, evaluate_samples, summarize_scores
from src.plotting import plot_loss_curve, plot_roc_curve, plot_score_histogram
from src.training import train_swap_test_loss


def ensure_output_dirs() -> None:
    """Create output directories used by the experiment."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)


def save_circuit_diagrams() -> None:
    """Write Qiskit-generated circuit diagrams to a text artifact."""

    content = "\n\n".join(
        [
            "Input encoding circuit",
            encoding_diagram(),
            "Variational encoder circuit",
            ansatz_diagram(),
            "Actual SWAP-test circuit",
            swap_test_diagram(),
        ]
    )
    (OUTPUT_DIR / "circuit_diagrams.txt").write_text(content + "\n", encoding="utf-8")


def build_score_frame(
    normal_angles: np.ndarray,
    anomaly_angles: np.ndarray,
    normal_eval: dict[str, np.ndarray],
    anomaly_eval: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Combine sample angles and computed scores into one table."""

    rows = []
    for label, angles, values in [
        ("normal", normal_angles, normal_eval),
        ("anomaly", anomaly_angles, anomaly_eval),
    ]:
        for idx, sample in enumerate(angles):
            rows.append(
                {
                    "sample_type": label,
                    "x1": sample[0],
                    "x2": sample[1],
                    "actual_swap_test_loss": values["actual_swap_test_loss"][idx],
                    "reconstruction_score": values["reconstruction_score"][idx],
                    "reconstruction_fidelity": values["reconstruction_fidelity"][idx],
                    "trash_z_expectation": values["trash_z_expectation"][idx],
                }
            )
    return pd.DataFrame(rows)


def build_summary_frame(
    normal_eval: dict[str, np.ndarray],
    anomaly_eval: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Create summary statistics for normal and anomalous scores."""

    rows = []
    score_names = {
        "actual_swap_test_loss": "actual_swap_test",
        "reconstruction_score": "reconstruction",
        "reconstruction_fidelity": "reconstruction_fidelity",
        "trash_z_expectation": "trash_z_expectation",
    }
    for key, method in score_names.items():
        for label, values in [("normal", normal_eval[key]), ("anomaly", anomaly_eval[key])]:
            rows.append({"method": method, "sample_type": label, **summarize_scores(values)})
    return pd.DataFrame(rows)


def build_metric_frame(
    validation_eval: dict[str, np.ndarray],
    normal_eval: dict[str, np.ndarray],
    anomaly_eval: dict[str, np.ndarray],
    config,
) -> pd.DataFrame:
    """Create anomaly-detection metrics for each score type."""

    rows = []
    for key, method in [
        ("actual_swap_test_loss", "actual_swap_test"),
        ("reconstruction_score", "reconstruction"),
    ]:
        rows.append(
            {
                "method": method,
                **classification_metrics(validation_eval[key], normal_eval[key], anomaly_eval[key], config),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full minimal VQAE experiment."""

    config = ExperimentConfig()
    ensure_output_dirs()
    save_circuit_diagrams()

    datasets = make_datasets(config.data)

    start_time = time.perf_counter()
    training_result = train_swap_test_loss(datasets["train_normal"], config.optimizer, config.simulation)
    train_seconds = time.perf_counter() - start_time

    normal_eval = evaluate_samples(training_result.theta, datasets["test_normal"], config.simulation)
    anomaly_eval = evaluate_samples(training_result.theta, datasets["test_anomaly"], config.simulation)
    validation_eval = evaluate_samples(training_result.theta, datasets["validation_normal"], config.simulation)
    train_eval = evaluate_samples(training_result.theta, datasets["train_normal"], config.simulation)

    pd.DataFrame(
        {
            "evaluation": np.arange(len(training_result.loss_history)),
            "loss": training_result.loss_history,
        }
    ).to_csv(TABLE_DIR / "loss_history.csv", index=False)

    pd.DataFrame(
        {
            "parameter": ["theta1", "theta2", "theta3", "theta4"],
            "value": training_result.theta,
        }
    ).to_csv(TABLE_DIR / "final_parameters.csv", index=False)

    score_frame = build_score_frame(
        datasets["test_normal"],
        datasets["test_anomaly"],
        normal_eval,
        anomaly_eval,
    )
    score_frame.to_csv(TABLE_DIR / "sample_scores.csv", index=False)

    summary_frame = build_summary_frame(normal_eval, anomaly_eval)
    summary_frame.to_csv(TABLE_DIR / "score_summary.csv", index=False)

    metric_frame = build_metric_frame(validation_eval, normal_eval, anomaly_eval, config.evaluation)
    metric_frame.to_csv(TABLE_DIR / "anomaly_metrics.csv", index=False)

    training_metrics = {
        "qiskit_version": qiskit.__version__,
        "training_objective": "sampled_swap_test",
        "shots": int(config.simulation.shots),
        "success": training_result.success,
        "message": training_result.message,
        "optimizer_evaluations": int(training_result.nfev),
        "training_time_seconds": float(train_seconds),
        "final_training_loss": float(training_result.loss_history[-1]),
        "mean_training_swap_test_loss": float(np.mean(train_eval["actual_swap_test_loss"])),
        "mean_training_reconstruction_fidelity": float(np.mean(train_eval["reconstruction_fidelity"])),
    }
    (OUTPUT_DIR / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2),
        encoding="utf-8",
    )

    plot_loss_curve(training_result.loss_history, FIGURE_DIR / "loss_curve.png")
    plot_score_histogram(
        normal_eval["actual_swap_test_loss"],
        anomaly_eval["actual_swap_test_loss"],
        "SWAP-Test Anomaly Scores",
        "SWAP-Test Loss P(aux=1)",
        FIGURE_DIR / "swap_score_histogram.png",
    )
    plot_score_histogram(
        normal_eval["reconstruction_score"],
        anomaly_eval["reconstruction_score"],
        "Reconstruction Anomaly Scores",
        "1 - Reconstruction Fidelity",
        FIGURE_DIR / "reconstruction_score_histogram.png",
    )
    plot_roc_curve(
        normal_eval["actual_swap_test_loss"],
        anomaly_eval["actual_swap_test_loss"],
        FIGURE_DIR / "swap_score_roc.png",
    )

    print("Minimal VQAE experiment finished.")
    print(f"Experiment root: {CONFIG_EXPERIMENT_ROOT}")
    print(f"Final training loss: {training_metrics['final_training_loss']:.6f}")
    print(f"SWAP-test ROC-AUC: {metric_frame.loc[0, 'roc_auc']:.6f}")
    print(f"Reconstruction ROC-AUC: {metric_frame.loc[1, 'roc_auc']:.6f}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
