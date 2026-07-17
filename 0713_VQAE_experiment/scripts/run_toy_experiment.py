"""Shared experiment runner logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.config import ExperimentConfig, load_config
from src.data.synthetic import make_toy_datasets
from src.data.preprocessing import FeaturePreprocessor
from src.quantum.trainer import train_vqae
from src.quantum.evaluator import evaluate_angle_batch, evaluate_with_seeds
from src.quantum.backend import create_backend_runner
from src.reporting.metrics import classification_metrics
from src.reporting.plots import plot_loss_curve, plot_score_histogram, plot_roc_curve, plot_pr_curve
from src.reporting.artifacts import (
    create_experiment_dir,
    save_config_copy,
    save_reproducibility_metadata,
    save_training_metrics,
)


def _angles_to_features(angles: np.ndarray) -> np.ndarray:
    """Toy data already in angle space; return as-is."""

    return angles


def run_toy_experiment(config_path: Path) -> Path:
    """Run toy VQAE experiment and return output directory."""

    config = load_config(config_path)
    output_dir = create_experiment_dir(EXPERIMENT_ROOT / "outputs", config.experiment_name)
    save_config_copy(config_path, output_dir)
    save_reproducibility_metadata(output_dir, {"experiment_name": config.experiment_name})

    datasets = make_toy_datasets(config.data.toy, n_features=config.vqae.n_input_qubits)

    # Train
    train_result = train_vqae(datasets["train_normal"], config)

    # Evaluate with multiple noise seeds if configured
    normal_test = datasets["test_normal"]
    anomaly_test = datasets["test_anomaly"]
    val_normal = datasets["validation_normal"]

    eval_backend_mode = "ideal" if config.training_mode == "ideal_training" else "ibm_fake_noisy"
    all_metrics = []

    for seed in config.evaluation.noise_seeds[:1 if config.training_mode == "ideal_training" else 3]:
        if config.training_mode == "ideal_training":
            backend = create_backend_runner("ideal", shots=config.backend.shots, seed_simulator=seed)
            normal_eval = evaluate_angle_batch(train_result.theta, normal_test, config, backend)
            anomaly_eval = evaluate_angle_batch(train_result.theta, anomaly_test, config, backend)
            val_eval = evaluate_angle_batch(train_result.theta, val_normal, config, backend)
        else:
            config_eval = config
            backend = create_backend_runner(
                "ibm_fake_noisy",
                fake_backend_name=config.backend.fake_backend_name,
                shots=config.backend.shots,
                seed_simulator=seed,
            )
            normal_eval = evaluate_angle_batch(train_result.theta, normal_test, config_eval, backend, include_ideal_diagnostics=False)
            anomaly_eval = evaluate_angle_batch(train_result.theta, anomaly_test, config_eval, backend, include_ideal_diagnostics=False)
            val_eval = evaluate_angle_batch(train_result.theta, val_normal, config_eval, backend, include_ideal_diagnostics=False)

        metrics = classification_metrics(
            val_eval["swap_test_loss"],
            normal_eval["swap_test_loss"],
            anomaly_eval["swap_test_loss"],
            config.evaluation.validation_quantile,
        )
        metrics["noise_seed"] = seed
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "tables" / "anomaly_metrics.csv", index=False)

    # Save scores
    rows = []
    for label, angles, eval_dict in [
        ("normal", normal_test, normal_eval),
        ("anomaly", anomaly_test, anomaly_eval),
    ]:
        for i, a in enumerate(angles):
            row = {"sample_type": label, **{f"a{j}": a[j] for j in range(len(a))}}
            for k, v in eval_dict.items():
                row[k] = v[i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "tables" / "sample_scores.csv", index=False)

    pd.DataFrame({"evaluation": range(len(train_result.loss_history)), "loss": train_result.loss_history}).to_csv(
        output_dir / "tables" / "loss_history.csv", index=False
    )
    pd.DataFrame({"parameter": [f"theta{i+1}" for i in range(len(train_result.theta))], "value": train_result.theta}).to_csv(
        output_dir / "tables" / "final_parameters.csv", index=False
    )

    save_training_metrics(output_dir, {
        "training_objective": "sampled_swap_test",
        "training_mode": config.training_mode,
        "backend_mode": config.backend.mode,
        "fake_backend_name": config.backend.fake_backend_name,
        "shots": config.backend.shots,
        "success": train_result.success,
        "message": train_result.message,
        "optimizer_evaluations": train_result.nfev,
        "training_time_seconds": train_result.training_time_seconds,
        "final_training_loss": float(train_result.loss_history[-1]),
    })

    fig_dir = output_dir / "figures"
    plot_loss_curve(train_result.loss_history, fig_dir / "loss_curve.png")
    plot_score_histogram(
        normal_eval["swap_test_loss"],
        anomaly_eval["swap_test_loss"],
        "SWAP-Test Anomaly Scores",
        "SWAP-Test Loss",
        fig_dir / "swap_score_histogram.png",
    )
    plot_roc_curve(normal_eval["swap_test_loss"], anomaly_eval["swap_test_loss"], fig_dir / "swap_score_roc.png")
    plot_pr_curve(normal_eval["swap_test_loss"], anomaly_eval["swap_test_loss"], fig_dir / "precision_recall_curve.png")

    print(f"Experiment finished: {config.experiment_name}")
    print(f"Output: {output_dir}")
    print(f"ROC-AUC: {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run toy VQAE experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_toy_experiment(Path(args.config))
