"""Run BugsInPy VQAE experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.config import load_config
from src.data.preprocessing import FeaturePreprocessor
from src.data.splits import build_bugsinpy_splits
from src.data.bugsinpy_features import filter_complete_feature_pairs, load_processed_features
from src.quantum.trainer import train_vqae
from src.quantum.evaluator import evaluate_angle_batch
from src.quantum.backend import create_backend_runner
from src.reporting.metrics import classification_metrics
from src.reporting.plots import plot_loss_curve, plot_score_histogram, plot_roc_curve, plot_pr_curve
from src.reporting.artifacts import create_experiment_dir, save_config_copy, save_reproducibility_metadata, save_training_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BugsInPy VQAE experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_path = EXPERIMENT_ROOT / "data" / "processed" / "bugsinpy_features.csv"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed features not found: {processed_path}. "
            "Run scripts/run_bugsinpy_collection.py first."
        )

    df = load_processed_features(processed_path)
    features = config.data.bugsinpy.selected_features
    df = filter_complete_feature_pairs(df, features)
    n_pairs = df.groupby(["project", "bug_id", "test_id"]).ngroups
    print(
        f"Using {len(df)} rows / {n_pairs} test pairs "
        f"across {df.groupby(['project', 'bug_id']).ngroups} bugs"
    )
    splits = build_bugsinpy_splits(df)
    for name, split_df in splits.items():
        n_fixed = int((split_df["label"] == 0).sum())
        n_buggy = int((split_df["label"] == 1).sum())
        print(f"  {name}: {len(split_df)} rows (fixed={n_fixed}, buggy={n_buggy})")

    for name, split_df in splits.items():
        manifest_path = EXPERIMENT_ROOT / "data" / "manifests" / f"{name}_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(manifest_path, index=False)

    preprocessor = FeaturePreprocessor(feature_columns=features)
    train_angles = preprocessor.fit_transform(splits["train"])
    val_df = splits["validation"]
    val_angles = preprocessor.transform(val_df)
    val_normal_mask = (val_df["label"].to_numpy() == 0)

    output_dir = create_experiment_dir(EXPERIMENT_ROOT / "outputs", config.experiment_name)
    save_config_copy(Path(args.config), output_dir)
    save_reproducibility_metadata(output_dir)

    train_result = train_vqae(train_angles, config)

    all_metrics = []
    test_df = splits["test"]
    normal_mask = test_df["label"].to_numpy() == 0
    anomaly_mask = test_df["label"].to_numpy() == 1
    test_angles = preprocessor.transform(test_df)
    normal_angles = test_angles[normal_mask]
    anomaly_angles = test_angles[anomaly_mask]

    for seed in config.evaluation.noise_seeds:
        backend = create_backend_runner(
            "ibm_fake_noisy",
            fake_backend_name=config.backend.fake_backend_name,
            shots=config.backend.shots,
            seed_simulator=seed,
        )
        normal_eval = evaluate_angle_batch(train_result.theta, normal_angles, config, backend, include_ideal_diagnostics=False)
        anomaly_eval = evaluate_angle_batch(train_result.theta, anomaly_angles, config, backend, include_ideal_diagnostics=False)
        val_eval = evaluate_angle_batch(train_result.theta, val_angles, config, backend, include_ideal_diagnostics=False)
        val_normal_scores = np.asarray(val_eval["swap_test_loss"])[val_normal_mask]
        metrics = classification_metrics(
            val_normal_scores,
            normal_eval["swap_test_loss"],
            anomaly_eval["swap_test_loss"],
            config.evaluation.validation_quantile,
        )
        metrics["noise_seed"] = seed
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "tables" / "anomaly_metrics.csv", index=False)

    save_training_metrics(output_dir, {
        "training_objective": "sampled_swap_test",
        "training_mode": config.training_mode,
        "final_training_loss": float(train_result.loss_history[-1]),
        "optimizer_evaluations": train_result.nfev,
        "training_time_seconds": train_result.training_time_seconds,
    })

    fig_dir = output_dir / "figures"
    plot_loss_curve(train_result.loss_history, fig_dir / "loss_curve.png")
    plot_score_histogram(
        normal_eval["swap_test_loss"],
        anomaly_eval["swap_test_loss"],
        "SWAP-Test Anomaly Scores (BugsInPy)",
        "SWAP-Test Loss",
        fig_dir / "swap_score_histogram.png",
    )
    plot_roc_curve(normal_eval["swap_test_loss"], anomaly_eval["swap_test_loss"], fig_dir / "swap_score_roc.png")
    plot_pr_curve(normal_eval["swap_test_loss"], anomaly_eval["swap_test_loss"], fig_dir / "precision_recall_curve.png")

    print(f"BugsInPy experiment finished. Output: {output_dir}")
    print(f"ROC-AUC: {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")


if __name__ == "__main__":
    main()
