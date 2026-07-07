"""Run classical baselines on BugsInPy features."""

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
from src.data.bugsinpy_features import load_processed_features
from src.baselines.isolation_forest import train_isolation_forest, anomaly_scores as if_scores
from src.baselines.classical_autoencoder import train_classical_autoencoder, reconstruction_anomaly_scores
from src.reporting.metrics import classification_metrics
from src.reporting.plots import plot_baseline_comparison
from src.reporting.artifacts import create_experiment_dir, save_config_copy, save_reproducibility_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical baselines")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_path = EXPERIMENT_ROOT / "data" / "processed" / "bugsinpy_features.csv"
    if not processed_path.exists():
        raise FileNotFoundError("Run run_bugsinpy_collection.py first.")

    df = load_processed_features(processed_path)
    features = config.data.bugsinpy.selected_features
    splits = build_bugsinpy_splits(df)

    preprocessor = FeaturePreprocessor(feature_columns=features)
    X_train = preprocessor.fit_transform(splits["train"])
    X_val = preprocessor.transform(splits["validation"])
    X_test = preprocessor.transform(splits["test"])

    y_test = splits["test"]["label"].values
    n_normal = int((y_test == 0).sum())
    normal_scores_dict = {}

    output_dir = create_experiment_dir(EXPERIMENT_ROOT / "outputs", f"{config.experiment_name}_baselines")
    save_config_copy(Path(args.config), output_dir)
    save_reproducibility_metadata(output_dir)

    # Isolation Forest
    if_model = train_isolation_forest(X_train)
    if_all = if_scores(if_model, X_test)
    if_val = if_scores(if_model, X_val)
    if_metrics = classification_metrics(
        if_val[: len(X_val)],
        if_all[:n_normal],
        if_all[n_normal:],
        config.evaluation.validation_quantile,
    )
    if_metrics["method"] = "isolation_forest"
    normal_scores_dict["Isolation Forest"] = (if_all[:n_normal], if_all[n_normal:])

    rows = [if_metrics]

    # Classical Autoencoder
    try:
        ae_model = train_classical_autoencoder(X_train, latent_dim=config.vqae.n_latent_qubits)
        ae_all = reconstruction_anomaly_scores(ae_model, X_test)
        ae_val = reconstruction_anomaly_scores(ae_model, X_val)
        ae_metrics = classification_metrics(
            ae_val,
            ae_all[:n_normal],
            ae_all[n_normal:],
            config.evaluation.validation_quantile,
        )
        ae_metrics["method"] = "classical_autoencoder"
        rows.append(ae_metrics)
        normal_scores_dict["Classical AE"] = (ae_all[:n_normal], ae_all[n_normal:])
    except ImportError as exc:
        print(f"Classical Autoencoder skipped: {exc}")

    pd.DataFrame(rows).to_csv(output_dir / "tables" / "baseline_metrics.csv", index=False)
    plot_baseline_comparison(normal_scores_dict, output_dir / "figures" / "baseline_comparison.png")
    print(f"Baselines finished. Output: {output_dir}")


if __name__ == "__main__":
    main()
