# MITACS Globalink 研究紀錄

## In École Polytechnique de Montréal - Montréal

Research workspace for variational quantum autoencoder (VQAE) anomaly detection experiments.

### 2026 07 06 VQAE Experiment

All experiment code, configs, scripts, and tests live in **[VQAE_experiment/](VQAE_experiment/)**.

```powershell
python -m pip install -r requirements.txt
python VQAE_experiment/scripts/smoke_test.py
python VQAE_experiment/scripts/run_toy_experiment.py --config VQAE_experiment/configs/toy_ideal.yaml
python -m pytest VQAE_experiment/tests/ -v
```

See [VQAE_experiment/README.md](VQAE_experiment/README.md) for full documentation (English with 中文 notes).
