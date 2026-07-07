This folder contains a small proof-of-concept experiment for validating a two-qubit Variational Quantum Autoencoder (VQAE) before using real software-testing datasets.

This version is an ideal Qiskit simulation proof of concept. It builds Qiskit
circuits, trains on the actual sampled SWAP-test loss via `StatevectorSampler`,
and uses exact `Statevector` / `DensityMatrix` tools only for diagnostics. It
does not yet run noisy simulators or hardware backends.

## Structure

```text
20260703_experiment/
  README.md
  scripts/
    run_minimal_vqae.py
    smoke_test_qiskit.py
  src/
    ansatz.py
    config.py
    data.py
    evaluation.py
    plotting.py
    quantum.py
    training.py
  outputs/
    figures/
    tables/
```

Root-level dependencies are stored in:

```text
requirements.txt
```



## Install

From the repository root:

```powershell
python -m pip install -r requirements.txt
```



## Run

From the repository root:

```powershell
python 20260703_experiment/scripts/run_minimal_vqae.py
```

Run the smoke checks:

```powershell
python 20260703_experiment/scripts/smoke_test_qiskit.py
```



## Experiment Design

The input state is:

```text
Ry(x1)|0> tensor Ry(x2)|0>
```

Normal data uses a shared angle:

```text
x1 = x2 = x, where x ~ N(0.5, 0.05^2)
```

Anomalous data breaks the correlation while staying near the normal mean:

```text
x ~ N(0.5, 0.05^2)
delta ~ U(0.15, 0.30)
x1 = x + delta
x2 = x - delta
```

The encoder ansatz is:

```text
[Ry(theta3) tensor Ry(theta4)] * CNOT(q0 -> q1) * [Ry(theta1) tensor Ry(theta2)]
```

Qubit assignment:

```text
q0 = latent qubit
q1 = trash qubit
```

The Qiskit encoder circuit is built with:

```python
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)
qc.ry(theta[2], 0)
qc.ry(theta[3], 1)
```



## Outputs

Running the script writes:

```text
outputs/circuit_diagrams.txt
outputs/training_metrics.json
outputs/tables/final_parameters.csv
outputs/tables/loss_history.csv
outputs/tables/sample_scores.csv
outputs/tables/score_summary.csv
outputs/tables/anomaly_metrics.csv
outputs/figures/loss_curve.png
outputs/figures/swap_score_histogram.png
outputs/figures/reconstruction_score_histogram.png
outputs/figures/swap_score_roc.png
```



## Notes

The implementation uses Qiskit for quantum execution:

```text
QuantumCircuit.ry / QuantumCircuit.cx for encoding and encoder construction
StatevectorSampler for the actual sampled SWAP-test loss (training + scoring)
DensityMatrix, partial_trace, and state_fidelity for reconstruction fidelity
Statevector marginals for the exact trash-Z diagnostic
```

Training minimizes the sampled SWAP-test loss directly. Because it is shot-based,
the objective is noisy; a fixed sampler seed keeps it reproducible for a given
theta, and all SWAP circuits in a batch run in a single sampler job for speed.

Qiskit bitstrings are displayed in big-endian order, but qargs refer to circuit
qubit indices. The code includes smoke assertions to keep q0 as latent and q1 as
trash during partial trace and reconstruction.

Threshold-based metrics use validation normal samples to choose the anomaly
threshold:

```text
threshold = quantile(validation_normal_scores, 0.95)
```

ROC-AUC is still computed on the held-out normal and anomalous test samples.

The compression loss is sampled from an actual four-qubit Qiskit SWAP-test
circuit:

```text
q0 = latent
q1 = trash
q2 = |0> reference
q3 = auxiliary

H(auxiliary) -> controlled-SWAP(auxiliary, trash, reference) -> H(auxiliary)
```

This SWAP-test loss is the single compression objective used for both training
and anomaly scoring. The exact trash-Z expectation and reconstruction fidelity
are retained only as diagnostics that confirm the trash qubit is compressed
toward |0>; they are not separate anomaly-detection models.

This is suitable for validating the training and anomaly-scoring pipeline before
adding noisy simulators, hardware backends, or BugsInPy feature encodings.