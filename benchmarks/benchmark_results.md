# MLBlackBox Benchmark Report

Comprehensive performance, latency, and fault-detection benchmark for the MLBlackBox Neural Network Flight Recorder.

---

## 1. Runtime Overhead & In-Memory Efficiency

Benchmarks were conducted on a 500-epoch training run using the standardized Iris binary classification task.

| Configuration | 500 Epochs Runtime | Per-Epoch Latency | Normalized Overhead | Faults Detected | False Positives |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla PyTorch Baseline** | `0.3849s` | `0.77 ms/epoch` | Baseline (`0.0%`) | — | — |
| **MLBlackBox Instrumented** | `0.5448s` | `1.09 ms/epoch` | `+41.55%` (`+0.32 ms`) | `0` | **`0.0%` (Zero)** |

> [!NOTE]
> Pure telemetry calculation (per-layer gradient norms, rolling statistics, health scoring, and layer risk ranking) adds merely **0.32 milliseconds** per epoch. Writing full JSON audit logs and `.pt` model weights to disk can be throttled using standard checkpoint intervals (`save_every_n_epochs`) for large models.

---

## 2. Fault Detection, Forecast & Automated Recovery

Each failure scenario was evaluated using `--fresh` execution isolation.

| Scenario & Script | Simulated Defect | Detected Fault | Detection Epoch | Detection Latency | Rollback Target | Precursor Risk Score | AI Diagnostic Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`demo_explosion.py`** | High LR (`0.5`) + unscaled inputs | `GRADIENT_EXPLOSION` | **Epoch 3** | `< 1 epoch` | Restored to **Epoch 2** | `CRITICAL` (`98/100`) | **`95/100` (High)** |
| **`demo_vanishing.py`** | 4 saturating Sigmoid layers | `TRAINING_STAGNATION` | **Epoch 18** | Multi-epoch plateau | Restored to **Epoch 2** | `HIGH` (`58/100`) | **`78/100` (Medium)** |
| **`demo_nan.py`** | Unclamped `log(0.0)` in loss | `NAN_INF_PROPAGATION` | **Epoch 3** | **Immediate** (Mid-batch) | Restored to **Epoch 2** | `CRITICAL` (`100/100`)| **`98/100` (High)** |
| **`test_iris.py`** | Healthy MLP convergence | *None (Nominal)* | **Epoch 500** | — | — | `LOW` (`0/100`) | **`95/100` (High)** |

---

## 3. Key Observations

1. **Immediate Break on Arithmetic Failure**: In `demo_nan.py`, the flight recorder catches `nan` or `inf` before `loss.backward()` finishes, preventing parameter weight tensors from becoming irreversibly corrupted.
2. **Deterministic Checkpoint Lineage**: In all three failure demos, the network weights, optimizer states, scheduler steps, and RNG states were restored to the healthy checkpoint from Epoch 2, enabling seamless continuation with `trainer.resume()`.
3. **Precursor Warning Visibility**: The failure forecasting engine flagged `High` or `Critical` risk prior to or concurrent with the fault event, providing early warning signals before automated safety triggers engaged.
