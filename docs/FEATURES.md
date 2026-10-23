# MLBlackBox Feature Guide

Detailed specification of every component, metric, and interface in the MLBlackBox observability suite.

---

## 1. Core Telemetry & Observability
- **Accurate Gradient Capture**: Gradients are measured between `loss.backward()` and `optimizer.step()`, capturing true gradient dynamics instead of zeroed or stale tensors.
- **Layer-Wise Forensics**: Global gradient norms as well as per-layer statistics (mean, variance, min, max) are evaluated across every named parameter module.
- **Weight Magnitude Drift**: Tracks weight L2 norms and growth rates to preemptively identify exploding weights or parameter saturation.

---

## 2. Five-Fault Detection Engine
- **`GRADIENT_EXPLOSION`**: Triggers when gradient norms exceed the dynamic rolling baseline ($r_{\text{mean}} + \sigma \cdot r_{\text{std}}$) and gradient norm $>0.1$.
- **`GRADIENT_VANISHING`**: Detects when gradients collapse toward machine epsilon while training progress stalls.
- **`NAN_INF_PROPAGATION`**: Real-time arithmetic watchdog that stops execution at the exact batch and layer where invalid floating-point values occur.
- **`LOSS_SPIKE`**: Detects sudden non-convergent multipliers ($>3\times$ to $5\times$) in loss values caused by bad batches or aggressive learning rates.
- **`TRAINING_STAGNATION`**: Flags plateaus where loss remains trapped within a narrow spread for $>15$ epochs despite gradient activity.

---

## 3. Checkpoint Manager & Automated Recovery
- **Architecture Validation**: Checkpoints serialize architecture signatures; attempts to load weights into mismatched layer configurations raise clean `ArchitectureMismatchError` exceptions instead of shape crashes.
- **Full State Serialization**: Checkpoints save `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, epoch index, and RNG states (`torch`, `cuda`, `numpy`, `random`).
- **Seamless Resumption**:
  ```python
  trainer.resume(checkpoint_target=2, learning_rate=0.001, max_grad_norm=1.0)
  ```
- **Automated Rollback**: One-call rollback flow:
  ```python
  trainer.resume_from_last_healthy_checkpoint()
  ```
- **Retention & Archives**: Automated retention policies (`keep_latest_n=5`, preserving the last clean checkpoint) and one-click `.tar.gz` export and import.

---

## 4. Training Health Score (0–100)
Composite index calculated every epoch:
- **Loss Convergence**: Penalties for upward drift or wild oscillations.
- **Gradient Stability**: Penalties for high variance or near-zero decay.
- **NaN / Arithmetic Safety**: Immediate drop to 0 if NaNs occur.
- **Stagnation Risk**: Deductions for persistent plateaus.

---

## 5. Early-Warning Failure Forecasting
Statistical precursor detection running locally without deep learning models:
- Computes categorical risk levels: `Low`, `Medium`, `High`, `Critical`.
- Estimates lead time to potential failure (e.g. *~2 epochs*).
- Formulates advisory preventive recommendations (e.g. *Enable gradient clipping*).
- **Advisory Only Guarantee**: Forecasts **never** halt training or force rollbacks autonomously.

---

## 6. AI Training Doctor
Offline, zero-dependency diagnostic assistant providing actionable root-cause hypotheses:
- Generates headline, what happened, when it started, affected layers, likely root cause, evidence citations, and ranked remediation steps.
- Includes copyable PyTorch code snippets.
- Optional LLM mode with strict telemetry sanitization and automatic fallback to local rule-based AI.

---

## 7. Streamlit Control Room Dashboard
- 9 dedicated pages: **Overview**, **Training Metrics**, **Layer Health**, **Fault Timeline**, **Checkpoints**, **Run History**, **AI Training Doctor**, **Failure Forecast**, and **Settings & Reports**.
- Cyberpunk dark theme with high-contrast cards, responsive charts, and status indicators.
- Multi-format report exports: Markdown (`.md`), CSV (`.csv`), JSON (`.json`), and printable HTML (`.html`).
