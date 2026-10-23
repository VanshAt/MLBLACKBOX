# Failure Forecasting & Precursor Detection

The **Failure Forecaster** provides statistical early-warning signals for training instabilities before catastrophic divergence occurs.

---

## 1. Principles of Precursor Forecasting

Traditional fault detectors only trigger *after* weights have exploded or NaNs have corrupted matrices. The Failure Forecaster analyzes rolling multi-epoch trends to catch instability precursors:

- **Exponential Moving Average (EMA)**: Detects accelerating gradient trends.
- **Rolling Z-Scores**: Evaluates whether gradient norms deviate significantly from the baseline.
- **Variance Tracking**: Monitors gradient instability across mini-batches.
- **Monotonic Decay Detection**: Identifies sustained derivative collapse across layers.

---

## 2. Categorical Risk Levels

| Risk Level | Score Range | Event Identifier | Typical Time-to-Failure |
| :--- | :--- | :--- | :--- |
| **Low** | `0 – 29` | `FORECAST_LOW` | Nominal (No imminent threat) |
| **Medium** | `30 – 54` | `FORECAST_MEDIUM` | `~5 epochs` |
| **High** | `55 – 79` | `FORECAST_HIGH` | `~2 epochs` |
| **Critical** | `80 – 100` | `FORECAST_CRITICAL` | `~1 epoch` (Immediate precursor) |

---

## 3. Strict Non-Intervention Safety Guarantee

> [!CAUTION]
> **Advisory-Only Enforcement**: Failure forecasts **never** autonomously halt training or execute rollbacks. Automated state rollback is reserved strictly for confirmed runtime faults verified by the FaultDetector engine.

---

## 4. Example Forecast Output

```json
{
  "epoch": 3,
  "overall_risk_level": "High",
  "overall_risk_score": 75,
  "likely_failure_type": "GRADIENT_EXPLOSION",
  "confidence": 85,
  "estimated_epochs_to_failure": 2,
  "warning_signals": [
    "Rapid gradient acceleration: 6.4x increase over previous epoch.",
    "Gradient mean exceeds rolling average by 3.2 standard deviations."
  ],
  "recommended_action": {
    "title": "Save Checkpoint & Enable Gradient Clipping",
    "why_it_may_help": "Bounds gradient magnitudes to 1.0, neutralizing exponential growth.",
    "code_snippet": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
  }
}
```
