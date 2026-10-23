# AI Training Doctor & Privacy Architecture

The **AI Training Doctor** is an expert ML observability diagnostician designed to bridge the gap between low-level gradient metrics and actionable PyTorch remediation steps.

---

## 1. Local Rule-Based Engine (Default)

The core diagnostic assistant is 100% offline, deterministic, and requires no network access or third-party dependencies.

### Capabilities:
- Deep multi-source telemetry correlation: analyzes audit logs, loss curves, gradient evolutions, health scores, layer risk scores, and checkpoint metadata.
- Precursor vs. Post-Mortem Differentiation: Clearly distinguishes between early precursor warnings (`[FORECAST EARLY WARNING]`) and confirmed fatal breakdowns.
- Structured Diagnostic Output:
  - **Headline**: Short summary of the event.
  - **What Happened**: Clear timeline and description.
  - **When It Started**: Specific epoch or batch index.
  - **Affected Layers**: Specific module keys.
  - **Likely Root Cause**: Detailed mathematical and architectural reasoning.
  - **Confidence**: Quantitative score (0–100) and qualitative rating (`Low`, `Medium`, `High`).
  - **Evidence**: Concrete citations from the audit logs.
  - **Rollback Recommendation**: Boolean advisory.
  - **Ranked Remediation Plan**: Prioritized steps with expected impact, risk level, and copyable PyTorch code.

---

## 2. Optional LLM Mode & Fallback Architecture

Users can optionally enable OpenAI-compatible LLM hypotheses in the dashboard or via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MLBLACKBOX_LLM_MODEL="gpt-4o-mini"
```

### Automatic Fallback Rules:
1. If `OPENAI_API_KEY` is empty or missing, LLM mode automatically defaults to the Local Rule-Based Doctor.
2. If the API request times out or encounters network failures, it reverts immediately to the Local Rule-Based Doctor.
3. If the LLM response fails JSON schema validation, it falls back to the Local Rule-Based Doctor.

---

## 3. Strict Privacy & Telemetry Sanitization Guarantees

When LLM mode is active, MLBlackBox enforces strict data privacy controls:

- **NO Raw Model Weights**: Model parameter weights and bias tensors are never included in prompts.
- **NO Dataset Samples or Labels**: Private training data, patient records, or text tokens are never sent.
- **NO Source Code**: User model code or repository scripts are never transmitted.
- **Transmitted Data**: Only anonymized, downsampled numerical metrics (e.g. `loss: 0.054`, `gradient_norm: 1.2e-4`, `health_score: 85`, `fault_type: GRADIENT_EXPLOSION`).

---

## 4. Advisory Disclaimer

> [!IMPORTANT]
> All AI Training Doctor outputs—both local rule-based analyses and LLM-assisted explanations—are **diagnostic hypotheses**. They are designed to assist ML practitioners and must be reviewed prior to modifying production pipelines. MLBlackBox never alters model architectures or training hyperparameters autonomously.
