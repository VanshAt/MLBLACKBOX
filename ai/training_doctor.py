"""
ai/training_doctor.py — Training Doctor: AI-powered diagnostic assistant for MLBlackBox.

Provides:
1. Local rule-based offline diagnosis engine (always available, deterministic, zero-dependency).
2. Optional OpenAI-compatible LLM diagnostic layer that falls back automatically to rule-based engine.
3. Telemetry sanitizer ensuring raw weights, datasets, and code are NEVER sent to external APIs.
4. Structured diagnosis objects with evidence citation, confidence scoring, rollback advice, and code fixes.
"""

import os
import json
import math
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

from tracker.fault_detector import FaultType


# -----------------------------------------------------------------------------
# Structured Diagnosis Data Classes
# -----------------------------------------------------------------------------

@dataclass
class RemediationStep:
    title: str
    why_it_may_help: str
    expected_impact: str
    risk_level: str  # "Low", "Medium", "High"
    confidence: str  # "Low", "Medium", "High"
    code_snippet: str = ""


@dataclass
class TrainingDoctorReport:
    headline: str
    what_happened: str
    when_it_started: str
    affected_layers: List[str]
    likely_root_cause: str
    confidence_level: str  # "Low", "Medium", "High"
    confidence_score: int  # 0 to 100
    evidence: List[str]
    rollback_recommended: bool
    ranked_remediation_steps: List[RemediationStep]
    code_snippets: List[str]
    limitations: List[str]
    mode: str = "Local Rule-Based AI"  # or "AI-Generated Hypothesis (LLM)"
    disclaimer: str = (
        "AI recommendations are strictly advisory and hypotheses based on summarized "
        "training telemetry. Do not automatically apply code modifications without human verification."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# PyTorch Code Snippets Library
# -----------------------------------------------------------------------------

SNIPPETS = {
    "clip_grad_norm": (
        "# 1. Enable Gradient Clipping in your training loop\n"
        "import torch\n\n"
        "# Place immediately after loss.backward() and before optimizer.step():\n"
        "loss.backward()\n"
        "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
        "optimizer.step()\n"
    ),
    "reduce_learning_rate": (
        "# 2. Reduce Optimizer Learning Rate\n"
        "# Scale down learning rate by 5x-10x\n"
        "for param_group in optimizer.param_groups:\n"
        "    param_group['lr'] = param_group['lr'] * 0.1\n"
        "print(f\"Updated learning rate to {optimizer.param_groups[0]['lr']}\")\n"
    ),
    "leaky_relu_replace": (
        "# 3. Replace standard ReLU with LeakyReLU or GELU to avoid dying neurons\n"
        "import torch.nn as nn\n\n"
        "class RobustNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        # LeakyReLU maintains 0.01 gradient slope for negative activations\n"
        "        self.act = nn.LeakyReLU(negative_slope=0.01)\n"
        "        # Or alternatively: self.act = nn.GELU()\n"
    ),
    "batchnorm_insert": (
        "# 4. Add Batch Normalization between Linear/Conv layers and Activations\n"
        "import torch.nn as nn\n\n"
        "layer_block = nn.Sequential(\n"
        "    nn.Linear(in_features=64, out_features=128),\n"
        "    nn.BatchNorm1d(num_features=128),\n"
        "    nn.ReLU()\n"
        ")\n"
    ),
    "input_normalization": (
        "# 5. Normalize / Standardize Inputs before Forward Pass\n"
        "# Ensure inputs have mean ~ 0 and std ~ 1\n"
        "mean = x_train.mean(dim=0, keepdim=True)\n"
        "std = x_train.std(dim=0, keepdim=True) + 1e-7\n"
        "x_normalized = (x_train - mean) / std\n"
    ),
    "safe_log_clamp": (
        "# 6. Safe Logarithm Clamping to prevent NaN/Inf in Loss Computation\n"
        "import torch\n\n"
        "# Ensure probabilities are bounded away from 0.0 and 1.0\n"
        "eps = 1e-7\n"
        "probs_clamped = torch.clamp(probs, min=eps, max=1.0 - eps)\n"
        "loss = -torch.mean(targets * torch.log(probs_clamped))\n"
    ),
    "add_momentum": (
        "# 7. Add Momentum or Switch to AdamW Optimizer\n"
        "import torch.optim as optim\n\n"
        "# Escape flat saddle points with momentum\n"
        "optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)\n"
        "# Or use adaptive moment estimation:\n"
        "# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n"
    ),
    "residual_connection": (
        "# 8. Add Residual Skip Connection\n"
        "import torch.nn as nn\n\n"
        "class ResidualBlock(nn.Module):\n"
        "    def __init__(self, dim):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(dim, dim)\n"
        "        self.act = nn.ReLU()\n"
        "    def forward(self, x):\n"
        "        return x + self.act(self.fc(x))  # Skip connection protects gradients\n"
    ),
}


# -----------------------------------------------------------------------------
# Local Rule-Based Diagnostic Engine
# -----------------------------------------------------------------------------

class LocalTrainingDoctor:
    """
    100% offline, rule-based diagnostic engine. Analyzes audit logs, fault events,
    layer risks, health scores, and gradient trends to produce deep forensic root-cause reports.
    """

    def diagnose(
        self,
        audit_history: List[Dict[str, Any]],
        run_record: Optional[Dict[str, Any]] = None,
        layer_risks: Optional[List[Dict[str, Any]]] = None,
    ) -> TrainingDoctorReport:
        if not audit_history:
            return TrainingDoctorReport(
                headline="No training telemetry available to analyze",
                what_happened="The audit history is empty.",
                when_it_started="N/A",
                affected_layers=[],
                likely_root_cause="Training has not started or telemetry was not loaded.",
                confidence_level="Low",
                confidence_score=0,
                evidence=["Empty audit history list."],
                rollback_recommended=False,
                ranked_remediation_steps=[],
                code_snippets=[],
                limitations=["No epoch data available."],
            )

        fault_records = [r for r in audit_history if r.get("fault_detected")]
        latest_record = audit_history[-1]

        if fault_records:
            primary_fault = fault_records[-1]
            fault_type = primary_fault.get("fault_type", "UNKNOWN_FAULT")
            return self._diagnose_fault(fault_type, primary_fault, audit_history, run_record, layer_risks)

        # In case fault_detected was not explicitly stored in the record but telemetry contains faults:
        # 1. NaN or Inf in loss or nan_layer
        for rec in reversed(audit_history):
            l_val = rec.get("loss")
            if (l_val is not None and (math.isnan(l_val) or math.isinf(l_val))) or rec.get("nan_layer", -1) >= 0:
                rec_synth = dict(rec)
                rec_synth["fault_type"] = FaultType.NAN_INF_PROPAGATION
                rec_synth["evidence"] = {"loss": l_val, "nan_layer": rec.get("nan_layer", -1)}
                return self._diagnose_fault(FaultType.NAN_INF_PROPAGATION, rec_synth, audit_history, run_record, layer_risks)

        # Check for active Forecast early warning before falling back to synthetic post-mortem heuristics
        forecast_info = latest_record.get("forecast")
        if forecast_info and forecast_info.get("overall_risk_level") in ("High", "Critical"):
            return self._diagnose_healthy_or_degraded(audit_history, latest_record, layer_risks)

        # 2. Gradient explosion check in telemetry

        if len(audit_history) >= 2:
            last_grad = latest_record.get("gradient", {}).get("global", {}).get("mean")
            prev_grad = audit_history[-2].get("gradient", {}).get("global", {}).get("mean")
            if last_grad is not None and (last_grad > 100.0 or (prev_grad is not None and last_grad > prev_grad * 10 and last_grad > 5.0)):
                rec_synth = dict(latest_record)
                rec_synth["fault_type"] = FaultType.GRADIENT_EXPLOSION
                rec_synth["evidence"] = {"current_gradient_mean": last_grad, "prev_mean": prev_grad}
                return self._diagnose_fault(FaultType.GRADIENT_EXPLOSION, rec_synth, audit_history, run_record, layer_risks)

        # 3. Loss spike check
        if len(audit_history) >= 2:
            last_loss = latest_record.get("loss")
            prev_loss = audit_history[-2].get("loss")
            if last_loss is not None and prev_loss is not None and prev_loss > 0 and last_loss > prev_loss * 5.0 and last_loss > 5.0:
                rec_synth = dict(latest_record)
                rec_synth["fault_type"] = FaultType.LOSS_SPIKE
                rec_synth["evidence"] = {"spike_ratio": last_loss / prev_loss}
                return self._diagnose_fault(FaultType.LOSS_SPIKE, rec_synth, audit_history, run_record, layer_risks)

        # 4. Check for prolonged loss stagnation (>12 epochs without progress)
        if len(audit_history) >= 12:
            recent_losses = [r.get("loss") for r in audit_history[-12:] if r.get("loss") is not None]
            if len(recent_losses) >= 12 and (max(recent_losses) - min(recent_losses) < 0.005) and recent_losses[-1] > 0.2:
                rec_synth = dict(latest_record)
                rec_synth["fault_type"] = FaultType.TRAINING_STAGNATION
                rec_synth["first_epoch"] = audit_history[-12].get("epoch", 0)
                rec_synth["evidence"] = {"stagnation_epochs": 12, "loss": recent_losses[-1]}
                return self._diagnose_fault(FaultType.TRAINING_STAGNATION, rec_synth, audit_history, run_record, layer_risks)

        # Check for degrading health or unflagged anomalies
        return self._diagnose_healthy_or_degraded(audit_history, latest_record, layer_risks)

    def _diagnose_fault(
        self,
        fault_type: str,
        fault_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        run_record: Optional[Dict[str, Any]],
        layer_risks: Optional[List[Dict[str, Any]]],
    ) -> TrainingDoctorReport:
        evidence: List[str] = []
        code_snippets: List[str] = []
        steps: List[RemediationStep] = []
        affected_layers: List[str] = []

        detected_epoch = fault_record.get("epoch", len(history) - 1)
        suspected_layer = fault_record.get("suspected_layer")
        raw_evidence = fault_record.get("evidence", {})

        if suspected_layer is not None:
            affected_layers.append(f"layer_{suspected_layer}")

        # Extract top risky layers
        if layer_risks:
            for lr in layer_risks[:3]:
                l_name = lr.get("layer", "")
                if l_name and l_name not in affected_layers and lr.get("risk_score", 0) > 30:
                    affected_layers.append(l_name)

        if not affected_layers:
            affected_layers = ["global_network"]

        # Health score delta
        h_start = history[0].get("health_score", 100)
        h_end = fault_record.get("health_score", history[-1].get("health_score", 0))
        evidence.append(f"Training Health Score dropped from {h_start} to {h_end} at epoch {detected_epoch}.")

        # 1. GRADIENT EXPLOSION
        if fault_type == FaultType.GRADIENT_EXPLOSION:
            headline = f"Catastrophic Gradient Explosion Detected at Epoch {detected_epoch}"
            what_happened = (
                f"Backpropagation gradients abruptly exploded to extreme values at epoch {detected_epoch}, "
                f"destabilizing network weights and rendering optimization chaotic."
            )
            when_it_started = f"Epoch {fault_record.get('first_epoch', detected_epoch)}"

            # Evidence citation
            prev_norm = None
            if detected_epoch > 0 and (detected_epoch - 1) < len(history):
                prev_norm = history[detected_epoch - 1].get("gradient", {}).get("global", {}).get("mean")
            curr_norm = fault_record.get("gradient", {}).get("global", {}).get("mean")

            if prev_norm is not None and curr_norm is not None:
                evidence.append(
                    f"Global gradient norm spiked from {prev_norm:.4e} to {curr_norm:.4e} between epochs {detected_epoch-1} and {detected_epoch}."
                )
            if "gradient_norm" in raw_evidence:
                evidence.append(
                    f"Suspected layer {suspected_layer} recorded gradient norm {raw_evidence['gradient_norm']:.4f} "
                    f"exceeding dynamic threshold {raw_evidence.get('threshold', 3.0):.4f}."
                )

            likely_root_cause = (
                "Excessive learning rate, unnormalized input features, or absence of gradient clipping, "
                "causing exponential gradient magnification across deep layers during backpropagation."
            )
            confidence_level = "High"
            confidence_score = 95
            rollback_recommended = True

            steps.append(
                RemediationStep(
                    title="Enable Gradient Clipping",
                    why_it_may_help="Clips gradient norm to a maximum ceiling (e.g. 1.0) before optimizer.step(), preventing parameter divergence.",
                    expected_impact="Instantly eliminates gradient spikes and stabilizes updates.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["clip_grad_norm"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Reduce Learning Rate by 5x-10x",
                    why_it_may_help="Smaller optimizer steps prevent jumping into steep cliffs of the loss surface.",
                    expected_impact="Smooths loss trajectory and keeps updates well-scaled.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["reduce_learning_rate"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Normalize or Standardize Input Data",
                    why_it_may_help="Unscaled feature magnitudes directly scale early layer weight gradients.",
                    expected_impact="Equalizes gradient magnitudes across feature dimensions.",
                    risk_level="Medium",
                    confidence="Medium",
                    code_snippet=SNIPPETS["input_normalization"],
                )
            )
            code_snippets = [SNIPPETS["clip_grad_norm"], SNIPPETS["reduce_learning_rate"]]

        # 2. GRADIENT VANISHING
        elif fault_type == FaultType.GRADIENT_VANISHING:
            headline = f"Severe Gradient Vanishing Diagnosed at Epoch {detected_epoch}"
            what_happened = (
                f"Gradient magnitudes collapsed towards machine epsilon around epoch {detected_epoch}, "
                f"preventing early layers from updating and stalling learning."
            )
            when_it_started = f"Epoch {fault_record.get('first_epoch', max(0, detected_epoch - 5))}"

            curr_norm = fault_record.get("gradient", {}).get("global", {}).get("mean", 0.0)
            evidence.append(f"Mean gradient collapsed to {curr_norm:.4e} (near zero) over recent epochs.")
            if "recent_mean" in raw_evidence:
                evidence.append(f"Recent gradient rolling mean collapsed to {raw_evidence['recent_mean']:.4e}.")

            likely_root_cause = (
                "Saturating activation functions (such as Sigmoid or Tanh in deep architectures) "
                "causing derivative decay (<= 0.25^N), or dying ReLU units whose activations remain negative."
            )
            confidence_level = "High"
            confidence_score = 90
            rollback_recommended = True

            steps.append(
                RemediationStep(
                    title="Switch to LeakyReLU or GELU Activations",
                    why_it_may_help="Ensures non-zero gradients even for negative pre-activations, preventing neuron death.",
                    expected_impact="Revives dead network paths and restores gradient flow.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["leaky_relu_replace"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Add Batch Normalization or LayerNorm",
                    why_it_may_help="Keeps layer inputs in the sensitive, non-saturating regime of activation functions.",
                    expected_impact="Preserves gradient scale across deep layers.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["batchnorm_insert"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Add Residual Skip Connections",
                    why_it_may_help="Provides an uninterrupted gradient highway directly from loss back to initial layers.",
                    expected_impact="Eliminates exponential vanishing in deep architectures.",
                    risk_level="Medium",
                    confidence="High",
                    code_snippet=SNIPPETS["residual_connection"],
                )
            )
            code_snippets = [SNIPPETS["leaky_relu_replace"], SNIPPETS["batchnorm_insert"]]

        # 3. NAN / INF PROPAGATION
        elif fault_type == FaultType.NAN_INF_PROPAGATION:
            batch_idx = fault_record.get("batch_idx", 0)
            headline = f"Immediate NaN/Inf Arithmetic Fault at Epoch {detected_epoch}, Batch {batch_idx}"
            what_happened = (
                f"A NaN (Not a Number) or infinite floating-point value occurred during computation at epoch "
                f"{detected_epoch}, batch {batch_idx}. Training was halted before corrupting weights."
            )
            when_it_started = f"Epoch {detected_epoch}, Batch {batch_idx}"
            evidence.append(f"Loss or layer activations registered NaN/Inf at epoch {detected_epoch}, batch {batch_idx}.")
            if "loss" in raw_evidence:
                evidence.append(f"Computed loss value was: {raw_evidence['loss']}.")

            likely_root_cause = (
                "Numerical domain violation, such as log(0.0) in cross-entropy, division by zero, "
                "or catastrophic overflow following an unmitigated gradient explosion."
            )
            confidence_level = "High"
            confidence_score = 98
            rollback_recommended = True

            steps.append(
                RemediationStep(
                    title="Clamp Logarithm Arguments (Safe Epsilon)",
                    why_it_may_help="Prevents evaluating log(0.0) which yields -Infinity and cascades into NaNs.",
                    expected_impact="Guarantees valid mathematical bounds for loss evaluation.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["safe_log_clamp"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Enable Gradient Clipping",
                    why_it_may_help="Prevents huge gradients from creating Inf values in weight tensors.",
                    expected_impact="Protects floating point numbers from overflow.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["clip_grad_norm"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Inspect Training Data for NaNs / Corrupt Samples",
                    why_it_may_help="Missing values or null fields in dataset tensors will propagate directly into model weights.",
                    expected_impact="Removes bad inputs at the source.",
                    risk_level="Low",
                    confidence="Medium",
                    code_snippet=SNIPPETS["input_normalization"],
                )
            )
            code_snippets = [SNIPPETS["safe_log_clamp"], SNIPPETS["clip_grad_norm"]]

        # 4. LOSS SPIKE
        elif fault_type == FaultType.LOSS_SPIKE:
            headline = f"Abrupt Loss Spike Detected at Epoch {detected_epoch}"
            what_happened = (
                f"Loss suddenly multiplied by several multiples at epoch {detected_epoch}, "
                f"indicating severe prediction error or loss surface disruption."
            )
            when_it_started = f"Epoch {detected_epoch}"
            loss_now = fault_record.get("loss")
            loss_prev = history[detected_epoch - 1].get("loss") if detected_epoch > 0 else None
            if loss_prev and loss_now:
                evidence.append(f"Loss spiked from {loss_prev:.4f} to {loss_now:.4f} (ratio: {loss_now/max(loss_prev, 1e-6):.2f}x).")

            likely_root_cause = (
                "An outlier batch containing corrupt labels/features, a sudden learning rate mismatch, "
                "or a sharp loss barrier encountered during optimization."
            )
            confidence_level = "Medium"
            confidence_score = 82
            rollback_recommended = True

            steps.append(
                RemediationStep(
                    title="Inspect & Filter Outlier Batches",
                    why_it_may_help="Outlier samples with anomalous targets cause enormous prediction errors and destabilizing gradients.",
                    expected_impact="Ensures smooth training batches.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["input_normalization"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Reduce Learning Rate",
                    why_it_may_help="Limits parameter displacement when a difficult batch is encountered.",
                    expected_impact="Prevents single batch errors from destroying trained representations.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["reduce_learning_rate"],
                )
            )
            code_snippets = [SNIPPETS["reduce_learning_rate"]]

        # 5. TRAINING STAGNATION / LOSS STAGNATION
        else:
            headline = f"Loss Stagnation & Optimization Plateau at Epoch {detected_epoch}"
            what_happened = (
                f"Training loss has ceased improving over multiple consecutive epochs ({detected_epoch}). "
                f"Model progress is trapped on a plateau or saddle point."
            )
            when_it_started = f"Epoch {fault_record.get('first_epoch', max(0, detected_epoch - 10))}"
            evidence.append(f"Loss has remained stagnant around {fault_record.get('loss', 0.0):.4f} for >10 epochs.")

            likely_root_cause = (
                "Insufficient learning rate, optimizer trapped in a flat saddle point, "
                "or model capacity too low for the complexity of the dataset."
            )
            confidence_level = "Medium"
            confidence_score = 78
            rollback_recommended = True

            steps.append(
                RemediationStep(
                    title="Add Momentum or Switch to AdamW",
                    why_it_may_help="Momentum accelerates past flat plateaus and helps escape shallow local minima.",
                    expected_impact="Revitalizes descent direction and continues loss reduction.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet=SNIPPETS["add_momentum"],
                )
            )
            steps.append(
                RemediationStep(
                    title="Adjust Learning Rate (Warmup or Cyclical LR)",
                    why_it_may_help="A slightly higher learning rate or learning rate schedule can break stagnation.",
                    expected_impact="Provides kinetic energy to traverse flat loss surfaces.",
                    risk_level="Low",
                    confidence="Medium",
                    code_snippet=SNIPPETS["reduce_learning_rate"],
                )
            )
            code_snippets = [SNIPPETS["add_momentum"]]

        limitations = [
            "Diagnosis generated from aggregate telemetry; inspect individual sample losses if dataset corruption is suspected.",
            "Verify checkpoint integrity before applying rollback.",
        ]

        return TrainingDoctorReport(
            headline=headline,
            what_happened=what_happened,
            when_it_started=when_it_started,
            affected_layers=affected_layers,
            likely_root_cause=likely_root_cause,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            evidence=evidence,
            rollback_recommended=rollback_recommended,
            ranked_remediation_steps=steps,
            code_snippets=code_snippets,
            limitations=limitations,
            mode="Local Rule-Based AI",
        )

    def _diagnose_healthy_or_degraded(
        self,
        history: List[Dict[str, Any]],
        latest: Dict[str, Any],
        layer_risks: Optional[List[Dict[str, Any]]],
    ) -> TrainingDoctorReport:
        score = latest.get("health_score", 100)
        curr_loss = latest.get("loss", 0.0)
        curr_acc = latest.get("accuracy")

        evidence = [
            f"Zero fatal training faults detected across {len(history)} epochs.",
            f"Current Training Health Score is {score}/100.",
            f"Latest loss: {curr_loss:.4f}." + (f" (Accuracy: {curr_acc:.2%})" if curr_acc is not None else ""),
        ]

        # Check if the latest telemetry has an active Failure Forecast
        forecast_info = latest.get("forecast")
        if forecast_info and forecast_info.get("overall_risk_level") in ("High", "Critical"):

            f_level = forecast_info.get("overall_risk_level")
            f_score = forecast_info.get("overall_risk_score", 75)
            f_threat = forecast_info.get("likely_failure_type", "Impending Fault")
            f_epochs = forecast_info.get("estimated_epochs_to_failure", 2)
            f_signals = forecast_info.get("warning_signals", [])
            f_action = forecast_info.get("recommended_action", {})

            headline = f"Early Warning: {f_level} Risk of {f_threat} within ~{f_epochs} Epochs"
            what_happened = (
                f"Statistical forecasting engine detected early failure precursors. "
                f"While no fatal fault has halted training yet, metric trajectories indicate a {f_level.lower()} "
                f"probability of {f_threat} failure within approximately {f_epochs} epochs."
            )
            likely_root_cause = (
                f"Emerging instability in training dynamics: {'; '.join(f_signals[:2])}"
            )
            confidence_level = "High" if f_level == "Critical" else "Medium"
            confidence_score = f_score
            rollback_recommended = False  # Forecasts NEVER mandate automatic rollback; strictly advisory

            evidence_list = [
                f"[FORECAST EARLY WARNING] {f_level} risk score ({f_score}/100) for {f_threat}.",
                f"Estimated epochs until possible divergence: ~{f_epochs} epochs.",
            ]
            evidence_list.extend(f_signals)

            steps = []
            if f_action:
                steps.append(
                    RemediationStep(
                        title=f_action.get("title", "Apply Preventive Action"),
                        why_it_may_help=f_action.get("why_it_may_help", "Mitigates early warning instability."),
                        expected_impact=f_action.get("expected_impact", "Maintains training trajectory."),
                        risk_level=f_action.get("risk_level", "Low"),
                        confidence=f_action.get("confidence", "High"),
                        code_snippet=f_action.get("code_snippet", ""),
                    )
                )
            if not steps:
                steps.append(
                    RemediationStep(
                        title="Enable Gradient Clipping & Checkpoint Safety",
                        why_it_may_help="Caps gradient surges before they escalate into fatal divergence.",
                        expected_impact="Stabilizes parameter updates without loss of representation.",
                        risk_level="Low",
                        confidence="High",
                        code_snippet=SNIPPETS["clip_grad_norm"],
                    )
                )

            return TrainingDoctorReport(
                headline=headline,
                what_happened=what_happened,
                when_it_started=f"Epoch {latest.get('epoch', len(history)-1)} (Precursor Signal)",
                affected_layers=["global_network"],
                likely_root_cause=likely_root_cause,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                evidence=evidence_list,
                rollback_recommended=rollback_recommended,
                ranked_remediation_steps=steps,
                code_snippets=[s.code_snippet for s in steps if s.code_snippet],
                limitations=[
                    "Precursor forecast: This is an advisory early-warning prediction, NOT a confirmed fatal fault.",
                    "Training has not been halted automatically. Review preventive recommendations.",
                ],
                mode="Local Rule-Based AI (Early-Warning Forecaster)",
            )

        # Check if any layer risk is rising
        elevated_risks = [r for r in (layer_risks or []) if r.get("risk_score", 0) > 40]

        if elevated_risks:
            r0 = elevated_risks[0]
            evidence.append(f"Layer '{r0.get('layer')}' shows rising risk ({r0.get('risk_score')}/100): {r0.get('likely_issue')}.")
            headline = f"Training Stable with Preemptive Risk in {r0.get('layer')}"
            what_happened = (
                f"Training is progressing without fatal failures, but early warning indicators in layer "
                f"'{r0.get('layer')}' indicate potential upcoming instability."
            )
            likely_root_cause = r0.get("likely_issue", "Minor gradient fluctuation.")
            confidence_level = "Medium"
            confidence_score = 75
            rollback_recommended = False
            steps = [
                RemediationStep(
                    title=f"Monitor Gradient Dynamics in {r0.get('layer')}",
                    why_it_may_help="Preemptively checking gradient variance avoids sudden explosions or decay in future epochs.",
                    expected_impact="Maintains long-term training stability.",
                    risk_level="Low",
                    confidence="Medium",
                    code_snippet=SNIPPETS["clip_grad_norm"],
                )
            ]
            snippets = [SNIPPETS["clip_grad_norm"]]
        else:
            headline = f"Healthy Training Trajectory (Health {score}/100)"
            what_happened = (
                f"Neural network optimization is proceeding smoothly. Loss is descending stably, "
                f"and all gradient norms remain within safe adaptive bounds."
            )
            likely_root_cause = "No anomalies detected. Network convergence is on schedule."
            confidence_level = "High"
            confidence_score = 95
            rollback_recommended = False
            steps = [
                RemediationStep(
                    title="Continue Training Routine",
                    why_it_may_help="Current hyperparameter configuration is well-matched to dataset complexity.",
                    expected_impact="Target convergence without intervention.",
                    risk_level="Low",
                    confidence="High",
                    code_snippet="# No intervention required. System is healthy.",
                )
            ]
            snippets = []

        return TrainingDoctorReport(
            headline=headline,
            what_happened=what_happened,
            when_it_started="N/A (Healthy)",
            affected_layers=[],
            likely_root_cause=likely_root_cause,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            evidence=evidence,
            rollback_recommended=rollback_recommended,
            ranked_remediation_steps=steps,
            code_snippets=snippets,
            limitations=["Predictive analysis based on current epoch telemetry."],
            mode="Local Rule-Based AI",
        )


# -----------------------------------------------------------------------------
# Optional LLM Diagnostic Layer with Fallback & Privacy Guarantees
# -----------------------------------------------------------------------------

class LLMTrainingDoctor:
    """
    OpenAI-compatible LLM diagnostic assistant.
    Features:
      - Reads solely from environment variables (OPENAI_API_KEY, OPENAI_BASE_URL, MLBLACKBOX_LLM_MODEL)
      - Strictly sanitizes telemetry: NEVER sends weights, datasets, or user code.
      - Enforces strict JSON schema validation.
      - Gracefully and transparently falls back to LocalTrainingDoctor if keys are missing or requests fail.
    """

    def __init__(self, local_doctor: Optional[LocalTrainingDoctor] = None):
        self.local_doctor = local_doctor or LocalTrainingDoctor()

    def is_available(self) -> bool:
        """Returns True only if an API key is configured in the environment."""
        return bool(os.getenv("OPENAI_API_KEY", "").strip())

    def build_sanitized_payload(
        self,
        audit_history: List[Dict[str, Any]],
        run_record: Optional[Dict[str, Any]] = None,
        layer_risks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Builds a privacy-preserving summary.
        STRICT PRIVACY GUARANTEE:
        - No parameter weights or biases.
        - No raw dataset samples or labels.
        - No user source code or local filepaths.
        - Only summary numbers: epochs, losses, gradient norms, health scores.
        """
        total_epochs = len(audit_history)
        latest = audit_history[-1] if audit_history else {}
        faults = [
            {
                "epoch": r.get("epoch"),
                "fault_type": r.get("fault_type"),
                "severity": r.get("severity"),
                "suspected_layer": r.get("suspected_layer"),
                "evidence_summary": str(r.get("evidence", {})),
            }
            for r in audit_history
            if r.get("fault_detected")
        ]

        # Anonymized metric samples (max 10 uniformly spaced epochs)
        step = max(1, total_epochs // 10)
        sampled_telemetry = []
        for i in range(0, total_epochs, step):
            rec = audit_history[i]
            sampled_telemetry.append({
                "epoch": rec.get("epoch"),
                "loss": rec.get("loss"),
                "accuracy": rec.get("accuracy"),
                "grad_mean": rec.get("gradient", {}).get("global", {}).get("mean"),
                "health_score": rec.get("health_score"),
            })

        sanitized_risks = [
            {
                "layer": lr.get("layer"),
                "risk_score": lr.get("risk_score"),
                "trend": lr.get("trend"),
                "likely_issue": lr.get("likely_issue"),
            }
            for lr in (layer_risks or [])[:5]
        ]

        return {
            "total_epochs": total_epochs,
            "latest_loss": latest.get("loss"),
            "latest_health_score": latest.get("health_score"),
            "faults_detected": faults,
            "sampled_epoch_metrics": sampled_telemetry,
            "layer_risk_summary": sanitized_risks,
            "run_status": run_record.get("status") if run_record else "unknown",
        }

    def diagnose(
        self,
        audit_history: List[Dict[str, Any]],
        run_record: Optional[Dict[str, Any]] = None,
        layer_risks: Optional[List[Dict[str, Any]]] = None,
    ) -> TrainingDoctorReport:
        """
        Attempts LLM diagnosis. If credentials are unset or the call fails,
        transparently returns LocalTrainingDoctor diagnosis.
        """
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return self.local_doctor.diagnose(audit_history, run_record, layer_risks)

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("MLBLACKBOX_LLM_MODEL", "gpt-4o-mini")

        sanitized_summary = self.build_sanitized_payload(audit_history, run_record, layer_risks)

        prompt = (
            "You are MLBlackBox Training Doctor, an expert ML observability diagnostician.\n"
            "Analyze this anonymized neural network training telemetry and provide a structured diagnosis.\n"
            "Return valid JSON matching this exact schema:\n"
            "{\n"
            '  "headline": "Short title",\n'
            '  "what_happened": "Clear explanation",\n'
            '  "when_it_started": "Epoch or timeframe",\n'
            '  "affected_layers": ["layer_name"],\n'
            '  "likely_root_cause": "Detailed technical root cause",\n'
            '  "confidence_level": "High" | "Medium" | "Low",\n'
            '  "confidence_score": 0-100,\n'
            '  "evidence": ["bullet 1", "bullet 2"],\n'
            '  "rollback_recommended": true | false,\n'
            '  "remediation_steps": [\n'
            "    {\n"
            '      "title": "Action title",\n'
            '      "why_it_may_help": "Reasoning",\n'
            '      "expected_impact": "Impact",\n'
            '      "risk_level": "Low" | "Medium" | "High",\n'
            '      "confidence": "High" | "Medium" | "Low",\n'
            '      "code_snippet": "PyTorch code snippet"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"TELEMETRY SUMMARY:\n{json.dumps(sanitized_summary, indent=2)}"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a professional ML reliability assistant. Output strictly JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                return self._parse_and_validate(parsed)
        except Exception:
            # Fall back to local engine without crashing
            report = self.local_doctor.diagnose(audit_history, run_record, layer_risks)
            report.limitations.append("LLM request was unavailable or failed; seamlessly reverted to Local Rule-Based AI.")
            return report

    def _parse_and_validate(self, data: Dict[str, Any]) -> TrainingDoctorReport:
        """Validates and sanitizes structured LLM response."""
        headline = str(data.get("headline", "AI Diagnostic Hypothesis"))
        what_happened = str(data.get("what_happened", ""))
        when_it_started = str(data.get("when_it_started", "Unknown"))
        affected_layers = [str(l) for l in data.get("affected_layers", [])]
        likely_root_cause = str(data.get("likely_root_cause", ""))
        confidence_level = str(data.get("confidence_level", "Medium"))
        if confidence_level not in ("Low", "Medium", "High"):
            confidence_level = "Medium"

        try:
            confidence_score = max(0, min(100, int(data.get("confidence_score", 75))))
        except Exception:
            confidence_score = 75

        evidence = [str(e) for e in data.get("evidence", [])]
        rollback_recommended = bool(data.get("rollback_recommended", False))

        remed_list = []
        snippets = []
        for s in data.get("remediation_steps", []):
            code = str(s.get("code_snippet", ""))
            if code:
                snippets.append(code)
            remed_list.append(
                RemediationStep(
                    title=str(s.get("title", "Remediation Step")),
                    why_it_may_help=str(s.get("why_it_may_help", "")),
                    expected_impact=str(s.get("expected_impact", "")),
                    risk_level=str(s.get("risk_level", "Low")),
                    confidence=str(s.get("confidence", "Medium")),
                    code_snippet=code,
                )
            )

        return TrainingDoctorReport(
            headline=headline,
            what_happened=what_happened,
            when_it_started=when_it_started,
            affected_layers=affected_layers,
            likely_root_cause=likely_root_cause,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            evidence=evidence,
            rollback_recommended=rollback_recommended,
            ranked_remediation_steps=remed_list,
            code_snippets=snippets,
            limitations=[
                "AI-generated hypothesis based on aggregate telemetry metrics.",
                "Review code snippets before executing in production.",
            ],
            mode="AI-Generated Hypothesis (LLM)",
        )


# -----------------------------------------------------------------------------
# Convenience Entrypoint
# -----------------------------------------------------------------------------

def diagnose_run(
    audit_history: List[Dict[str, Any]],
    run_record: Optional[Dict[str, Any]] = None,
    layer_risks: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
) -> TrainingDoctorReport:
    """
    Primary interface for diagnosing training runs.
    If use_llm is True and credentials are present, attempts LLM explanation;
    otherwise executes the local offline rule-based doctor.
    """
    local_doctor = LocalTrainingDoctor()
    if use_llm:
        llm_doctor = LLMTrainingDoctor(local_doctor=local_doctor)
        return llm_doctor.diagnose(audit_history, run_record, layer_risks)
    return local_doctor.diagnose(audit_history, run_record, layer_risks)
