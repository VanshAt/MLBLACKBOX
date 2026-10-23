"""
ai/failure_forecaster.py — Lightweight early-warning failure forecasting system.

Calculates:
- overall_risk_level: "Low", "Medium", "High", "Critical"
- overall_risk_score: 0-100
- likely_failure_type: predicted failure type
- confidence: 0-100
- estimated_epochs_to_failure: estimated epochs before failure occurs
- warning_signals: list of detected early warning indicators
- recommended_action: structured preventive remediation
- per_fault_risk_scores: dict mapping fault types to 0-100 scores

100% offline, lightweight, deterministic, heuristic & statistical.
No deep learning models required.
ADVISORY ONLY: Forecasts NEVER trigger automatic rollback or halts.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from tracker.fault_detector import FaultType


class ForecastRiskLevel:
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ForecastEventType:
    FORECAST_LOW = "FORECAST_LOW"
    FORECAST_MEDIUM = "FORECAST_MEDIUM"
    FORECAST_HIGH = "FORECAST_HIGH"
    FORECAST_CRITICAL = "FORECAST_CRITICAL"


@dataclass
class PreventiveRecommendation:
    title: str
    why_it_may_help: str
    expected_impact: str
    risk_level: str  # "Low", "Medium", "High"
    confidence: str  # "Low", "Medium", "High"
    code_snippet: str = ""


@dataclass
class FailureForecast:
    epoch: int
    overall_risk_level: str  # "Low", "Medium", "High", "Critical"
    overall_risk_score: int  # 0 to 100
    likely_failure_type: str  # FaultType string or "None"
    confidence: int  # 0 to 100
    estimated_epochs_to_failure: Optional[int]
    warning_signals: List[str]
    recommended_action: PreventiveRecommendation
    per_fault_risk_scores: Dict[str, int]
    event_type: str = ForecastEventType.FORECAST_LOW

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommended_action"] = asdict(self.recommended_action)
        return d


class FailureForecaster:
    """
    Rolling statistical forecaster that detects precursor signals of failure.
    Uses exponential moving averages, z-scores, rate of change, and threshold heuristics.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def forecast(
        self,
        current_record: Dict[str, Any],
        history: List[Dict[str, Any]],
        layer_risks: Optional[List[Dict[str, Any]]] = None,
    ) -> FailureForecast:
        epoch = current_record.get("epoch", len(history))
        all_records = list(history)
        if not all_records or all_records[-1].get("epoch") != epoch:
            all_records.append(current_record)

        # 1. Component Risk Evaluators
        exp_score, exp_signals = self._eval_explosion_risk(current_record, all_records, layer_risks)
        van_score, van_signals = self._eval_vanishing_risk(current_record, all_records)
        nan_score, nan_signals = self._eval_nan_risk(current_record, all_records)
        spk_score, spk_signals = self._eval_spike_risk(current_record, all_records)
        stg_score, stg_signals = self._eval_stagnation_risk(current_record, all_records)

        per_fault_scores = {
            FaultType.GRADIENT_EXPLOSION: exp_score,
            FaultType.GRADIENT_VANISHING: van_score,
            FaultType.NAN_INF_PROPAGATION: nan_score,
            FaultType.LOSS_SPIKE: spk_score,
            FaultType.TRAINING_STAGNATION: stg_score,
        }

        # 2. Overall Risk Score & Primary Threat
        # Overall risk takes maximum component threat with fractional blending from secondary threats
        sorted_faults = sorted(per_fault_scores.items(), key=lambda x: x[1], reverse=True)
        primary_threat, max_score = sorted_faults[0]
        secondary_blend = sum(s for _, s in sorted_faults[1:]) / (len(sorted_faults) - 1) * 0.15
        overall_score = min(100, int(round(max_score + secondary_blend)))

        # Consolidate Warning Signals
        warning_signals = []
        if primary_threat == FaultType.GRADIENT_EXPLOSION:
            warning_signals.extend(exp_signals)
        elif primary_threat == FaultType.GRADIENT_VANISHING:
            warning_signals.extend(van_signals)
        elif primary_threat == FaultType.NAN_INF_PROPAGATION:
            warning_signals.extend(nan_signals)
        elif primary_threat == FaultType.LOSS_SPIKE:
            warning_signals.extend(spk_signals)
        elif primary_threat == FaultType.TRAINING_STAGNATION:
            warning_signals.extend(stg_signals)

        # Append secondary signals if elevated
        for f_type, f_score in sorted_faults[1:]:
            if f_score >= 40:
                if f_type == FaultType.GRADIENT_EXPLOSION:
                    warning_signals.extend(exp_signals[:1])
                elif f_type == FaultType.GRADIENT_VANISHING:
                    warning_signals.extend(van_signals[:1])
                elif f_type == FaultType.NAN_INF_PROPAGATION:
                    warning_signals.extend(nan_signals[:1])
                elif f_type == FaultType.LOSS_SPIKE:
                    warning_signals.extend(spk_signals[:1])
                elif f_type == FaultType.TRAINING_STAGNATION:
                    warning_signals.extend(stg_signals[:1])

        if not warning_signals:
            warning_signals.append("All rolling metrics and layer gradient statistics are operating within nominal thresholds.")

        # 3. Overall Risk Level Classification
        if overall_score >= 80:
            risk_level = ForecastRiskLevel.CRITICAL
            event_type = ForecastEventType.FORECAST_CRITICAL
            est_epochs = 1
            confidence = min(98, 70 + overall_score // 4)
        elif overall_score >= 55:
            risk_level = ForecastRiskLevel.HIGH
            event_type = ForecastEventType.FORECAST_HIGH
            est_epochs = 2
            confidence = min(90, 60 + overall_score // 3)
        elif overall_score >= 30:
            risk_level = ForecastRiskLevel.MEDIUM
            event_type = ForecastEventType.FORECAST_MEDIUM
            est_epochs = 5
            confidence = 65
        else:
            risk_level = ForecastRiskLevel.LOW
            event_type = ForecastEventType.FORECAST_LOW
            est_epochs = None
            confidence = 90
            primary_threat = "None"

        # 4. Actionable Recommendation
        rec = self._generate_preventive_action(risk_level, primary_threat)

        return FailureForecast(
            epoch=epoch,
            overall_risk_level=risk_level,
            overall_risk_score=overall_score,
            likely_failure_type=primary_threat,
            confidence=confidence,
            estimated_epochs_to_failure=est_epochs,
            warning_signals=warning_signals,
            recommended_action=rec,
            per_fault_risk_scores=per_fault_scores,
            event_type=event_type,
        )

    # -------------------------------------------------------------------------
    # Risk Evaluator: Gradient Explosion
    # -------------------------------------------------------------------------
    def _eval_explosion_risk(
        self,
        current: Dict[str, Any],
        history: List[Dict[str, Any]],
        layer_risks: Optional[List[Dict[str, Any]]],
    ) -> (int, List[str]):
        signals = []
        score = 0
        curr_g = current.get("gradient", {}).get("global", {}).get("mean")
        if curr_g is None or math.isnan(curr_g):
            return 0, signals

        # Immediate extreme magnitude
        if curr_g > 1000.0:
            score = max(score, 95)
            signals.append(f"Gradient norm ({curr_g:.2e}) has reached catastrophic explosion levels.")
        elif curr_g > 50.0:
            score = max(score, 75)
            signals.append(f"Gradient norm ({curr_g:.2f}) is abnormally elevated above safe limits.")

        # Rate of change against rolling baseline
        if len(history) >= 2:
            prev_g = history[-2].get("gradient", {}).get("global", {}).get("mean")
            if prev_g is not None and prev_g > 0:
                ratio = curr_g / prev_g
                if ratio > 5.0 and curr_g > 2.0:
                    score = max(score, min(95, int(50 + ratio * 5)))
                    signals.append(f"Rapid gradient acceleration: {ratio:.1f}x increase over previous epoch ({prev_g:.3f} -> {curr_g:.3f}).")
                elif ratio > 2.0 and curr_g > 1.5:
                    score = max(score, 45)
                    signals.append(f"Gradient norm climbing rapidly: {ratio:.1f}x growth ({prev_g:.3f} -> {curr_g:.3f}).")

        # Rolling z-score check
        if len(history) >= 3:
            window = [
                h.get("gradient", {}).get("global", {}).get("mean")
                for h in history[-self.window_size:-1]
                if h.get("gradient", {}).get("global", {}).get("mean") is not None
            ]
            if len(window) >= 2:
                mean = sum(window) / len(window)
                std = math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
                if std > 1e-6:
                    z = (curr_g - mean) / std
                    if z > 3.0 and curr_g > 1.0:
                        score = max(score, 70)
                        signals.append(f"Gradient mean exceeds rolling average by {z:.1f} standard deviations.")

        # Layer specific explosion risk
        if layer_risks:
            for lr in layer_risks:
                if lr.get("trend") in ("EXPLODING", "RISING") and lr.get("risk_score", 0) > 40:
                    score = max(score, lr.get("risk_score", 0))
                    signals.append(f"Layer '{lr.get('layer')}' gradient trajectory marked as {lr.get('trend')} (Risk {lr.get('risk_score')}/100).")
                    break

        return min(100, score), signals

    # -------------------------------------------------------------------------
    # Risk Evaluator: Gradient Vanishing
    # -------------------------------------------------------------------------
    def _eval_vanishing_risk(
        self, current: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> (int, List[str]):
        signals = []
        score = 0
        curr_g = current.get("gradient", {}).get("global", {}).get("mean")
        if curr_g is None or math.isnan(curr_g):
            return 0, signals

        if curr_g < 1e-6 and curr_g >= 0:
            score = max(score, 90)
            signals.append(f"Gradient norm ({curr_g:.2e}) is near machine zero, signaling complete gradient death.")
        elif curr_g < 1e-4 and len(history) >= 3:
            score = max(score, 65)
            signals.append(f"Persistent micro-gradients ({curr_g:.2e}) detected across consecutive epochs.")

        # Persistent multi-epoch decay
        if len(history) >= 5:
            recent_g = [
                h.get("gradient", {}).get("global", {}).get("mean")
                for h in history[-5:]
                if h.get("gradient", {}).get("global", {}).get("mean") is not None
            ]
            if len(recent_g) == 5:
                is_decaying = all(recent_g[i] > recent_g[i+1] for i in range(4))
                if is_decaying and recent_g[-1] < 0.05:
                    score = max(score, 60)
                    signals.append(f"Monotonic gradient decay observed over 5 consecutive epochs ({recent_g[0]:.4f} -> {recent_g[-1]:.4f}).")

        return min(100, score), signals

    # -------------------------------------------------------------------------
    # Risk Evaluator: NaN / Inf Propagation
    # -------------------------------------------------------------------------
    def _eval_nan_risk(
        self, current: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> (int, List[str]):
        signals = []
        score = 0
        loss = current.get("loss")
        nan_layer = current.get("nan_layer", -1)

        # Confirmed or immediate nan
        if (loss is not None and (math.isnan(loss) or math.isinf(loss))) or nan_layer >= 0:
            score = 100
            signals.append(f"NaN/Inf detected in loss or layer {nan_layer} activations.")
            return score, signals

        # Near-Inf or extreme loss value
        if loss is not None and loss > 1e10:
            score = max(score, 95)
            signals.append(f"Loss magnitude ({loss:.2e}) has reached numerical instability boundaries.")
        elif loss is not None and loss > 1e6:
            score = max(score, 75)
            signals.append(f"Extremely high loss ({loss:.2e}) threatens arithmetic overflow.")


        # Extreme weight magnitudes
        w_max = current.get("weight_magnitude", {}).get("global", {}).get("max")
        if w_max is not None and w_max > 10000.0:
            score = max(score, 70)
            signals.append(f"Maximum network weight ({w_max:.1f}) threatens layer activation overflow.")

        return min(100, score), signals

    # -------------------------------------------------------------------------
    # Risk Evaluator: Loss Spike
    # -------------------------------------------------------------------------
    def _eval_spike_risk(
        self, current: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> (int, List[str]):
        signals = []
        score = 0
        curr_loss = current.get("loss")
        if curr_loss is None or math.isnan(curr_loss):
            return 0, signals

        if len(history) >= 2:
            prev_loss = history[-2].get("loss")
            if prev_loss is not None and prev_loss > 0 and not math.isnan(prev_loss):
                ratio = curr_loss / prev_loss
                if curr_loss > 1e10:
                    # When loss reaches 1e10+, it is numerical overflow / NaN precursor
                    score = max(score, 70)
                    signals.append(f"Astronomical loss ({curr_loss:.2e}) reached overflow scale.")
                elif ratio > 5.0 and curr_loss > 2.0:
                    score = max(score, min(95, int(40 + ratio * 8)))
                    signals.append(f"Loss multiplied by {ratio:.1f}x relative to previous epoch ({prev_loss:.4f} -> {curr_loss:.4f}).")
                elif ratio > 2.5 and curr_loss > 1.0:
                    score = max(score, 50)
                    signals.append(f"Loss rose sharply by {ratio:.1f}x ({prev_loss:.4f} -> {curr_loss:.4f}).")


        # Divergence between loss and accuracy
        curr_acc = current.get("accuracy")
        if curr_acc is not None and len(history) >= 2:
            prev_acc = history[-2].get("accuracy")
            if prev_acc is not None and prev_acc - curr_acc > 0.25:
                score = max(score, 55)
                signals.append(f"Severe accuracy drop ({prev_acc:.1%} -> {curr_acc:.1%}) concurrent with loss destabilization.")

        return min(100, score), signals

    # -------------------------------------------------------------------------
    # Risk Evaluator: Training Stagnation
    # -------------------------------------------------------------------------
    def _eval_stagnation_risk(
        self, current: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> (int, List[str]):
        signals = []
        score = 0
        if len(history) < 8:
            return 0, signals

        losses = [h.get("loss") for h in history[-8:] if h.get("loss") is not None and not math.isnan(h.get("loss", 0.0))]
        if len(losses) == 8 and losses[-1] > 0.2:
            spread = max(losses) - min(losses)
            if spread < 0.002:
                score = max(score, 80)
                signals.append(f"Complete loss plateau: variation is under 0.002 over the last 8 epochs.")
            elif spread < 0.01:
                score = max(score, 55)
                signals.append(f"Loss is stagnating: minimal change ({spread:.4f} spread) across 8 consecutive epochs.")

        return min(100, score), signals

    # -------------------------------------------------------------------------
    # Preventive Action Generator
    # -------------------------------------------------------------------------
    def _generate_preventive_action(
        self, risk_level: str, failure_type: str
    ) -> PreventiveRecommendation:
        if risk_level == ForecastRiskLevel.LOW:
            return PreventiveRecommendation(
                title="Continue Training Routine",
                why_it_may_help="Telemetry trends indicate normal convergence and stable gradient dynamics.",
                expected_impact="Smooth convergence without intervention.",
                risk_level="Low",
                confidence="High",
                code_snippet="# Training is healthy. No intervention required.",
            )

        if risk_level == ForecastRiskLevel.MEDIUM:
            if failure_type == FaultType.GRADIENT_EXPLOSION:
                return PreventiveRecommendation(
                    title="Monitor Risky Layers & Consider Learning Rate Reduction",
                    why_it_may_help="Gradients are exhibiting upward variance that may precede explosive divergence.",
                    expected_impact="Smooths loss trajectory before instability escalates.",
                    risk_level="Low",
                    confidence="Medium",
                    code_snippet=(
                        "# Consider preemptively lowering learning rate:\n"
                        "for pg in optimizer.param_groups:\n"
                        "    pg['lr'] *= 0.5\n"
                    ),
                )
            elif failure_type == FaultType.TRAINING_STAGNATION:
                return PreventiveRecommendation(
                    title="Evaluate Momentum or Learning Rate Schedule",
                    why_it_may_help="Loss progression has slowed, signaling an approaching saddle point.",
                    expected_impact="Re-energizes gradient descent to traverse flat loss regions.",
                    risk_level="Low",
                    confidence="Medium",
                    code_snippet="# Add momentum or scheduler step to escape plateau",
                )
            else:
                return PreventiveRecommendation(
                    title="Increase Monitoring Frequency",
                    why_it_may_help="Moderate metric fluctuations detected; monitor next 2-3 epochs closely.",
                    expected_impact="Ensures early detection if instability worsens.",
                    risk_level="Low",
                    confidence="Medium",
                )

        if risk_level == ForecastRiskLevel.HIGH:
            return PreventiveRecommendation(
                title="Save Checkpoint & Enable Gradient Clipping",
                why_it_may_help="High probability of impending divergence within 2 epochs.",
                expected_impact="Bounds gradient magnitudes to 1.0, neutralizing exponential growth.",
                risk_level="Medium",
                confidence="High",
                code_snippet=(
                    "# Enable gradient clipping before the next optimizer step:\n"
                    "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
                ),
            )

        # CRITICAL
        return PreventiveRecommendation(
            title="Preemptively Pause Training or Roll Back",
            why_it_may_help="Critical instability threshold reached. Immediate divergence or numerical overflow is imminent.",
            expected_impact="Preserves clean model parameters and avoids wasting compute on corrupt iterations.",
            risk_level="High",
            confidence="High",
            code_snippet=(
                "# Roll back to the last clean checkpoint before resuming:\n"
                "trainer.resume_from_last_healthy_checkpoint(max_grad_norm=1.0)\n"
            ),
        )
