"""
health_score.py — Training Health Score calculation for MLBlackBox.

Computes a composite health score from 0 to 100 based on:
  - Loss stability (spikes, explosive variance)
  - Gradient stability (exploding, vanishing, erratic magnitude)
  - NaN/Inf risk (presence of non-finite numbers)
  - Learning progress (steady descent vs stagnation)
  - Stagnation risk
  - Checkpoint health
"""

import math
from typing import Dict, Any, List, Optional, Tuple


def calculate_health_score(
    current_record: Dict[str, Any],
    history: List[Dict[str, Any]],
    stagnation_threshold: int = 15,
) -> Tuple[int, str, Dict[str, float]]:
    """
    Compute training health score (0-100), concise explanation, and component breakdown.

    Returns:
        (score, explanation, components)
    """
    loss = current_record.get("loss", 0.0)
    nan_layer = current_record.get("nan_layer", -1)
    fault_detected = current_record.get("fault_detected")

    # 1. Immediate Critical Penalties
    if loss is None or math.isnan(loss) or math.isinf(loss) or nan_layer >= 0:
        return (
            0,
            "Health 0/100: Critical failure — NaN or Inf detected in loss/gradients.",
            {"loss_stability": 0.0, "gradient_stability": 0.0, "nan_risk": 0.0, "progress": 0.0, "stagnation_risk": 0.0}
        )

    # Component scores out of 20 each (total 100)
    loss_stability = 20.0
    gradient_stability = 20.0
    nan_risk = 20.0
    progress = 20.0
    stagnation_risk = 20.0

    notes: List[str] = []

    # Evaluate against history if available
    if history and len(history) >= 2:
        prev_record = history[-2]
        prev_loss = prev_record.get("loss")

        # Loss spike check
        if prev_loss and prev_loss > 0 and not math.isnan(prev_loss):
            ratio = loss / prev_loss
            if ratio > 3.0:
                loss_stability = max(0.0, 20.0 - min(20.0, (ratio - 1.0) * 5.0))
                notes.append(f"sudden loss increase ({ratio:.1f}x)")
            elif ratio > 1.5:
                loss_stability = 12.0
                notes.append("moderate loss bump")

        # Gradient stability check
        g_curr = current_record.get("gradient", {}).get("global", {})
        g_mean = g_curr.get("mean", 0.0)
        if g_mean is not None and not math.isnan(g_mean):
            if g_mean < 1e-6:
                gradient_stability = 5.0
                notes.append("gradients collapsing (< 1e-6)")
            elif g_mean > 1000.0:
                gradient_stability = 2.0
                notes.append("extremely high gradient magnitude")
            elif g_mean > 50.0:
                gradient_stability = 10.0
                notes.append("elevated gradient magnitude")

        # Check per-layer declining gradients
        per_layer = current_record.get("gradient", {}).get("per_layer", [])
        declining_layers = []
        for l in per_layer:
            l_mean = l.get("mean", 0.0)
            if l_mean is not None and not math.isnan(l_mean) and l_mean < 1e-5:
                declining_layers.append(str(l.get("layer", l.get("name", "?"))))
        if declining_layers:
            gradient_stability = max(4.0, gradient_stability - len(declining_layers) * 3.0)
            notes.append(f"weak gradients in layer(s) {', '.join(declining_layers[:2])}")

        # Stagnation check over recent epochs
        recent = history[-min(stagnation_threshold, len(history)):]
        recent_losses = [r.get("loss") for r in recent if r.get("loss") is not None and not math.isnan(r.get("loss"))]
        if len(recent_losses) >= 5:
            delta = max(recent_losses) - min(recent_losses)
            if delta < 0.001 and recent_losses[-1] > 0.05:
                stagnation_risk = 5.0
                notes.append("training stagnated with minimal loss delta")
            elif delta < 0.005 and recent_losses[-1] > 0.05:
                stagnation_risk = 12.0

        # Progress check
        first_recent = recent_losses[0] if recent_losses else loss
        if loss < first_recent:
            progress = 20.0
        else:
            progress = 14.0

    if fault_detected:
        loss_stability = min(loss_stability, 5.0)
        gradient_stability = min(gradient_stability, 5.0)
        notes.append(f"fault flagged: {fault_detected}")

    total_score = int(round(loss_stability + gradient_stability + nan_risk + progress + stagnation_risk))
    total_score = max(0, min(100, total_score))

    if total_score >= 85:
        verdict = "Training is healthy and progressing smoothly."
    elif total_score >= 70:
        verdict = "Training is stable, but watch for mild variations."
    elif total_score >= 50:
        verdict = "Training stability is degrading; monitor layer dynamics."
    else:
        verdict = "High risk of failure; intervention recommended."

    if notes:
        explanation = f"Health {total_score}/100: {verdict} Note: {'; '.join(notes[:2])}."
    else:
        explanation = f"Health {total_score}/100: {verdict}"

    components = {
        "loss_stability": round(loss_stability, 1),
        "gradient_stability": round(gradient_stability, 1),
        "nan_risk": round(nan_risk, 1),
        "progress": round(progress, 1),
        "stagnation_risk": round(stagnation_risk, 1),
    }

    return total_score, explanation, components
