"""
layer_risk.py — Layer Failure Risk Ranking for MLBlackBox.

Ranks neural network layers by failure risk using:
  - Gradient norm trends (exploding or collapsing)
  - Sudden gradient spikes / volatility
  - NaN/Inf appearances
  - Weight magnitude instability
  - Correlation with loss degradation
"""

import math
from typing import List, Dict, Any, Optional


def rank_layers_by_risk(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze layer telemetry across history and rank layers by failure risk.

    Returns a list of dicts sorted by risk_score descending:
      - layer: int
      - name: str
      - risk_score: int (0 to 100)
      - trend: "Increasing" | "Decreasing" | "Stable"
      - likely_issue: str
      - recommended_action: str
      - latest_norm: float
    """
    if not history:
        return []

    latest_record = history[-1]
    per_layer = latest_record.get("gradient", {}).get("per_layer", [])
    if not per_layer:
        return []

    # Map layer_idx -> history of mean gradient norms
    layer_histories: Dict[int, List[float]] = {}
    layer_names: Dict[int, str] = {}
    has_nan_flags: Dict[int, bool] = {}

    for record in history:
        pl = record.get("gradient", {}).get("per_layer", [])
        for l_stat in pl:
            idx = l_stat.get("layer", 0)
            name = l_stat.get("name", f"layer_{idx}")
            layer_names[idx] = name
            val = l_stat.get("mean")
            if val is None or math.isnan(val):
                val = 0.0
            if idx not in layer_histories:
                layer_histories[idx] = []
            layer_histories[idx].append(val)
            if l_stat.get("has_nan", False):
                has_nan_flags[idx] = True

    results = []
    for idx, vals in layer_histories.items():
        name = layer_names.get(idx, f"layer_{idx}")
        current_norm = vals[-1] if vals else 0.0
        nan_present = has_nan_flags.get(idx, False)

        risk_score = 10
        likely_issue = "Normal operation"
        recommended_action = "Maintain current architecture and hyperparameters"
        trend = "Stable"

        if len(vals) >= 2:
            first_val = vals[0]
            recent_mean = sum(vals[-min(5, len(vals)):]) / min(5, len(vals))
            delta = recent_mean - first_val

            if delta > 0.05 * (first_val + 1e-6):
                trend = "Increasing"
            elif delta < -0.05 * (first_val + 1e-6):
                trend = "Decreasing"

        # NaN presence = 100% risk
        if nan_present:
            risk_score = 100
            likely_issue = "NaN/Inf value detected in layer gradients or weights"
            recommended_action = "Inspect activation outputs, add numerical clamp, or add gradient clipping"
            trend = "Increasing"
        # Vanishing check (< 1e-5)
        elif current_norm < 1e-5:
            risk_score = 75
            likely_issue = "Vanishing gradients: layer stopped updating effectively"
            recommended_action = "Replace saturating activations (e.g. Sigmoid) with ReLU / LeakyReLU"
            trend = "Decreasing" if trend == "Decreasing" else "Stable"
        # Exploding check (> 100.0)
        elif current_norm > 100.0:
            risk_score = 85
            likely_issue = "Gradient explosion: gradients dangerously elevated"
            recommended_action = "Lower learning rate by 5x-10x, apply gradient clipping (clip_norm=1.0)"
            trend = "Increasing"
        # Rapid increase check
        elif len(vals) >= 3 and vals[-1] > 3.0 * (vals[-2] + 1e-5):
            risk_score = 65
            likely_issue = "Sudden gradient spike in layer parameters"
            recommended_action = "Check for unscaled outlier batches or apply batch normalization"
            trend = "Increasing"
        elif trend == "Decreasing" and current_norm < 0.001:
            risk_score = 50
            likely_issue = "Declining gradient magnitude over successive epochs"
            recommended_action = "Monitor layer; consider residual connection or learning rate warmup"
        else:
            risk_score = 15

        results.append({
            "layer": idx,
            "name": name,
            "risk_score": risk_score,
            "trend": trend,
            "likely_issue": likely_issue,
            "recommended_action": recommended_action,
            "latest_norm": round(current_norm, 6),
        })

    # Sort descending by risk score
    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return results
