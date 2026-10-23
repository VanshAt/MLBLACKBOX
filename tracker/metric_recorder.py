"""
metric_recorder.py — Audit trail recorder for MLBlackBox.

Records a comprehensive per-epoch log to logs/audit_log.json.
Think of it as the flight data recorder — logs everything continuously.
The fault detector and backtracker read this to identify trends.

Per-epoch record schema:
{
    "epoch": 14,
    "loss": 0.342,
    "accuracy": 0.91,
    "gradient": {
        "global": {"max": ..., "min": ..., "mean": ..., "std": ...},
        "per_layer": [{"layer": 0, "max": ..., ...}, ...]
    },
    "weight_magnitude": {"max": ..., "min": ..., "mean": ...},
    "learning_rate": 0.01,
    "epoch_time_ms": 142.3,
    "batch_idx": 47,
    "nan_layer": -1,
    "fault_detected": null or "GRADIENT_EXPLOSION"
}
"""

import json
import os
from typing import List, Optional, Dict, Any


AUDIT_LOG_PATH = os.path.join("logs", "audit_log.json")


class MetricRecorder:
    """
    Maintains the full audit trail of a training run.

    Appends one record per epoch to audit_log.json.
    Provides trend computation (rate of change) over a sliding window
    that the fault detector uses to distinguish gradual decay from sudden spikes.
    """

    def __init__(self, log_path: str = AUDIT_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # In-memory history for fast trend computation (mirrors the file)
        self._history: List[Dict[str, Any]] = []

        # Load existing log if present (supports resuming a training run)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._history = []

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        epoch: int,
        loss: float,
        accuracy: Optional[float],
        gradient_stats: Dict[str, Any],
        weight_stats: Dict[str, float],
        learning_rate: float,
        epoch_time_ms: float,
        batch_idx: int,
        nan_layer: int = -1,
        fault_detected: Optional[str] = None,
        health_score: Optional[int] = None,
        health_explanation: Optional[str] = None,
        layer_risks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Append one epoch record to the audit log.
        """
        # Automatically compute health score if not provided
        if health_score is None:
            from tracker.health_score import calculate_health_score
            temp_record = {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "gradient": gradient_stats,
                "nan_layer": nan_layer,
                "fault_detected": fault_detected,
            }
            computed_score, computed_exp, _ = calculate_health_score(temp_record, self._history)
            health_score = computed_score
            health_explanation = computed_exp

        # Automatically calculate failure forecast
        forecast_data = None
        try:
            from ai.failure_forecaster import FailureForecaster
            forecaster = FailureForecaster()
            temp_rec = {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "gradient": gradient_stats,
                "weight_magnitude": weight_stats,
                "nan_layer": nan_layer,
                "health_score": health_score,
            }
            fc = forecaster.forecast(temp_rec, self._history, layer_risks)
            forecast_data = fc.to_dict()
        except Exception:
            forecast_data = None

        record = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "gradient": gradient_stats,
            "weight_magnitude": weight_stats,
            "learning_rate": learning_rate,
            "epoch_time_ms": epoch_time_ms,
            "batch_idx": batch_idx,
            "nan_layer": nan_layer,
            "fault_detected": fault_detected,
            "health_score": health_score,
            "health_explanation": health_explanation,
            "layer_risks": layer_risks or [],
            "forecast": forecast_data,
        }

        self._history.append(record)
        self._flush()
        return record


    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self) -> List[Dict[str, Any]]:
        """Return the full in-memory history list."""
        return list(self._history)

    def get_epoch(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Return the record for a specific epoch, or None if not found."""
        for rec in self._history:
            if rec["epoch"] == epoch:
                return rec
        return None

    def get_last_n(self, n: int) -> List[Dict[str, Any]]:
        """Return the most recent N epoch records."""
        return self._history[-n:] if len(self._history) >= n else list(self._history)

    # ------------------------------------------------------------------
    # Trend computation
    # ------------------------------------------------------------------

    def compute_trends(self, n: int = 10) -> Dict[str, Any]:
        """
        Compute the rate of change of key metrics over the last N epochs.

        Used by the fault detector to identify gradual trends (e.g., gradient
        magnitude growing 15% per epoch for 6 epochs before explosion).

        Returns dict with:
          loss_trend         : average change in loss per epoch (negative = improving)
          gradient_trend     : average change in mean gradient magnitude per epoch
          loss_values        : list of last N loss values
          gradient_values    : list of last N global mean gradient magnitudes
          gradient_std_values: list of last N global gradient std values
        """
        recent = self.get_last_n(n)
        if len(recent) < 2:
            return {
                "loss_trend": 0.0,
                "gradient_trend": 0.0,
                "loss_values": [r["loss"] for r in recent],
                "gradient_values": [],
                "gradient_std_values": [],
                "n_epochs": len(recent),
            }

        losses = [r["loss"] for r in recent]
        grad_means = []
        grad_stds = []

        for r in recent:
            g = r.get("gradient", {}).get("global", {})
            grad_means.append(g.get("mean", 0.0))
            grad_stds.append(g.get("std", 0.0))

        # Rate of change = (last - first) / (n-1)
        n_intervals = len(recent) - 1
        loss_trend = (losses[-1] - losses[0]) / n_intervals if n_intervals > 0 else 0.0
        grad_trend = (
            (grad_means[-1] - grad_means[0]) / n_intervals
            if n_intervals > 0 and grad_means
            else 0.0
        )

        return {
            "loss_trend": loss_trend,
            "gradient_trend": grad_trend,
            "loss_values": losses,
            "gradient_values": grad_means,
            "gradient_std_values": grad_stds,
            "n_epochs": len(recent),
        }

    def get_gradient_rolling_stats(self, window: int = 10) -> Dict[str, float]:
        """
        Compute rolling mean and std of global gradient magnitude over last `window` epochs.
        Used by the adaptive threshold in fault detection.

        Returns:
            dict with rolling_mean and rolling_std
        """
        recent = self.get_last_n(window)
        grad_means = []
        for r in recent:
            g = r.get("gradient", {}).get("global", {})
            m = g.get("mean", 0.0)
            if m is not None:
                grad_means.append(m)

        if not grad_means:
            return {"rolling_mean": 0.0, "rolling_std": 0.0}

        n = len(grad_means)
        mean = sum(grad_means) / n
        variance = sum((v - mean) ** 2 for v in grad_means) / n
        import math
        std = math.sqrt(variance)

        return {"rolling_mean": mean, "rolling_std": std}

    def get_accuracy_history(self, n: int = 20) -> List[Optional[float]]:
        """Return accuracy values for last N epochs (for stagnation/vanishing detection)."""
        recent = self.get_last_n(n)
        return [r.get("accuracy") for r in recent]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self):
        """Write current in-memory history to disk."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def __repr__(self):
        return (
            f"MetricRecorder(log='{self.log_path}', "
            f"epochs_logged={len(self._history)})"
        )
