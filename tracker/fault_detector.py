"""
fault_detector.py — Five-fault detection engine for MLBlackBox.

Runs after every epoch and classifies whether something has gone wrong.
Uses adaptive thresholds based on rolling statistics — fixed thresholds
don't work across different architectures and learning rates.

The five fault types:
  1. GRADIENT_EXPLOSION  — gradients spike above adaptive threshold
  2. GRADIENT_VANISHING  — gradients collapse, network stops learning
  3. NAN_INF_PROPAGATION — any NaN or Inf in network values
  4. LOSS_SPIKE          — loss suddenly multiplies (bad batch or high LR)
  5. TRAINING_STAGNATION — loss stuck at high value for many epochs

Each detected fault includes:
  - fault_type      : string identifier
  - severity        : "low" | "medium" | "critical"
  - first_epoch     : estimated onset epoch (not just where it was detected)
  - detected_epoch  : epoch when this detector fired
  - evidence        : dict of supporting metric values
  - suspected_layer : int or None (for gradient faults)
  - batch_idx       : batch index at fault epoch (for spike faults)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from tracker.metric_recorder import MetricRecorder


# ------------------------------------------------------------------
# Fault type constants
# ------------------------------------------------------------------

class FaultType:
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    GRADIENT_VANISHING = "GRADIENT_VANISHING"
    NAN_INF_PROPAGATION = "NAN_INF_PROPAGATION"
    LOSS_SPIKE = "LOSS_SPIKE"
    TRAINING_STAGNATION = "TRAINING_STAGNATION"


@dataclass
class FaultResult:
    """Describes a detected training fault."""
    fault_type: str
    severity: str                    # "low", "medium", "critical"
    detected_epoch: int
    first_epoch: int                 # estimated onset
    evidence: Dict[str, Any] = field(default_factory=dict)
    suspected_layer: Optional[int] = None
    batch_idx: Optional[int] = None
    description: str = ""


class FaultDetector:
    """
    Stateful fault detector that maintains history and runs all five
    fault checks after each epoch.

    Instantiate once and call check() after each epoch record is recorded.
    """

    def __init__(
        self,
        recorder: MetricRecorder,
        explosion_window: int = 10,
        explosion_sigma: float = 3.0,
        vanishing_threshold: float = 1e-6,
        vanishing_no_improvement_epochs: int = 20,
        spike_multiplier: float = 5.0,
        stagnation_min_delta: float = 0.001,
        stagnation_epochs: int = 15,
        stagnation_min_loss: float = 0.05,
    ):
        """
        Args:
            recorder                    : MetricRecorder instance (shared with trainer)
            explosion_window            : rolling window for adaptive explosion threshold
            explosion_sigma             : number of std devs above rolling mean = explosion
            vanishing_threshold         : mean gradient below this = vanishing
            vanishing_no_improvement_epochs: how many epochs of no accuracy gain confirms vanishing
            spike_multiplier            : loss > previous * this = spike fault
            stagnation_min_delta        : loss must change by at least this to count as progress
            stagnation_epochs           : how many consecutive static epochs = stagnation
            stagnation_min_loss         : only flag stagnation if loss is above this (avoids
                                          flagging converged networks as stagnated)
        """
        self.recorder = recorder
        self.explosion_window = explosion_window
        self.explosion_sigma = explosion_sigma
        self.vanishing_threshold = vanishing_threshold
        self.vanishing_no_improvement_epochs = vanishing_no_improvement_epochs
        self.spike_multiplier = spike_multiplier
        self.stagnation_min_delta = stagnation_min_delta
        self.stagnation_epochs = stagnation_epochs
        self.stagnation_min_loss = stagnation_min_loss

        # Internal state
        self._fault_epochs: List[int] = []   # all epochs where a fault was detected
        self._stagnation_counter: int = 0
        self._prev_loss: Optional[float] = None

    # ------------------------------------------------------------------
    # Main check — call after each epoch
    # ------------------------------------------------------------------

    def check(self, epoch_record: dict) -> Optional[FaultResult]:
        """
        Run all five fault checks against the current epoch record.

        Args:
            epoch_record: the dict appended to the audit log for this epoch

        Returns:
            FaultResult if a fault was detected, None otherwise.
            Priority order: NaN > Explosion > Vanishing > Spike > Stagnation
        """
        epoch = epoch_record["epoch"]
        history = self.recorder.read()

        # 1. NaN/Inf — highest priority, check immediately
        nan_result = self._check_nan(epoch_record)
        if nan_result:
            self._fault_epochs.append(epoch)
            return nan_result

        # 2. Gradient Explosion
        explosion_result = self._check_explosion(epoch_record, history)
        if explosion_result:
            self._fault_epochs.append(epoch)
            return explosion_result

        # 3. Gradient Vanishing
        vanishing_result = self._check_vanishing(epoch_record, history)
        if vanishing_result:
            self._fault_epochs.append(epoch)
            return vanishing_result

        # 4. Loss Spike
        spike_result = self._check_spike(epoch_record)
        if spike_result:
            self._fault_epochs.append(epoch)
            self._prev_loss = epoch_record.get("loss")
            return spike_result

        # 5. Stagnation
        stagnation_result = self._check_stagnation(epoch_record)
        if stagnation_result:
            self._fault_epochs.append(epoch)
            self._prev_loss = epoch_record.get("loss")
            return stagnation_result

        # Clean epoch — update previous loss tracker
        self._prev_loss = epoch_record.get("loss")
        return None

    def get_fault_epochs(self) -> List[int]:
        """Return list of all epochs where a fault was detected."""
        return list(self._fault_epochs)

    # ------------------------------------------------------------------
    # Fault 1 — NaN / Inf Propagation
    # ------------------------------------------------------------------

    def _check_nan(self, record: dict) -> Optional[FaultResult]:
        """
        Check for NaN/Inf in loss, gradients, or network values.
        Layer-level NaN location comes from backprop.py's nan tracker.
        """
        epoch = record["epoch"]
        loss = record.get("loss", 0.0)
        nan_layer = record.get("nan_layer", -1)

        # NaN in loss
        if math.isnan(loss) or math.isinf(loss):
            return FaultResult(
                fault_type=FaultType.NAN_INF_PROPAGATION,
                severity="critical",
                detected_epoch=epoch,
                first_epoch=epoch,
                evidence={"loss": str(loss), "nan_layer": nan_layer},
                suspected_layer=nan_layer if nan_layer >= 0 else None,
                description="NaN or Inf detected in training loss.",
            )

        # NaN in gradient stats
        grad_global = record.get("gradient", {}).get("global", {})
        if grad_global.get("has_nan", False):
            return FaultResult(
                fault_type=FaultType.NAN_INF_PROPAGATION,
                severity="critical",
                detected_epoch=epoch,
                first_epoch=epoch,
                evidence={"nan_in_gradients": True, "nan_layer": nan_layer},
                suspected_layer=nan_layer if nan_layer >= 0 else None,
                description="NaN or Inf detected in gradient values.",
            )

        # NaN flagged during backprop
        if nan_layer >= 0:
            return FaultResult(
                fault_type=FaultType.NAN_INF_PROPAGATION,
                severity="critical",
                detected_epoch=epoch,
                first_epoch=epoch,
                evidence={"nan_layer": nan_layer},
                suspected_layer=nan_layer,
                description=f"NaN detected during backward pass in layer {nan_layer}.",
            )

        return None

    # ------------------------------------------------------------------
    # Fault 2 — Gradient Explosion
    # ------------------------------------------------------------------

    def _check_explosion(
        self, record: dict, history: List[dict]
    ) -> Optional[FaultResult]:
        """
        Adaptive threshold: flag if current gradient mean is more than
        explosion_sigma standard deviations above the rolling mean.
        """
        epoch = record["epoch"]
        grad_global = record.get("gradient", {}).get("global", {})
        current_mean = grad_global.get("mean", 0.0)

        if current_mean is None or math.isnan(current_mean):
            return None

        # Need at least explosion_window epochs of history for rolling stats
        rolling = self.recorder.get_gradient_rolling_stats(self.explosion_window)
        r_mean = rolling["rolling_mean"]
        r_std = rolling["rolling_std"]

        threshold = r_mean + self.explosion_sigma * r_std

        # Don't fire if threshold is essentially zero (no history yet)
        if threshold < 1e-9 or len(history) < 3:
            return None

        if current_mean > threshold:
            # Estimate first appearance by scanning backward
            first_epoch = self._estimate_fault_onset(
                history, metric_key=("gradient", "global", "mean"),
                threshold=threshold,
            )

            # Find which layer exploded most
            per_layer = record.get("gradient", {}).get("per_layer", [])
            worst_layer = None
            worst_val = -1.0
            for layer_stat in per_layer:
                v = layer_stat.get("max", 0.0)
                if v is not None and v > worst_val:
                    worst_val = v
                    worst_layer = layer_stat.get("layer")

            return FaultResult(
                fault_type=FaultType.GRADIENT_EXPLOSION,
                severity="critical" if current_mean > threshold * 10 else "medium",
                detected_epoch=epoch,
                first_epoch=first_epoch,
                evidence={
                    "current_gradient_mean": current_mean,
                    "rolling_mean": r_mean,
                    "rolling_std": r_std,
                    "adaptive_threshold": threshold,
                    "worst_layer_max": worst_val,
                },
                suspected_layer=worst_layer,
                batch_idx=record.get("batch_idx"),
                description=(
                    f"Gradient mean ({current_mean:.4f}) exceeded adaptive threshold "
                    f"({threshold:.4f} = {r_mean:.4f} + {self.explosion_sigma}σ)."
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Fault 3 — Gradient Vanishing
    # ------------------------------------------------------------------

    def _check_vanishing(
        self, record: dict, history: List[dict]
    ) -> Optional[FaultResult]:
        """
        Two-signal detection:
          Signal 1: mean gradient below vanishing_threshold
          Signal 2: no accuracy improvement in vanishing_no_improvement_epochs epochs
        Both signals must be present (prevents false positives on converged nets).
        """
        epoch = record["epoch"]
        grad_global = record.get("gradient", {}).get("global", {})
        current_mean = grad_global.get("mean", 0.0)

        if current_mean is None or math.isnan(current_mean):
            return None

        # Signal 1: tiny gradient
        if current_mean >= self.vanishing_threshold:
            return None

        # Signal 2: no accuracy improvement (for classification)
        acc_history = self.recorder.get_accuracy_history(
            self.vanishing_no_improvement_epochs
        )
        acc_values = [a for a in acc_history if a is not None]

        # If accuracy exists and is improving, this might be a converged net, not vanishing
        if acc_values:
            acc_range = max(acc_values) - min(acc_values)
            # If accuracy changed by more than 1%, network is still learning
            if acc_range > 0.01:
                return None

        first_epoch = self._estimate_fault_onset(
            history,
            metric_key=("gradient", "global", "mean"),
            threshold=self.vanishing_threshold,
            direction="below",
        )

        return FaultResult(
            fault_type=FaultType.GRADIENT_VANISHING,
            severity="medium",
            detected_epoch=epoch,
            first_epoch=first_epoch,
            evidence={
                "current_gradient_mean": current_mean,
                "vanishing_threshold": self.vanishing_threshold,
                "accuracy_range_last_n": acc_range if acc_values else None,
                "no_improvement_epochs": self.vanishing_no_improvement_epochs,
            },
            description=(
                f"Gradient mean ({current_mean:.2e}) below threshold "
                f"({self.vanishing_threshold:.2e}). Network may have stopped learning."
            ),
        )

    # ------------------------------------------------------------------
    # Fault 4 — Loss Spike
    # ------------------------------------------------------------------

    def _check_spike(self, record: dict) -> Optional[FaultResult]:
        """
        Loss increased by more than spike_multiplier × previous loss.
        Stores the batch index that caused the spike.
        """
        epoch = record["epoch"]
        loss = record.get("loss", 0.0)

        if self._prev_loss is None or math.isnan(loss) or math.isnan(self._prev_loss):
            return None
        if self._prev_loss < 1e-10:
            return None  # avoid division by near-zero

        ratio = loss / self._prev_loss
        if ratio > self.spike_multiplier:
            severity = "critical" if ratio > 50 else "medium"
            return FaultResult(
                fault_type=FaultType.LOSS_SPIKE,
                severity=severity,
                detected_epoch=epoch,
                first_epoch=epoch,
                evidence={
                    "current_loss": loss,
                    "previous_loss": self._prev_loss,
                    "ratio": ratio,
                    "spike_multiplier_threshold": self.spike_multiplier,
                },
                batch_idx=record.get("batch_idx"),
                description=(
                    f"Loss spiked from {self._prev_loss:.4f} to {loss:.4f} "
                    f"({ratio:.1f}× increase). Check batch {record.get('batch_idx')} for outliers."
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Fault 5 — Training Stagnation
    # ------------------------------------------------------------------

    def _check_stagnation(self, record: dict) -> Optional[FaultResult]:
        """
        Loss has not decreased by stagnation_min_delta for stagnation_epochs
        consecutive epochs, AND the loss is above stagnation_min_loss
        (to distinguish stagnation from convergence).
        """
        epoch = record["epoch"]
        loss = record.get("loss", 0.0)

        if loss is None or math.isnan(loss):
            return None

        # Only flag if loss is meaningfully high (not converged)
        if loss < self.stagnation_min_loss:
            self._stagnation_counter = 0
            return None

        # Update stagnation counter
        if self._prev_loss is not None:
            delta = self._prev_loss - loss  # positive = improving
            if delta < self.stagnation_min_delta:
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
        
        if self._stagnation_counter >= self.stagnation_epochs:
            first_epoch = max(1, epoch - self._stagnation_counter)
            return FaultResult(
                fault_type=FaultType.TRAINING_STAGNATION,
                severity="low",
                detected_epoch=epoch,
                first_epoch=first_epoch,
                evidence={
                    "current_loss": loss,
                    "stagnation_counter": self._stagnation_counter,
                    "min_delta": self.stagnation_min_delta,
                    "stagnation_threshold_epochs": self.stagnation_epochs,
                },
                description=(
                    f"Loss ({loss:.4f}) has not improved by >{self.stagnation_min_delta} "
                    f"for {self._stagnation_counter} consecutive epochs."
                ),
            )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_fault_onset(
        self,
        history: List[dict],
        metric_key: tuple,
        threshold: float,
        direction: str = "above",
    ) -> int:
        """
        Walk backward through history to find the first epoch where
        the metric crossed the threshold. Returns the epoch number.

        Args:
            history   : full audit log
            metric_key: nested key tuple, e.g. ("gradient", "global", "mean")
            threshold : the crossing value
            direction : "above" (metric exceeded threshold) or "below"
        """
        def get_nested(record, keys):
            val = record
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                else:
                    return None
            return val

        # Walk backward from end to find where it first crossed
        first_crossing = history[-1]["epoch"] if history else 1
        for rec in reversed(history):
            val = get_nested(rec, metric_key)
            if val is None or math.isnan(val):
                continue
            crossed = (val > threshold) if direction == "above" else (val < threshold)
            if not crossed:
                # Found the last epoch where metric was still clean
                return rec["epoch"] + 1
            else:
                first_crossing = rec["epoch"]

        return first_crossing
