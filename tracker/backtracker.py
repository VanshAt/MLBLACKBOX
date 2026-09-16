"""
backtracker.py — Rewind, root cause analysis, and resume for MLBlackBox.

When a fault is detected, the backtracker:
  1. REWIND  — loads the last clean checkpoint, restoring all weights/biases
  2. ANALYZE — diffs metrics between clean epoch and fault epoch to find root cause
  3. EXPLAIN — maps the fault pattern to a cause + fix from the cause library
  4. RESUME  — returns the restored network ready to continue training

The Cause Library maps fault patterns to probable causes and concrete fixes.
This is the component that doesn't exist in any other framework:
plain-English root cause explanation with actionable suggestions.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import torch

from tracker.checkpoint import CheckpointManager
from tracker.metric_recorder import MetricRecorder
from tracker.fault_detector import FaultResult, FaultType


# ------------------------------------------------------------------
# Cause Library
# ------------------------------------------------------------------

CAUSE_LIBRARY = {
    # ---- Gradient Explosion patterns --------------------------------
    (FaultType.GRADIENT_EXPLOSION, "unscaled_features"): {
        "cause": (
            "Input features with large magnitude amplify gradients through the "
            "network. When a feature value is 1000× larger than the weight scale, "
            "the gradient for that weight becomes proportionally large."
        ),
        "fixes": [
            "Normalize input features to [0, 1] using min-max scaling",
            "Standardize features to mean=0, std=1 using z-score normalization",
            "Reduce learning rate by 10× (e.g., 0.01 → 0.001)",
            "Add gradient clipping: clip all gradients to max abs value of 1.0",
            "Resume training from the last clean checkpoint",
        ],
    },
    (FaultType.GRADIENT_EXPLOSION, "high_learning_rate"): {
        "cause": (
            "Learning rate is too high for the current region of the loss surface. "
            "The optimizer is taking steps so large it overshoots the minimum and "
            "the loss surface curves sharply, amplifying the next gradient."
        ),
        "fixes": [
            "Reduce learning rate by 10× (exponential decay)",
            "Add gradient clipping: clip all gradients to max abs value of 1.0",
            "Use learning rate warmup (start at 10% of target LR for first 10 epochs)",
            "Resume training from the last clean checkpoint",
        ],
    },
    (FaultType.GRADIENT_EXPLOSION, "deep_layer"): {
        "cause": (
            "Gradients are exploding in a specific layer, suggesting weight "
            "initialization or architecture is causing instability in that layer."
        ),
        "fixes": [
            "Reduce weight initialization range (use [-0.01, 0.01] instead of [-0.1, 0.1])",
            "Add batch normalization after the problem layer",
            "Add gradient clipping: clip all gradients to max abs value of 1.0",
            "Resume training from the last clean checkpoint",
        ],
    },
    # ---- Gradient Vanishing patterns --------------------------------
    (FaultType.GRADIENT_VANISHING, "sigmoid_deep"): {
        "cause": (
            "Sigmoid activation derivative is at most 0.25. In a network with N "
            "hidden layers using sigmoid, the gradient shrinks by at least 4^N "
            "as it propagates backward, becoming effectively zero before reaching "
            "early layers."
        ),
        "fixes": [
            "Replace sigmoid activations in hidden layers with ReLU",
            "ReLU derivative is 0 or 1 — it does not shrink gradients",
            "Keep sigmoid only on the output layer for binary classification",
            "Re-initialize the network and retrain from epoch 1",
        ],
    },
    (FaultType.GRADIENT_VANISHING, "deep_relu"): {
        "cause": (
            "Very deep ReLU networks can suffer from 'dying ReLU' where many neurons "
            "output 0 permanently because their pre-activation values are always negative. "
            "The gradient through dead neurons is exactly 0."
        ),
        "fixes": [
            "Use Leaky ReLU (outputs 0.01×z for z<0) instead of standard ReLU",
            "Reduce learning rate — large updates can push neurons into permanent death",
            "Check weight initialization — very negative initial weights cause mass death",
            "Re-initialize with a larger positive bias to reduce dead neurons",
        ],
    },
    (FaultType.GRADIENT_VANISHING, "saddle_point"): {
        "cause": (
            "Network appears to be near a saddle point: gradients exist but are "
            "consistently tiny across all layers, and loss is not improving. "
            "The optimizer is exploring a flat region of the loss surface."
        ),
        "fixes": [
            "Add momentum to the optimizer (momentum=0.9 typical)",
            "Slightly increase learning rate to escape the flat region",
            "Try a different random seed for re-initialization",
            "Add small random noise to weights to perturb out of saddle point",
        ],
    },
    # ---- NaN Propagation patterns -----------------------------------
    (FaultType.NAN_INF_PROPAGATION, "loss_nan"): {
        "cause": (
            "NaN in loss typically means the network's output went out of the "
            "valid range for the loss function. Common causes: sigmoid output "
            "of exactly 0 or 1 fed to binary cross-entropy (log(0) = -inf), "
            "or catastrophically large weights producing inf."
        ),
        "fixes": [
            "Binary Cross-Entropy: ensure epsilon clamping in log (already built in)",
            "Check if weights have grown to Inf — likely preceded by gradient explosion",
            "Reduce learning rate significantly",
            "Add gradient clipping before the next run",
            "Restore from last clean checkpoint",
        ],
    },
    (FaultType.NAN_INF_PROPAGATION, "layer_nan"): {
        "cause": (
            "NaN appeared first in a specific layer, suggesting that layer's "
            "weights or activations overflowed. Often caused by gradient explosion "
            "in previous epochs that was not detected in time."
        ),
        "fixes": [
            "Add gradient clipping immediately",
            "Inspect input data for extreme values (unscaled features)",
            "Add weight magnitude monitoring — weights > 1000 are a warning sign",
            "Restore from last clean checkpoint",
        ],
    },
    # ---- Loss Spike patterns ----------------------------------------
    (FaultType.LOSS_SPIKE, "outlier_batch"): {
        "cause": (
            "A single data batch with extreme feature values caused the network to "
            "produce a very large prediction error, generating a correspondingly "
            "large gradient that pushed weights into a bad region."
        ),
        "fixes": [
            "Inspect and remove or cap outlier values in the training data",
            "Normalize/standardize all features before training",
            "Add gradient clipping to limit the damage from future outlier batches",
            "Reduce learning rate to reduce sensitivity to individual batches",
            "Restore from last clean checkpoint",
        ],
    },
    # ---- Stagnation patterns ----------------------------------------
    (FaultType.TRAINING_STAGNATION, "saddle_point"): {
        "cause": (
            "Loss is not improving despite gradients being present. The network "
            "is likely in a flat region or saddle point of the loss surface where "
            "gradient direction is near-orthogonal to the descent direction."
        ),
        "fixes": [
            "Add momentum (0.9) to the optimizer to build up velocity",
            "Try increasing learning rate slightly to escape the flat region",
            "Try a different random seed",
            "Verify training data labels are not corrupted",
        ],
    },
    (FaultType.TRAINING_STAGNATION, "architecture"): {
        "cause": (
            "Network may be too small to represent the function it is trying to learn, "
            "or the network may be too large with redundant neurons interfering with "
            "each other."
        ),
        "fixes": [
            "Try a wider network (more neurons per layer)",
            "Try adding one more hidden layer",
            "Check that output activation matches the task "
            "(Linear for regression, Sigmoid for binary classification)",
        ],
    },
}


# ------------------------------------------------------------------
# BacktrackResult data object
# ------------------------------------------------------------------

class BacktrackResult:
    """Output of a full backtrack operation."""

    def __init__(
        self,
        fault: FaultResult,
        clean_epoch: int,
        clean_record: dict,
        fault_record: dict,
        root_cause: dict,
        network_restored: bool,
    ):
        self.fault = fault
        self.clean_epoch = clean_epoch
        self.clean_record = clean_record
        self.fault_record = fault_record
        self.root_cause = root_cause
        self.network_restored = network_restored


# ------------------------------------------------------------------
# Backtracker
# ------------------------------------------------------------------

class Backtracker:
    """
    Rewinds the network to a clean state and produces a root cause analysis
    report when a fault is detected.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        recorder: MetricRecorder,
    ):
        self.checkpoints = checkpoint_manager
        self.recorder = recorder

    def backtrack(
        self,
        fault: FaultResult,
        network: torch.nn.Module,
    ) -> BacktrackResult:
        """
        Full backtrack pipeline: find clean epoch → restore → analyze → explain.

        Args:
            fault  : the FaultResult from FaultDetector
            network: the live torch.nn.Module instance (will be restored in place)

        Returns:
            BacktrackResult with root cause and the restored network
        """
        # Step 1: Find the last clean epoch
        fault_log = self.checkpoints.list_epochs()
        clean_epoch = self.checkpoints.get_last_clean_epoch(
            fault.first_epoch, fault_log=None
        )

        if clean_epoch is None:
            # No clean checkpoint available — can't restore
            return BacktrackResult(
                fault=fault,
                clean_epoch=-1,
                clean_record={},
                fault_record=self.recorder.get_epoch(fault.detected_epoch) or {},
                root_cause={"cause": "No clean checkpoint available.", "fixes": []},
                network_restored=False,
            )

        # Step 2: Restore network weights from clean checkpoint
        clean_record_from_ckpt = self.checkpoints.load(clean_epoch, network)
        clean_record_from_log = self.recorder.get_epoch(clean_epoch) or clean_record_from_ckpt
        fault_record = self.recorder.get_epoch(fault.detected_epoch) or {}

        # Step 3: Compare clean vs fault metrics
        root_cause = self._analyze(fault, clean_record_from_log, fault_record)

        return BacktrackResult(
            fault=fault,
            clean_epoch=clean_epoch,
            clean_record=clean_record_from_log,
            fault_record=fault_record,
            root_cause=root_cause,
            network_restored=True,
        )

    def _analyze(
        self,
        fault: FaultResult,
        clean_record: dict,
        fault_record: dict,
    ) -> dict:
        """
        Diff metrics between the clean epoch and fault epoch to select
        the most likely cause from the Cause Library.
        """
        fault_type = fault.fault_type

        # Heuristics to pick the right sub-pattern
        sub_pattern = self._classify_sub_pattern(fault, clean_record, fault_record)

        key = (fault_type, sub_pattern)
        if key in CAUSE_LIBRARY:
            entry = CAUSE_LIBRARY[key]
        else:
            # Fallback: use any entry matching the fault type
            entry = next(
                (v for (ft, _), v in CAUSE_LIBRARY.items() if ft == fault_type),
                {"cause": "Unknown pattern.", "fixes": ["Inspect audit_log.json for trends."]},
            )

        # Build the diff evidence
        clean_loss = clean_record.get("loss", "N/A")
        fault_loss = fault_record.get("loss", "N/A")
        clean_grad = (
            clean_record.get("gradient", {}).get("global", {}).get("mean", "N/A")
        )
        fault_grad = (
            fault_record.get("gradient", {}).get("global", {}).get("mean", "N/A")
        )

        return {
            "fault_type": fault_type,
            "sub_pattern": sub_pattern,
            "cause": entry["cause"],
            "fixes": entry["fixes"],
            "metric_diff": {
                "clean_epoch": clean_record.get("epoch", "?"),
                "fault_epoch": fault_record.get("epoch", "?"),
                "loss": {"clean": clean_loss, "fault": fault_loss},
                "gradient_mean": {"clean": clean_grad, "fault": fault_grad},
                "suspected_layer": fault.suspected_layer,
                "batch_idx": fault.batch_idx,
            },
        }

    def _classify_sub_pattern(
        self,
        fault: FaultResult,
        clean_record: dict,
        fault_record: dict,
    ) -> str:
        """
        Heuristic rules to select the most specific sub-pattern for root cause lookup.
        """
        ft = fault.fault_type
        evidence = fault.evidence

        if ft == FaultType.GRADIENT_EXPLOSION:
            # Check if a specific layer is implicated
            if fault.suspected_layer is not None:
                return "deep_layer"
            # Check if gradient grew slowly (learning rate) vs suddenly (outlier)
            ratio = evidence.get("current_gradient_mean", 0) / max(
                evidence.get("rolling_mean", 1.0), 1e-9
            )
            return "high_learning_rate" if ratio < 20 else "unscaled_features"

        elif ft == FaultType.GRADIENT_VANISHING:
            # Try to infer if sigmoid is responsible by checking if gradient mean
            # was already very low in earlier epochs (slow decay = sigmoid)
            history = self.recorder.read()
            if len(history) >= 5:
                early_grad = history[0].get("gradient", {}).get("global", {}).get("mean", 1.0)
                recent_grad = evidence.get("current_gradient_mean", 0.0)
                if recent_grad is not None and early_grad is not None and early_grad > 0:
                    decay_ratio = recent_grad / early_grad
                    if decay_ratio < 0.01:
                        return "sigmoid_deep"
            return "saddle_point"

        elif ft == FaultType.NAN_INF_PROPAGATION:
            if fault.suspected_layer is not None:
                return "layer_nan"
            return "loss_nan"

        elif ft == FaultType.LOSS_SPIKE:
            return "outlier_batch"

        elif ft == FaultType.TRAINING_STAGNATION:
            # Check gradient magnitude — if it's normal but loss is stuck, saddle point
            grad_mean = fault_record.get("gradient", {}).get("global", {}).get("mean", 0.0)
            if grad_mean is not None and grad_mean > 1e-5:
                return "saddle_point"
            return "architecture"

        return "unknown"

    def resume(self, network: torch.nn.Module) -> torch.nn.Module:
        """
        Return the network after restoration — it is ready to resume training.
        Caller should call trainer.train() again with reduced epochs or adjusted LR.
        """
        return network
