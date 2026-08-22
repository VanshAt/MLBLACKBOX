"""
checkpoint.py — Checkpoint save/load/list system for MLBlackBox.

A checkpoint is a complete frozen snapshot of the model at a specific epoch.
It contains everything needed to restore the model to exactly that state.

Files are named: checkpoints/checkpoint_epoch_NNN.json
Zero-padded epoch numbers ensure correct alphabetical sort order.

IMPORTANT: Checkpoints are NEVER overwritten. All are kept.
The backtracker needs multiple epochs to compare and find trends.

Checkpoint JSON schema:
{
    "epoch": 14,
    "loss": 0.342,
    "accuracy": 0.91,
    "learning_rate": 0.01,
    "params": [[{neuron_dict}, ...], ...],    // [layer_idx][neuron_idx]
    "gradient_stats": {global: {...}, per_layer: [...]},
    "weight_stats": {max: ..., min: ..., mean: ...},
    "batch_idx": 47,
    "timestamp": "2026-08-22T08:00:00"
}
"""

import json
import os
import glob
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.network import Network


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "checkpoint_epoch_"
CHECKPOINT_EXT = ".json"


class CheckpointManager:
    """
    Manages saving, loading, and listing model checkpoints.

    Each checkpoint is stored as a separate JSON file and is never overwritten.
    Epoch numbers are zero-padded to 5 digits for correct sort order.
    """

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        network: "Network",
        epoch: int,
        loss: float,
        accuracy: Optional[float] = None,
        gradient_stats: Optional[dict] = None,
        weight_stats: Optional[dict] = None,
        batch_idx: int = 0,
        learning_rate: float = 0.01,
    ) -> str:
        """
        Save a complete model snapshot to disk.

        Args:
            network       : the Network instance
            epoch         : current epoch number
            loss          : training loss at this epoch
            accuracy      : training accuracy (None for regression)
            gradient_stats: output of gradient_tracker.compute_gradient_stats()
            weight_stats  : output of gradient_tracker.compute_weight_stats()
            batch_idx     : index of the last batch processed this epoch
            learning_rate : current learning rate

        Returns:
            str: path to the saved checkpoint file
        """
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "params": network.get_all_params(),
            "gradient_stats": gradient_stats or {},
            "weight_stats": weight_stats or {},
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "network_sizes": network.sizes,
        }

        path = self._epoch_path(epoch)

        # Never overwrite — if it already exists, skip
        if os.path.exists(path):
            return path

        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, epoch: int, network: "Network") -> dict:
        """
        Load a checkpoint and restore all weights/biases into the live network.

        Args:
            epoch  : epoch number to restore
            network: the Network instance to restore into

        Returns:
            dict: the full checkpoint record

        Raises:
            FileNotFoundError: if checkpoint for this epoch does not exist
        """
        path = self._epoch_path(epoch)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No checkpoint found for epoch {epoch} at '{path}'. "
                f"Available: {self.list_epochs()}"
            )

        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        network.set_all_params(checkpoint["params"])
        return checkpoint

    def load_record(self, epoch: int) -> dict:
        """
        Load a checkpoint record without restoring into a network.
        Used by the backtracker for comparison and analysis.
        """
        path = self._epoch_path(epoch)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint for epoch {epoch} at '{path}'.")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_epochs(self) -> List[int]:
        """
        Return a sorted list of all saved checkpoint epoch numbers.

        Returns:
            List[int]: epoch numbers in ascending order
        """
        pattern = os.path.join(
            self.checkpoint_dir,
            f"{CHECKPOINT_PREFIX}*{CHECKPOINT_EXT}"
        )
        files = glob.glob(pattern)
        epochs = []
        for f in files:
            basename = os.path.basename(f)
            try:
                ep_str = basename.replace(CHECKPOINT_PREFIX, "").replace(CHECKPOINT_EXT, "")
                epochs.append(int(ep_str))
            except ValueError:
                continue
        return sorted(epochs)

    def list_all(self) -> List[dict]:
        """
        Return summary dicts for all saved checkpoints (epoch, loss, accuracy, timestamp).
        Sorted by epoch ascending.
        """
        summaries = []
        for epoch in self.list_epochs():
            try:
                rec = self.load_record(epoch)
                summaries.append({
                    "epoch": rec["epoch"],
                    "loss": rec["loss"],
                    "accuracy": rec.get("accuracy"),
                    "timestamp": rec.get("timestamp"),
                })
            except Exception:
                continue
        return summaries

    def get_last_epoch(self) -> Optional[int]:
        """Return the most recently saved epoch number, or None if no checkpoints exist."""
        epochs = self.list_epochs()
        return epochs[-1] if epochs else None

    def get_last_clean_epoch(
        self, fault_epoch: int, fault_log: Optional[List[int]] = None
    ) -> Optional[int]:
        """
        Find the last epoch before fault_epoch that was clean (no fault detected).

        Args:
            fault_epoch: the epoch where a fault was first detected
            fault_log  : optional list of fault epoch numbers (from FaultDetector).
                         If provided, walks back to find first epoch before all faults.

        Returns:
            int or None: epoch number of the last clean checkpoint
        """
        epochs = self.list_epochs()
        candidates = [e for e in epochs if e < fault_epoch]
        if not candidates:
            return None

        if fault_log:
            # Walk back until we find an epoch not in the fault log
            for e in reversed(candidates):
                if e not in fault_log:
                    return e
            return candidates[0]  # fallback: earliest available

        # Without fault log, just return the epoch right before fault
        return candidates[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _epoch_path(self, epoch: int) -> str:
        """Construct the file path for a given epoch number."""
        filename = f"{CHECKPOINT_PREFIX}{epoch:05d}{CHECKPOINT_EXT}"
        return os.path.join(self.checkpoint_dir, filename)

    def __repr__(self):
        return (
            f"CheckpointManager(dir='{self.checkpoint_dir}', "
            f"saved={len(self.list_epochs())} checkpoints)"
        )
