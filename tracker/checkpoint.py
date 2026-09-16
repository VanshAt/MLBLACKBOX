"""
checkpoint.py — Checkpoint save/load/list system for MLBlackBox.

A checkpoint is a complete frozen snapshot of the model at a specific epoch.
It contains everything needed to restore the model to exactly that state.

Files are named: 
checkpoints/checkpoint_epoch_NNN.pt (PyTorch state dict)
checkpoints/checkpoint_epoch_NNN.json (Metadata)

IMPORTANT: Checkpoints are NEVER overwritten. All are kept.
The backtracker needs multiple epochs to compare and find trends.
"""

import json
import os
import glob
import torch
from datetime import datetime
from typing import List, Optional


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "checkpoint_epoch_"
CHECKPOINT_EXT_PT = ".pt"
CHECKPOINT_EXT_JSON = ".json"


class CheckpointManager:
    """
    Manages saving, loading, and listing model checkpoints.
    """

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        network: torch.nn.Module,
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
        """
        metadata = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "gradient_stats": gradient_stats or {},
            "weight_stats": weight_stats or {},
            "batch_idx": batch_idx,
            "timestamp": datetime.now().isoformat(),
        }

        path_pt = self._epoch_path_pt(epoch)
        path_json = self._epoch_path_json(epoch)

        # Never overwrite — if it already exists, skip
        if os.path.exists(path_pt) and os.path.exists(path_json):
            return path_pt

        # Save PyTorch state dict
        torch.save(network.state_dict(), path_pt)

        # Save metadata
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return path_pt

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, epoch: int, network: torch.nn.Module) -> dict:
        """
        Load a checkpoint and restore all weights/biases into the live network.
        """
        path_pt = self._epoch_path_pt(epoch)
        path_json = self._epoch_path_json(epoch)
        
        if not os.path.exists(path_pt) or not os.path.exists(path_json):
            raise FileNotFoundError(
                f"No checkpoint found for epoch {epoch}. "
                f"Available: {self.list_epochs()}"
            )

        # Load weights
        network.load_state_dict(torch.load(path_pt))
        
        # Load metadata
        with open(path_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def load_record(self, epoch: int) -> dict:
        """
        Load a checkpoint record without restoring into a network.
        Used by the backtracker for comparison and analysis.
        """
        path_json = self._epoch_path_json(epoch)
        if not os.path.exists(path_json):
            raise FileNotFoundError(f"No checkpoint metadata for epoch {epoch} at '{path_json}'.")

        with open(path_json, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_epochs(self) -> List[int]:
        """
        Return a sorted list of all saved checkpoint epoch numbers.
        """
        pattern = os.path.join(
            self.checkpoint_dir,
            f"{CHECKPOINT_PREFIX}*{CHECKPOINT_EXT_JSON}"
        )
        files = glob.glob(pattern)
        epochs = []
        for f in files:
            basename = os.path.basename(f)
            try:
                ep_str = basename.replace(CHECKPOINT_PREFIX, "").replace(CHECKPOINT_EXT_JSON, "")
                epochs.append(int(ep_str))
            except ValueError:
                continue
        return sorted(epochs)

    def list_all(self) -> List[dict]:
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
        epochs = self.list_epochs()
        return epochs[-1] if epochs else None

    def get_last_clean_epoch(
        self, fault_epoch: int, fault_log: Optional[List[int]] = None
    ) -> Optional[int]:
        epochs = self.list_epochs()
        candidates = [e for e in epochs if e < fault_epoch]
        if not candidates:
            return None

        if fault_log:
            for e in reversed(candidates):
                if e not in fault_log:
                    return e
            return candidates[0]

        return candidates[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _epoch_path_pt(self, epoch: int) -> str:
        filename = f"{CHECKPOINT_PREFIX}{epoch:05d}{CHECKPOINT_EXT_PT}"
        return os.path.join(self.checkpoint_dir, filename)

    def _epoch_path_json(self, epoch: int) -> str:
        filename = f"{CHECKPOINT_PREFIX}{epoch:05d}{CHECKPOINT_EXT_JSON}"
        return os.path.join(self.checkpoint_dir, filename)

    def __repr__(self):
        return (
            f"CheckpointManager(dir='{self.checkpoint_dir}', "
            f"saved={len(self.list_epochs())} checkpoints)"
        )
