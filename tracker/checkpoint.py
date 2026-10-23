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
from typing import List, Optional, Any, Dict


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "checkpoint_epoch_"
CHECKPOINT_EXT_PT = ".pt"
CHECKPOINT_EXT_JSON = ".json"


class CheckpointManager:
    """
    Manages saving, loading, and listing model checkpoints.
    """

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR, run_id: Optional[str] = None):
        if run_id:
            self.run_id = str(run_id)
            self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_id)
        else:
            self.run_id = None
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        global_step: int = 0,
        input_shape: Optional[List[int]] = None,
        output_shape: Optional[List[int]] = None,
    ) -> str:
        """
        Save a complete training-state snapshot to disk.
        """
        import random
        try:
            import numpy as np
            np_state = np.random.get_state()
        except ImportError:
            np_state = None

        param_names = []
        param_shapes = {}
        for name, param in network.named_parameters():
            param_names.append(name)
            param_shapes[name] = list(param.shape)

        metadata = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "learning_rate": learning_rate,
            "gradient_stats": gradient_stats or {},
            "weight_stats": weight_stats or {},
            "batch_idx": batch_idx,
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "creation_timestamp": datetime.now().isoformat(),
            "model_class_name": network.__class__.__name__,
            "optimizer_class_name": optimizer.__class__.__name__ if optimizer is not None else None,
            "param_names": param_names,
            "param_shapes": param_shapes,
            "input_shape": input_shape,
            "output_shape": output_shape,
        }

        path_pt = self._epoch_path_pt(epoch)
        path_json = self._epoch_path_json(epoch)

        # Never overwrite — if it already exists, skip
        if os.path.exists(path_pt) and os.path.exists(path_json):
            return path_pt

        # Save full training state
        payload = {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np_state,
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "metadata": metadata,
        }
        torch.save(payload, path_pt)

        # Save metadata JSON
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return path_pt

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        epoch: int,
        network: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> dict:
        """
        Load a checkpoint and restore weights/state into the live network,
        optimizer, scheduler, and RNG. Strictly validates architecture matching.
        """
        import random
        try:
            import numpy as np
        except ImportError:
            np = None

        path_pt = self._epoch_path_pt(epoch)
        path_json = self._epoch_path_json(epoch)
        
        if not os.path.exists(path_pt) or not os.path.exists(path_json):
            raise FileNotFoundError(
                f"No checkpoint found for epoch {epoch}. "
                f"Available: {self.list_epochs()}"
            )

        # Load metadata
        with open(path_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        device = "cpu"
        if torch.cuda.is_available() and getattr(network, "parameters", None):
            try:
                first_param = next(network.parameters(), None)
                if first_param is not None and first_param.is_cuda:
                    device = "cuda"
            except Exception:
                device = "cpu"

        checkpoint_data = torch.load(path_pt, map_location=device, weights_only=False)

        # Handle backward compatibility: older checkpoints saved model.state_dict() directly
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            model_sd = checkpoint_data["model_state_dict"]
            opt_sd = checkpoint_data.get("optimizer_state_dict")
            sched_sd = checkpoint_data.get("scheduler_state_dict")
            rng_states = checkpoint_data.get("rng_states")
        else:
            model_sd = checkpoint_data
            opt_sd = None
            sched_sd = None
            rng_states = None

        # --- Architecture validation ---
        # 1. Compare parameter keys and tensor shapes against current model
        curr_sd = network.state_dict()
        curr_keys = set(curr_sd.keys())
        saved_keys = set(model_sd.keys())

        if curr_keys != saved_keys:
            raise ValueError("Checkpoint architecture mismatch: refusing to restore incompatible checkpoint.")

        for k in curr_keys:
            if list(curr_sd[k].shape) != list(model_sd[k].shape):
                raise ValueError("Checkpoint architecture mismatch: refusing to restore incompatible checkpoint.")

        # Check metadata param_shapes if present
        meta_shapes = metadata.get("param_shapes", {})
        if meta_shapes:
            for k, expected_shape in meta_shapes.items():
                if k not in curr_sd or list(curr_sd[k].shape) != list(expected_shape):
                    raise ValueError("Checkpoint architecture mismatch: refusing to restore incompatible checkpoint.")

        # Load weights strictly
        network.load_state_dict(model_sd, strict=True)

        # Restore optimizer state if available and requested
        if optimizer is not None and opt_sd is not None:
            optimizer.load_state_dict(opt_sd)

        # Restore scheduler state if available and requested
        if scheduler is not None and sched_sd is not None:
            scheduler.load_state_dict(sched_sd)

        # Restore RNG states
        if restore_rng and rng_states:
            if "python" in rng_states and rng_states["python"] is not None:
                random.setstate(rng_states["python"])
            if np is not None and "numpy" in rng_states and rng_states["numpy"] is not None:
                np.random.set_state(rng_states["numpy"])
            if "torch" in rng_states and rng_states["torch"] is not None:
                torch.set_rng_state(rng_states["torch"])
            if torch.cuda.is_available() and "cuda" in rng_states and rng_states["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_states["cuda"])

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

    def delete_checkpoint(self, epoch: int) -> bool:
        """Delete a checkpoint's .pt and .json files."""
        path_pt = self._epoch_path_pt(epoch)
        path_json = self._epoch_path_json(epoch)
        deleted = False
        if os.path.exists(path_pt):
            os.remove(path_pt)
            deleted = True
        if os.path.exists(path_json):
            os.remove(path_json)
            deleted = True
        return deleted

    def apply_retention_policy(
        self,
        keep_latest_n: int = 5,
        keep_last_healthy: bool = True,
        last_healthy_epoch: Optional[int] = None,
        keep_best_loss: bool = True,
    ) -> List[int]:
        """
        Apply retention policy: retains latest N checkpoints, optionally the last
        healthy epoch, and optionally the best-loss epoch. Deletes the rest.
        Returns list of deleted epoch numbers.
        """
        all_epochs = self.list_epochs()
        if len(all_epochs) <= keep_latest_n:
            return []

        keep_set = set(all_epochs[-keep_latest_n:])

        if keep_last_healthy and last_healthy_epoch is not None and last_healthy_epoch in all_epochs:
            keep_set.add(last_healthy_epoch)

        if keep_best_loss:
            best_epoch = None
            min_loss = float("inf")
            for ep in all_epochs:
                try:
                    rec = self.load_record(ep)
                    loss = rec.get("loss")
                    if loss is not None and not math.isnan(loss) and loss < min_loss:
                        min_loss = loss
                        best_epoch = ep
                except Exception:
                    pass
            if best_epoch is not None:
                keep_set.add(best_epoch)

        deleted = []
        for ep in all_epochs:
            if ep not in keep_set:
                self.delete_checkpoint(ep)
                deleted.append(ep)
        return deleted

    def compare_checkpoints(self, epoch_a: int, epoch_b: int) -> Dict[str, Any]:
        """Compare metadata, losses, and weight differences between two checkpoint epochs."""
        rec_a = self.load_record(epoch_a)
        rec_b = self.load_record(epoch_b)

        path_pt_a = self._epoch_path_pt(epoch_a)
        path_pt_b = self._epoch_path_pt(epoch_b)

        data_a = torch.load(path_pt_a, map_location="cpu", weights_only=False)
        data_b = torch.load(path_pt_b, map_location="cpu", weights_only=False)

        sd_a = data_a.get("model_state_dict", data_a) if isinstance(data_a, dict) else data_a
        sd_b = data_b.get("model_state_dict", data_b) if isinstance(data_b, dict) else data_b

        weight_diffs = {}
        for k in sd_a.keys():
            if k in sd_b:
                diff = float(torch.norm(sd_b[k].float() - sd_a[k].float()).item())
                weight_diffs[k] = round(diff, 6)

        return {
            "epoch_a": epoch_a,
            "epoch_b": epoch_b,
            "loss_a": rec_a.get("loss"),
            "loss_b": rec_b.get("loss"),
            "accuracy_a": rec_a.get("accuracy"),
            "accuracy_b": rec_b.get("accuracy"),
            "param_norm_diffs": weight_diffs,
            "total_weight_movement": round(sum(weight_diffs.values()), 6),
        }

    def export_checkpoint(self, epoch: int, export_dir: str = "exports") -> str:
        """Export a checkpoint (.pt and .json) as a single zip archive."""
        import zipfile
        os.makedirs(export_dir, exist_ok=True)
        path_pt = self._epoch_path_pt(epoch)
        path_json = self._epoch_path_json(epoch)
        if not os.path.exists(path_pt) or not os.path.exists(path_json):
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} does not exist.")

        archive_name = f"checkpoint_epoch_{epoch:05d}.zip"
        archive_path = os.path.join(export_dir, archive_name)

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(path_pt, arcname=os.path.basename(path_pt))
            zf.write(path_json, arcname=os.path.basename(path_json))

        return archive_path

    def import_checkpoint(self, archive_path: str) -> int:
        """Import a zipped checkpoint into the current checkpoint directory. Returns imported epoch."""
        import zipfile
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive not found at {archive_path}")

        imported_epoch = -1
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                zf.extract(member, self.checkpoint_dir)
                if member.endswith(".json"):
                    try:
                        ep = int(member.replace("checkpoint_epoch_", "").replace(".json", ""))
                        imported_epoch = ep
                    except ValueError:
                        pass
        return imported_epoch

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
