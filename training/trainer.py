"""
trainer.py — Main training loop for MLBlackBox.

Ties the neural network engine (PyTorch) and fault tracking system together
via a callback-based architecture. Callbacks are fired after each
epoch without modifying the core training logic.
"""

import math
import time
import torch
from typing import List, Callable, Optional, Tuple, Any


from tracker.gradient_tracker import GradientAccumulator


class EpochRecord:
    """Data object passed to callbacks after each epoch."""

    def __init__(
        self,
        epoch: int,
        loss: float,
        accuracy: Optional[float],
        epoch_time_ms: float,
        batch_idx: int,
        network: torch.nn.Module,
        nan_layer: int,
        learning_rate: float,
        gradient_stats: Optional[dict] = None,
        pre_clip_norm: Optional[float] = None,
        post_clip_norm: Optional[float] = None,
    ):
        self.epoch = epoch
        self.loss = loss
        self.accuracy = accuracy
        self.epoch_time_ms = epoch_time_ms
        self.batch_idx = batch_idx  # index of last processed batch (for fault tracking)
        self.network = network
        self.nan_layer = nan_layer   # -1 if no NaN detected
        self.learning_rate = learning_rate
        self.gradient_stats = gradient_stats or {}
        self.pre_clip_norm = pre_clip_norm
        self.post_clip_norm = post_clip_norm


class Trainer:
    """
    Orchestrates the PyTorch training loop for a feedforward network.

    Supports:
      - Per-epoch callbacks (checkpoint, metric recorder, fault detector)
      - Gradient clipping (enabled by setting clip_norm)
      - Classification mode (computes accuracy alongside loss)
      - Early stopping via callback return value
      - Real-time gradient accumulation immediately post-backward
      - Input validation and instantaneous NaN/Inf propagation detection
    """

    def __init__(
        self,
        network: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_norm: Optional[float] = None,
        classification: bool = False,
        print_every: int = 100,
        scheduler: Optional[Any] = None,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        checkpoint_manager: Optional[Any] = None,
        metric_recorder: Optional[Any] = None,
    ):
        """
        Args:
            network            : the PyTorch Module to train
            loss_fn            : PyTorch loss function object (nn.MSELoss, nn.BCELoss, etc.)
            optimizer          : PyTorch Optimizer (e.g., torch.optim.SGD)
            clip_norm          : if set, gradients are clipped to this absolute max
                                 before the weight update. Set to 1.0 to prevent explosion.
            classification     : if True, computes accuracy after each epoch
            print_every        : print progress every N epochs (0 = silent)
            scheduler          : optional PyTorch learning rate scheduler
            run_id             : optional unique identifier for this training run
            parent_run_id      : optional parent run identifier if resumed
            checkpoint_manager : optional CheckpointManager instance
            metric_recorder    : optional MetricRecorder instance
        """
        from runs.session import RunSessionManager

        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.classification = classification
        self.print_every = print_every
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.metric_recorder = metric_recorder
        self.callbacks: List[Callable[[EpochRecord], Optional[bool]]] = []
        
        # Session Tracking
        self.session_manager = RunSessionManager()
        self.parent_run_id = parent_run_id
        ckpt_dir = self.checkpoint_manager.checkpoint_dir if self.checkpoint_manager else "checkpoints"
        log_path = self.metric_recorder.log_path if self.metric_recorder else "logs/audit_log.json"
        self.run_id = run_id or self.session_manager.start_run(
            model_name=network.__class__.__name__,
            parent_run_id=parent_run_id,
            checkpoint_dir=ckpt_dir,
            log_path=log_path,
        )

        # Last training inputs cached for convenient resumption
        self._last_X: Optional[torch.Tensor] = None
        self._last_y: Optional[torch.Tensor] = None
        self._current_epoch: int = 0

        # Determine learning rate from optimizer
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        self.grad_accumulator = GradientAccumulator(self.network)

    def add_callback(self, fn: Callable[[EpochRecord], Optional[bool]]):
        """
        Register a callback fired after each epoch.

        The callback receives an EpochRecord. If it returns True,
        training stops early (e.g., fault detector triggering a halt).
        """
        self.callbacks.append(fn)
        return self  # fluent interface

    def resume(
        self,
        checkpoint_epoch_or_path: Any,
        lr: Optional[float] = None,
        clip_norm: Optional[float] = None,
        epochs: Optional[int] = None,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        checkpoint_manager: Optional[Any] = None,
    ) -> List[dict]:
        """
        Resume training from a specific checkpoint epoch or path with optional
        adjusted hyperparameter overrides (learning rate, gradient clipping).

        Ensures full state (weights, optimizer, scheduler, RNG) is restored.
        Assigns a new run_id linked to the original run as parent_run_id.
        """
        from tracker.checkpoint import CheckpointManager

        mgr = checkpoint_manager or self.checkpoint_manager
        if mgr is None:
            if isinstance(checkpoint_epoch_or_path, str) and os.path.isdir(checkpoint_epoch_or_path):
                mgr = CheckpointManager(checkpoint_dir=checkpoint_epoch_or_path)
            else:
                mgr = CheckpointManager()

        if isinstance(checkpoint_epoch_or_path, int):
            target_epoch = checkpoint_epoch_or_path
        elif isinstance(checkpoint_epoch_or_path, str):
            # Extract epoch number if file path given
            base = os.path.basename(checkpoint_epoch_or_path)
            ep_str = base.replace("checkpoint_epoch_", "").replace(".pt", "").replace(".json", "")
            target_epoch = int(ep_str)
        else:
            raise ValueError(f"Invalid checkpoint specifier: {checkpoint_epoch_or_path}")

        # 1. Restore weights, optimizer, scheduler, RNG
        metadata = mgr.load(
            target_epoch,
            self.network,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            restore_rng=True,
        )

        # 2. Adjust hyperparameters if requested
        if lr is not None:
            self.learning_rate = lr
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
                self.scheduler.base_lrs = [lr for _ in self.optimizer.param_groups]
        if clip_norm is not None:
            self.clip_norm = clip_norm

        # 3. Create new resumed run session linked to parent
        old_run_id = self.run_id
        self.parent_run_id = old_run_id
        self.run_id = self.session_manager.start_run(
            model_name=self.network.__class__.__name__,
            parent_run_id=old_run_id,
            checkpoint_dir=mgr.checkpoint_dir,
            log_path=self.metric_recorder.log_path if self.metric_recorder else "logs/audit_log.json",
        )

        # 4. Log recovery event in audit log if metric recorder is active
        if self.metric_recorder is not None:
            self.metric_recorder.append(
                epoch=target_epoch,
                loss=metadata.get("loss", 0.0),
                accuracy=metadata.get("accuracy"),
                gradient_stats=metadata.get("gradient_stats", {}),
                weight_stats=metadata.get("weight_stats", {}),
                learning_rate=self.learning_rate,
                epoch_time_ms=0.0,
                batch_idx=metadata.get("batch_idx", 0),
                fault_detected=f"RESUMED_FROM_EPOCH_{target_epoch}",
            )

        # 5. Continue training
        resume_epochs = epochs or 50
        data_X = X if X is not None else self._last_X
        data_y = y if y is not None else self._last_y

        if data_X is None or data_y is None:
            print(f"Model restored to epoch {target_epoch}. Ready for training calls.")
            return []

        return self.train(data_X, data_y, epochs=resume_epochs, start_epoch=target_epoch + 1)

    def resume_from_last_healthy_checkpoint(
        self,
        fault_epoch: Optional[int] = None,
        lr: Optional[float] = None,
        clip_norm: Optional[float] = None,
        epochs: Optional[int] = None,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> List[dict]:
        """
        One-call automated recovery: finds last healthy epoch, restores full state,
        applies safe hyperparameter overrides, and resumes training.
        """
        if self.checkpoint_manager is None:
            raise ValueError("No CheckpointManager attached to Trainer.")

        current_ep = fault_epoch or self._current_epoch
        clean_epoch = self.checkpoint_manager.get_last_clean_epoch(current_ep)
        if clean_epoch is None:
            clean_epoch = self.checkpoint_manager.get_last_epoch()

        if clean_epoch is None:
            raise FileNotFoundError("No healthy checkpoint found to resume from.")

        print(f"Automated Recovery: Restoring from clean checkpoint at epoch {clean_epoch}...")
        return self.resume(
            checkpoint_epoch_or_path=clean_epoch,
            lr=lr,
            clip_norm=clip_norm,
            epochs=epochs,
            X=X,
            y=y,
        )

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        start_epoch: int = 1,
    ) -> List[dict]:
        """
        Run the full training loop with input validation, immediate gradient stats,
        and instant NaN/Inf detection.
        """
        self._last_X = X
        self._last_y = y

        # Validate inputs X and y for NaN/Inf before training begins
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError("Input tensor X contains NaN or Inf values. Refusing to train on corrupted data.")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("Target tensor y contains NaN or Inf values. Refusing to train on corrupted data.")

        history = []
        n_samples = X.shape[0]
        final_status = "completed"
        final_fault = None
        best_loss = float("inf")
        best_acc = 0.0

        for epoch in range(start_epoch, start_epoch + epochs):
            self._current_epoch = epoch
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            last_batch_idx = 0
            nan_layer = -1
            nan_detected = False

            self.grad_accumulator.reset()

            for batch_idx in range(n_samples):
                x_sample = X[batch_idx:batch_idx+1]
                y_sample = y[batch_idx:batch_idx+1]
                
                last_batch_idx = batch_idx

                # Forward pass
                prediction = self.network(x_sample)

                # Check prediction for NaN/Inf
                if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                    nan_detected = True
                    nan_layer = 0
                    total_loss = float("nan")
                    break

                # Compute loss
                loss_val = self.loss_fn(prediction, y_sample)

                # Check loss value for NaN/Inf
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    nan_detected = True
                    nan_layer = 0
                    total_loss = float("nan")
                    break

                total_loss += loss_val.item()

                # Classification accuracy
                if self.classification:
                    if y_sample.shape[1] == 1:
                        pred_class = 1 if prediction.item() >= 0.5 else 0
                        if pred_class == int(y_sample.item()):
                            correct += 1
                    else:
                        pred_class = torch.argmax(prediction).item()
                        true_class = torch.argmax(y_sample).item()
                        if pred_class == true_class:
                            correct += 1

                # Backward pass
                self.optimizer.zero_grad()
                loss_val.backward()

                # Check if NaN/Inf occurred in gradients
                detected_nan_layer = -1
                layer_idx = 0
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            detected_nan_layer = layer_idx
                            break
                    layer_idx += 1
                
                if detected_nan_layer >= 0:
                    nan_detected = True
                    nan_layer = detected_nan_layer
                    total_loss = float("nan")
                    break

                # Compute gradient statistics immediately after backward() and before step()
                pre_clip_norm = None
                post_clip_norm = None
                if self.clip_norm is not None:
                    pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_norm).item())
                    post_clip_norm = min(pre_clip_norm, float(self.clip_norm))

                self.grad_accumulator.capture_batch(
                    pre_clip_norm=pre_clip_norm,
                    post_clip_norm=post_clip_norm,
                )

                # Weight update
                self.optimizer.step()

                # Check optimizer buffers (e.g. Adam momentum/variance) for NaN/Inf
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.optimizer.state.get(p, {})
                        for k, v in state.items():
                            if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
                                nan_detected = True
                                nan_layer = 0
                                total_loss = float("nan")
                                break
                        if nan_detected:
                            break
                    if nan_detected:
                        break

                if nan_detected:
                    break

            # If scheduler is present and no NaN, step scheduler
            if self.scheduler is not None and not nan_detected:
                self.scheduler.step()

            # --- End of epoch ---
            epoch_time_ms = (time.time() - epoch_start) * 1000
            if nan_detected:
                avg_loss = float("nan")
            else:
                avg_loss = total_loss / n_samples if n_samples > 0 else float("nan")
            accuracy = (correct / n_samples) if (self.classification and not nan_detected) else None

            # Update learning rate tracker in case it changed
            self.learning_rate = self.optimizer.param_groups[0]['lr']

            accumulated_grad_stats = self.grad_accumulator.compute_epoch_stats()

            record = {
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy,
                "epoch_time_ms": round(epoch_time_ms, 2),
                "batch_idx": last_batch_idx,
                "nan_layer": nan_layer,
                "learning_rate": self.learning_rate,
            }
            history.append(record)

            # Print progress
            if self.print_every > 0 and epoch % self.print_every == 0:
                acc_str = f"  acc={accuracy:.4f}" if accuracy is not None else ""
                print(f"Epoch {epoch:>6}/{epochs}  loss={avg_loss:.6f}{acc_str}")

            # Fire callbacks — if any returns True, stop training
            epoch_obj = EpochRecord(
                epoch=epoch,
                loss=avg_loss,
                accuracy=accuracy,
                epoch_time_ms=epoch_time_ms,
                batch_idx=last_batch_idx,
                network=self.network,
                nan_layer=nan_layer,
                learning_rate=self.learning_rate,
                gradient_stats=accumulated_grad_stats,
                pre_clip_norm=accumulated_grad_stats.get("pre_clip_global_norm"),
                post_clip_norm=accumulated_grad_stats.get("post_clip_global_norm"),
            )
            should_stop = False
            for cb in self.callbacks:
                result = cb(epoch_obj)
                if result is True:
                    should_stop = True

            if avg_loss is not None and not math.isnan(avg_loss) and avg_loss < best_loss:
                best_loss = avg_loss
            if accuracy is not None and accuracy > best_acc:
                best_acc = accuracy

            if nan_detected or should_stop:
                final_status = "fault_detected" if nan_detected else "manually_stopped"
                final_fault = "NAN_INF_PROPAGATION" if nan_detected else "EARLY_STOPPING_CALLBACK"
                print(f"Training stopped early at epoch {epoch} by {'NaN/Inf trigger' if nan_detected else 'callback'}.")
                break

        # Record session completion
        self.session_manager.finish_run(
            run_id=self.run_id,
            status=final_status,
            total_epochs=len(history),
            best_loss=best_loss if best_loss != float("inf") else None,
            best_accuracy=best_acc if self.classification else None,
            fault_summary=final_fault,
        )

        return history
