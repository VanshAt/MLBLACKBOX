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
    ):
        self.epoch = epoch
        self.loss = loss
        self.accuracy = accuracy
        self.epoch_time_ms = epoch_time_ms
        self.batch_idx = batch_idx  # index of last processed batch (for fault tracking)
        self.network = network
        self.nan_layer = nan_layer   # -1 if no NaN detected
        self.learning_rate = learning_rate


class Trainer:
    """
    Orchestrates the PyTorch training loop for a feedforward network.

    Supports:
      - Per-epoch callbacks (checkpoint, metric recorder, fault detector)
      - Gradient clipping (enabled by setting clip_norm)
      - Classification mode (computes accuracy alongside loss)
      - Early stopping via callback return value
    """

    def __init__(
        self,
        network: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_norm: Optional[float] = None,
        classification: bool = False,
        print_every: int = 100,
    ):
        """
        Args:
            network       : the PyTorch Module to train
            loss_fn       : PyTorch loss function object (nn.MSELoss, nn.BCELoss, etc.)
            optimizer     : PyTorch Optimizer (e.g., torch.optim.SGD)
            clip_norm     : if set, gradients are clipped to this absolute max
                            before the weight update. Set to 1.0 to prevent explosion.
            classification: if True, computes accuracy after each epoch
            print_every   : print progress every N epochs (0 = silent)
        """
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.classification = classification
        self.print_every = print_every
        self.callbacks: List[Callable[[EpochRecord], Optional[bool]]] = []
        
        # Determine learning rate from optimizer
        self.learning_rate = self.optimizer.param_groups[0]['lr']

    def add_callback(self, fn: Callable[[EpochRecord], Optional[bool]]):
        """
        Register a callback fired after each epoch.

        The callback receives an EpochRecord. If it returns True,
        training stops early (e.g., fault detector triggering a halt).
        """
        self.callbacks.append(fn)
        return self  # fluent interface

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
    ) -> List[dict]:
        """
        Run the full training loop.

        Args:
            X     : Tensor of input samples [num_samples, input_dim]
            y     : Tensor of target values [num_samples, output_dim]
            epochs: number of complete passes over the dataset

        Returns:
            list of dicts — one record per epoch with epoch, loss, accuracy
        """
        history = []
        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            last_batch_idx = 0
            nan_layer = -1

            # In this simple implementation, we assume batch size = 1 
            # to match the old loop's per-sample tracking.
            for batch_idx in range(n_samples):
                x_sample = X[batch_idx:batch_idx+1]
                y_sample = y[batch_idx:batch_idx+1]
                
                last_batch_idx = batch_idx

                # Forward pass
                prediction = self.network(x_sample)

                # Compute loss
                loss_val = self.loss_fn(prediction, y_sample)

                # Guard against NaN/Inf loss before proceeding
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    nan_layer = 0 # Defaulting to 0 since we don't know the exact layer easily before backward
                    total_loss += 0.0  # don't contaminate aggregate
                    continue

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

                # Check if NaN propagated during backprop
                detected_nan_layer = -1
                layer_idx = 0
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            detected_nan_layer = layer_idx
                            break
                    layer_idx += 1
                
                if detected_nan_layer >= 0:
                    nan_layer = detected_nan_layer
                    self.optimizer.zero_grad() # Clear gradients to prevent nan weights
                    continue

                # Gradient clipping (if enabled)
                if self.clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_norm)

                # Weight update
                self.optimizer.step()

            # --- End of epoch ---
            epoch_time_ms = (time.time() - epoch_start) * 1000
            avg_loss = total_loss / n_samples if n_samples > 0 else float("nan")
            accuracy = (correct / n_samples) if self.classification else None

            # Update learning rate tracker in case it changed
            self.learning_rate = self.optimizer.param_groups[0]['lr']

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
            )
            should_stop = False
            for cb in self.callbacks:
                result = cb(epoch_obj)
                if result is True:
                    should_stop = True

            if should_stop:
                print(f"Training stopped early at epoch {epoch} by callback.")
                break

        return history
