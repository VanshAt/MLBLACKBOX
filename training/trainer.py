"""
trainer.py — Main training loop for MLBlackBox.

Ties the neural network engine and fault tracking system together
via a callback-based architecture. Callbacks are fired after each
epoch without modifying the core training logic.

Usage:
    from training.trainer import Trainer
    from core.network import Network
    from core.loss import MSE
    from core.activations import ReLU, Linear

    net = Network([2, 4, 1], activations=[ReLU(), Linear()])
    trainer = Trainer(net, loss_fn=MSE(), learning_rate=0.01)
    history = trainer.train(X, y, epochs=1000)
"""

import math
import time
from typing import List, Callable, Optional, Tuple, Any

from core.network import Network
from core.backprop import backprop, reset_nan_tracker, get_last_nan_layer
from training.updater import update_weights, clip_gradients


class EpochRecord:
    """Data object passed to callbacks after each epoch."""

    def __init__(
        self,
        epoch: int,
        loss: float,
        accuracy: Optional[float],
        epoch_time_ms: float,
        batch_idx: int,
        network: Network,
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
    Orchestrates the training loop for a feedforward network.

    Supports:
      - Per-epoch callbacks (checkpoint, metric recorder, fault detector)
      - Gradient clipping (enabled by setting clip_norm)
      - Classification mode (computes accuracy alongside loss)
      - Early stopping via callback return value
    """

    def __init__(
        self,
        network: Network,
        loss_fn,
        learning_rate: float = 0.01,
        clip_norm: Optional[float] = None,
        classification: bool = False,
        print_every: int = 100,
    ):
        """
        Args:
            network       : the Network instance to train
            loss_fn       : loss function object (MSE, BinaryCrossEntropy, etc.)
            learning_rate : gradient descent step size
            clip_norm     : if set, gradients are clipped to this absolute max
                            before the weight update. Set to 1.0 to prevent explosion.
            classification: if True, computes accuracy after each epoch
            print_every   : print progress every N epochs (0 = silent)
        """
        self.network = network
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.classification = classification
        self.print_every = print_every
        self.callbacks: List[Callable[[EpochRecord], Optional[bool]]] = []

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
        X: List[List[float]],
        y: List[List[float]],
        epochs: int,
    ) -> List[dict]:
        """
        Run the full training loop.

        Args:
            X     : list of input samples, each a list of floats
            y     : list of target values, each a list of floats
                    (wrap scalar targets in a list: [[0], [1], ...])
            epochs: number of complete passes over the dataset

        Returns:
            list of dicts — one record per epoch with epoch, loss, accuracy
        """
        history = []
        n_samples = len(X)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            correct = 0
            last_batch_idx = 0
            nan_layer = -1

            for batch_idx, (x_sample, y_sample) in enumerate(zip(X, y)):
                last_batch_idx = batch_idx
                reset_nan_tracker()

                # Forward pass
                prediction = self.network.forward(x_sample)

                # Compute loss
                loss_val = self.loss_fn.forward(prediction, y_sample)

                # Guard against NaN/Inf loss before proceeding
                if math.isnan(loss_val) or math.isinf(loss_val):
                    nan_layer = get_last_nan_layer()
                    total_loss += 0.0  # don't contaminate aggregate
                    continue

                total_loss += loss_val

                # Classification accuracy
                if self.classification:
                    if len(y_sample) == 1:
                        pred_class = 1 if prediction[0] >= 0.5 else 0
                        if pred_class == int(y_sample[0]):
                            correct += 1
                    else:
                        pred_class = prediction.index(max(prediction))
                        true_class = y_sample.index(max(y_sample))
                        if pred_class == true_class:
                            correct += 1

                # Backward pass
                loss_grad = self.loss_fn.derivative(prediction, y_sample)

                try:
                    backprop(self.network, loss_grad)
                except ValueError:
                    # NaN in loss gradient — skip this sample
                    self.network.clear_gradients()
                    nan_layer = 0  # originating layer unknown
                    continue

                # Check if NaN propagated during backprop
                detected_nan_layer = get_last_nan_layer()
                if detected_nan_layer >= 0:
                    nan_layer = detected_nan_layer

                # Gradient clipping (if enabled)
                if self.clip_norm is not None:
                    clip_gradients(self.network, self.clip_norm)

                # Weight update
                update_weights(self.network, self.learning_rate)

            # --- End of epoch ---
            epoch_time_ms = (time.time() - epoch_start) * 1000
            avg_loss = total_loss / n_samples if n_samples > 0 else float("nan")
            accuracy = (correct / n_samples) if self.classification else None

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
