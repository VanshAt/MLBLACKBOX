"""
backprop.py — Backpropagation engine for MLBlackBox.

This module provides the top-level backprop function that orchestrates
gradient flow from the loss through the network.

The actual per-layer and per-neuron chain-rule math lives in:
  layer.backward()  → layer.py
  neuron.backward() → neuron.py

This module ties them together and provides NaN/Inf detection during
the backward pass for early fault diagnosis.
"""

import math
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.network import Network


def backprop(network: "Network", loss_gradient: List[float]) -> None:
    """
    Run the full backward pass through the network.

    Propagates the loss gradient backward through every layer using the
    chain rule. Each layer updates its neurons' gradient accumulators.
    Weight update is a separate step (see training/updater.py).

    Also performs NaN/Inf detection at each layer boundary — if NaN is
    introduced during backprop, this identifies the originating layer
    before it spreads, enabling the fault detector to pinpoint the source.

    Args:
        network       : the Network instance (after a forward pass)
        loss_gradient : dL/d_output — list of floats from the loss derivative.
                        Length must equal network output size.

    Raises:
        ValueError: if loss_gradient contains NaN or Inf (caught before propagation)
    """
    # Validate that the incoming gradient is clean
    _check_gradient_health(loss_gradient, label="loss_gradient")

    gradient = list(loss_gradient)

    # Traverse layers in reverse — output layer first, input layer last
    for layer_idx in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[layer_idx]
        gradient = layer.backward(gradient)

        # Check for NaN/Inf after each layer to locate fault source
        # The fault_detector also checks this, but doing it here provides
        # layer-level precision for the backtracker's root cause analysis.
        if _has_nan_or_inf(gradient):
            # Don't raise — let the fault detector handle it with full context.
            # Mark the gradient vector so the fault detector knows which layer.
            # We log this via a module-level flag that the trainer can read.
            _last_bad_layer[0] = layer_idx
            break

    return


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

# Module-level state: index of the layer where NaN/Inf first appeared.
# Trainer reads this after each backward pass.
_last_bad_layer: List[int] = [-1]


def get_last_nan_layer() -> int:
    """
    Return the index of the layer where NaN/Inf was first detected
    during the most recent backward pass. Returns -1 if no fault occurred.
    """
    return _last_bad_layer[0]


def reset_nan_tracker():
    """Reset the NaN layer tracker. Call before each backward pass."""
    _last_bad_layer[0] = -1


def _has_nan_or_inf(values: List[float]) -> bool:
    """Return True if any value in the list is NaN or Inf."""
    for v in values:
        if math.isnan(v) or math.isinf(v):
            return True
    return False


def _check_gradient_health(values: List[float], label: str = "gradient"):
    """
    Validate that a gradient vector contains no NaN or Inf.
    Raises ValueError with a descriptive message if invalid.
    """
    for i, v in enumerate(values):
        if math.isnan(v):
            raise ValueError(
                f"NaN detected in {label}[{i}] before backprop started. "
                f"Check your loss function and forward pass for overflow."
            )
        if math.isinf(v):
            raise ValueError(
                f"Inf detected in {label}[{i}] before backprop started. "
                f"Check your loss function for division by zero or overflow."
            )
