"""
updater.py — Gradient descent weight updater for MLBlackBox.

Applies the gradient descent update rule to every weight and bias
in the network after a backward pass:

    weight = weight - learning_rate * weight_gradient
    bias   = bias   - learning_rate * bias_gradient

This is called after every forward+backward pass (SGD) or after
accumulating gradients over a mini-batch.

After updating, all gradient accumulators are cleared so they don't
accumulate across multiple epochs or batches.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.network import Network


def update_weights(network: "Network", learning_rate: float) -> None:
    """
    Apply gradient descent to all weights and biases in the network.

    Args:
        network      : the Network instance after forward + backward pass
        learning_rate: step size for gradient descent (typical: 0.001 to 0.01)

    Notes:
        - Gradients are cleared after update (prevents cross-epoch accumulation)
        - NaN/Inf in gradients is skipped with a warning rather than crashing,
          allowing the fault detector to handle the diagnosis
    """
    for layer_idx, layer in enumerate(network.layers):
        for neuron_idx, neuron in enumerate(layer.neurons):
            # Update each weight
            for i in range(len(neuron.weights)):
                grad = neuron.weight_gradients[i]

                # Skip NaN/Inf updates — let fault detector handle diagnosis
                if math.isnan(grad) or math.isinf(grad):
                    continue

                neuron.weights[i] -= learning_rate * grad

            # Update bias
            b_grad = neuron.bias_gradient
            if not (math.isnan(b_grad) or math.isinf(b_grad)):
                neuron.bias -= learning_rate * b_grad

    # Clear all gradients after update
    network.clear_gradients()


def clip_gradients(network: "Network", max_norm: float = 1.0) -> None:
    """
    Gradient clipping — caps all gradients to [-max_norm, +max_norm].

    Call this BEFORE update_weights() if gradient explosion is detected
    or suspected. This is one of the fixes suggested by the backtracker.

    Args:
        network  : the Network instance after backward pass
        max_norm : maximum absolute gradient value (default 1.0)
    """
    for layer in network.layers:
        for neuron in layer.neurons:
            neuron.weight_gradients = [
                max(-max_norm, min(max_norm, g))
                for g in neuron.weight_gradients
            ]
            neuron.bias_gradient = max(
                -max_norm, min(max_norm, neuron.bias_gradient)
            )
