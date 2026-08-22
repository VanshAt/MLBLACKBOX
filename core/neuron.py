"""
neuron.py — Single neuron implementation for MLBlackBox.

A neuron is the fundamental computational unit:
  z      = w1*x1 + w2*x2 + ... + wn*xn + bias
  output = activation(z)

During backpropagation the stored last_input and last_z are required
to compute weight gradients. This is the most commonly missed detail
when implementing backprop from scratch.
"""

import random
from typing import List, Optional
from core.activations import ReLU, get_activation


class Neuron:
    """
    A single neuron with weights, bias, activation function,
    and gradient storage for backpropagation.

    Attributes:
        weights         : list of floats, one per input
        bias            : float
        activation      : activation function object
        last_input      : inputs from most recent forward pass (needed for backprop)
        last_z          : pre-activation value from most recent forward pass
        weight_gradients: accumulated gradients for each weight (cleared after update)
        bias_gradient   : accumulated gradient for the bias (cleared after update)
    """

    def __init__(self, n_inputs: int, activation=None, seed: Optional[int] = None):
        """
        Args:
            n_inputs  : number of inputs this neuron receives
            activation: activation function instance (defaults to ReLU)
            seed      : optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Critical: small initialization prevents saturation before training starts.
        # Range [-0.1, 0.1] ensures initial outputs are near zero and gradients flow.
        self.weights: List[float] = [
            random.uniform(-0.1, 0.1) for _ in range(n_inputs)
        ]
        self.bias: float = 0.0

        self.activation = activation if activation is not None else ReLU()

        # State stored during forward pass — required for backprop
        self.last_input: List[float] = []
        self.last_z: float = 0.0

        # Gradient accumulators — set during backprop, cleared after weight update
        self.weight_gradients: List[float] = [0.0] * n_inputs
        self.bias_gradient: float = 0.0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: List[float]) -> float:
        """
        Compute neuron output from inputs.

        Steps:
          1. Compute z = dot(weights, inputs) + bias
          2. Store inputs and z for use during backprop
          3. Return activation(z)

        Args:
            inputs: list of float values, length must equal len(self.weights)

        Returns:
            float: activated output of this neuron
        """
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Neuron expected {len(self.weights)} inputs, got {len(inputs)}"
            )

        # Compute weighted sum
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

        # Store for backpropagation — do not skip this
        self.last_input = list(inputs)
        self.last_z = z

        return self.activation.forward(z)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, incoming_gradient: float) -> List[float]:
        """
        Compute and accumulate gradients for weights and bias.
        Returns the gradient to pass back to the previous layer.

        Chain rule breakdown:
          delta           = incoming_gradient × activation'(last_z)
          dL/dw_i         = delta × last_input[i]   → accumulated into weight_gradients
          dL/db           = delta                    → accumulated into bias_gradient
          dL/d_input[i]   = delta × weights[i]       → returned for previous layer

        Args:
            incoming_gradient: dL/d_output of this neuron (from next layer or loss)

        Returns:
            List[float]: gradients w.r.t. each input (one per input position)
        """
        # Gradient of loss w.r.t. z (pre-activation)
        activation_derivative = self.activation.derivative(self.last_z)
        delta = incoming_gradient * activation_derivative

        # Accumulate weight and bias gradients
        for i, x in enumerate(self.last_input):
            self.weight_gradients[i] += delta * x
        self.bias_gradient += delta

        # Gradient to send to previous layer (for each input position)
        input_gradients = [delta * w for w in self.weights]
        return input_gradients

    # ------------------------------------------------------------------
    # Parameter access (for checkpointing and restoration)
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return serializable dict of all parameters."""
        return {
            "weights": list(self.weights),
            "bias": self.bias,
            "activation": self.activation.name,
            "n_inputs": len(self.weights),
        }

    def set_params(self, params: dict):
        """Restore weights and bias from a checkpoint dict."""
        self.weights = list(params["weights"])
        self.bias = params["bias"]
        self.activation = get_activation(params["activation"])
        # Reset gradient accumulators on restore
        self.weight_gradients = [0.0] * len(self.weights)
        self.bias_gradient = 0.0

    def clear_gradients(self):
        """Zero out gradient accumulators. Call after each weight update."""
        self.weight_gradients = [0.0] * len(self.weights)
        self.bias_gradient = 0.0

    def __repr__(self):
        return (
            f"Neuron(n_inputs={len(self.weights)}, "
            f"activation={self.activation}, "
            f"bias={self.bias:.4f})"
        )
