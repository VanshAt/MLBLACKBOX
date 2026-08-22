"""
layer.py — A layer of neurons for MLBlackBox.

A Layer holds N neurons, all receiving the same input vector.
Each neuron produces one output value. The layer's output is the
concatenation of all neuron outputs.

Example: Layer(3, 4, relu) = 4 neurons, each with 3 inputs → 4 outputs.
  Parameters: 4 × 3 weights + 4 biases = 16 total parameters.
"""

from typing import List, Optional
from core.neuron import Neuron
from core.activations import ReLU, get_activation


class Layer:
    """
    A fully-connected layer containing a fixed number of neurons.

    All neurons receive the same input. Each neuron produces one output.
    The layer's forward pass returns a list of outputs (one per neuron).
    The layer's backward pass accepts incoming gradients (one per neuron)
    and returns the gradients to propagate to the previous layer.
    """

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        activation=None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_inputs  : number of input values each neuron receives
            n_neurons : number of neurons in this layer
            activation: activation function instance (shared prototype, each neuron
                        gets its own instance via the same class)
            seed      : optional seed for reproducible weight initialization
        """
        act_class = type(activation) if activation is not None else ReLU

        # Each neuron gets a separate seed offset for weight diversity
        self.neurons: List[Neuron] = [
            Neuron(
                n_inputs,
                activation=act_class(),
                seed=(seed + i) if seed is not None else None,
            )
            for i in range(n_neurons)
        ]

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Stored during forward pass — needed when examining layer inputs externally
        self.last_input: List[float] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Run every neuron on the input vector.

        Args:
            inputs: list of floats, length == n_inputs

        Returns:
            list of floats, length == n_neurons (one per neuron's output)
        """
        self.last_input = list(inputs)
        return [neuron.forward(inputs) for neuron in self.neurons]

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, incoming_gradients: List[float]) -> List[float]:
        """
        Backpropagate gradients through all neurons in this layer.

        Each neuron receives one scalar gradient (the dL/d_output for that neuron).
        Each neuron returns gradients w.r.t. its inputs.
        Since all neurons share the same inputs, we SUM their input-gradients
        at each input position to get the total gradient for that input.

        Args:
            incoming_gradients: list of floats, one per neuron (length == n_neurons)

        Returns:
            list of floats, one per input position (length == n_inputs)
              — the gradient to propagate to the previous layer
        """
        if len(incoming_gradients) != len(self.neurons):
            raise ValueError(
                f"Layer expected {len(self.neurons)} incoming gradients, "
                f"got {len(incoming_gradients)}"
            )

        # Accumulate input-gradients across all neurons
        # Start at zero for each input position
        input_grads = [0.0] * self.n_inputs

        for neuron, grad in zip(self.neurons, incoming_gradients):
            neuron_input_grads = neuron.backward(grad)
            for i, g in enumerate(neuron_input_grads):
                input_grads[i] += g

        return input_grads

    # ------------------------------------------------------------------
    # Parameter access (for checkpointing and restoration)
    # ------------------------------------------------------------------

    def get_params(self) -> List[dict]:
        """Return list of parameter dicts, one per neuron."""
        return [neuron.get_params() for neuron in self.neurons]

    def set_params(self, params: List[dict]):
        """Restore all neuron parameters from a list of dicts."""
        if len(params) != len(self.neurons):
            raise ValueError(
                f"Layer has {len(self.neurons)} neurons but received "
                f"params for {len(params)} neurons."
            )
        for neuron, p in zip(self.neurons, params):
            neuron.set_params(p)

    def clear_gradients(self):
        """Zero gradient accumulators in all neurons."""
        for neuron in self.neurons:
            neuron.clear_gradients()

    def get_all_weight_gradients(self) -> List[List[float]]:
        """
        Return all weight gradients from this layer.
        Shape: [n_neurons][n_inputs]
        Used by the gradient tracker for fault detection.
        """
        return [list(neuron.weight_gradients) for neuron in self.neurons]

    def __repr__(self):
        return (
            f"Layer(n_inputs={self.n_inputs}, n_neurons={self.n_neurons}, "
            f"activation={self.neurons[0].activation if self.neurons else 'N/A'})"
        )
