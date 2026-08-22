"""
network.py — Neural network that chains layers together for MLBlackBox.

The Network class composes Layer objects into a full feedforward network.
Data flows forward through layers sequentially; gradients flow backward
through layers sequentially during backpropagation.

Usage:
    from core.network import Network
    from core.activations import ReLU, Sigmoid

    # 3 inputs → hidden(4, relu) → hidden(4, relu) → output(1, sigmoid)
    net = Network([3, 4, 4, 1], activations=[ReLU(), ReLU(), Sigmoid()])
    output = net.forward([0.5, -0.3, 0.8])
"""

from typing import List, Optional
from core.layer import Layer
from core.activations import ReLU, Sigmoid, Linear, get_activation


class Network:
    """
    A feedforward neural network built from Layer objects.

    Architecture is defined by a list of sizes:
      sizes = [n_inputs, n_hidden_1, n_hidden_2, ..., n_output]

    Activations list must have exactly len(sizes) - 1 entries (one per layer).
    If omitted, defaults to ReLU for all hidden layers and Linear for output.
    """

    def __init__(
        self,
        sizes: List[int],
        activations: Optional[List] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            sizes      : list of ints defining the architecture.
                         e.g. [3, 4, 4, 1] → input_dim=3, two hidden layers of 4,
                         output_dim=1
            activations: list of activation instances, one per layer.
                         Length must be len(sizes) - 1.
                         Defaults: ReLU for hidden layers, Linear for output layer.
            seed       : optional random seed for reproducible initialization
        """
        if len(sizes) < 2:
            raise ValueError("Network requires at least 2 sizes (input + output).")

        n_layers = len(sizes) - 1

        # Build default activations: ReLU for hidden, Linear for output
        if activations is None:
            activations = [ReLU() for _ in range(n_layers - 1)] + [Linear()]
        elif len(activations) != n_layers:
            raise ValueError(
                f"Expected {n_layers} activations for {n_layers} layers, "
                f"got {len(activations)}."
            )

        self.layers: List[Layer] = []
        for i in range(n_layers):
            layer_seed = (seed + i * 100) if seed is not None else None
            self.layers.append(
                Layer(
                    n_inputs=sizes[i],
                    n_neurons=sizes[i + 1],
                    activation=activations[i],
                    seed=layer_seed,
                )
            )

        self.sizes = sizes
        self.n_layers = n_layers

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs: List[float]) -> List[float]:
        """
        Pass inputs through every layer sequentially.

        Args:
            inputs: list of floats, length == sizes[0]

        Returns:
            list of floats, length == sizes[-1] (network output)
        """
        current = list(inputs)
        for layer in self.layers:
            current = layer.forward(current)
        return current

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, loss_gradient: List[float]) -> None:
        """
        Propagate gradients backward through all layers.

        Args:
            loss_gradient: dL/d_output — list of floats, one per output neuron.
                           This comes from the loss function's derivative.
        """
        gradient = list(loss_gradient)
        # Traverse layers in reverse order (output → input)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    # ------------------------------------------------------------------
    # Parameter access (for checkpointing and restoration)
    # ------------------------------------------------------------------

    def get_all_params(self) -> List[List[dict]]:
        """
        Collect all parameters from every layer and neuron.

        Returns:
            list of layers, each a list of neuron param dicts.
            Shape: [layer_idx][neuron_idx] → dict(weights, bias, activation)
        """
        return [layer.get_params() for layer in self.layers]

    def set_all_params(self, params: List[List[dict]]):
        """
        Restore all weights and biases from a saved parameter structure.

        Args:
            params: same structure as returned by get_all_params()
        """
        if len(params) != len(self.layers):
            raise ValueError(
                f"Network has {len(self.layers)} layers but params has {len(params)} entries."
            )
        for layer, layer_params in zip(self.layers, params):
            layer.set_params(layer_params)

    def clear_gradients(self):
        """Zero all gradient accumulators across every neuron in every layer."""
        for layer in self.layers:
            layer.clear_gradients()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters (weights + biases)."""
        total = 0
        for layer in self.layers:
            for neuron in layer.neurons:
                total += len(neuron.weights) + 1  # +1 for bias
        return total

    def summary(self) -> str:
        """Return a human-readable architecture summary."""
        lines = ["Network Architecture:", "-" * 40]
        for i, layer in enumerate(self.layers):
            n = layer.neurons
            act = n[0].activation if n else "N/A"
            params = sum(len(neu.weights) + 1 for neu in n)
            lines.append(
                f"  Layer {i+1}: {layer.n_inputs} → {layer.n_neurons}  "
                f"activation={act}  params={params}"
            )
        lines.append("-" * 40)
        lines.append(f"  Total parameters: {self.count_parameters()}")
        return "\n".join(lines)

    def __repr__(self):
        return f"Network(sizes={self.sizes}, layers={self.n_layers})"
