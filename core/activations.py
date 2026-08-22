"""
activations.py — Activation functions for MLBlackBox neural network engine.

Each activation has:
  forward(z)     — computes output from pre-activation value z
  derivative(z)  — computes gradient of activation w.r.t. z

Usage notes:
  ReLU    → use for all hidden layers (no vanishing gradient)
  Sigmoid → use only on output layer for binary classification
  Linear  → use on output layer for regression
"""

import math


class ReLU:
    """
    Rectified Linear Unit.
    forward:    max(0, z)
    derivative: 1 if z > 0 else 0

    Why: derivative is always 0 or 1, preventing gradient shrinkage across
    many layers. The de-facto standard for hidden layers in modern networks.
    """

    name = "relu"

    def forward(self, z: float) -> float:
        return max(0.0, z)

    def derivative(self, z: float) -> float:
        return 1.0 if z > 0 else 0.0

    def __repr__(self):
        return "ReLU()"


class Sigmoid:
    """
    Logistic sigmoid function.
    forward:    1 / (1 + e^(-z))
    derivative: sigmoid(z) * (1 - sigmoid(z))   [always ≤ 0.25]

    WARNING: Do NOT use in hidden layers of deep networks. The derivative
    is at most 0.25, which compounds to near-zero gradients across many layers
    (vanishing gradient problem). Use only on output layers for binary classification.
    """

    name = "sigmoid"

    def forward(self, z: float) -> float:
        # Clamp z to prevent overflow in exp
        z_clamped = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def derivative(self, z: float) -> float:
        s = self.forward(z)
        return s * (1.0 - s)

    def __repr__(self):
        return "Sigmoid()"


class Linear:
    """
    Identity / linear activation. Output = z, derivative = 1.
    Use on the output layer for regression problems where the output
    can be any real-valued number.
    """

    name = "linear"

    def forward(self, z: float) -> float:
        return z

    def derivative(self, z: float) -> float:
        return 1.0

    def __repr__(self):
        return "Linear()"


# Convenience instances
relu = ReLU()
sigmoid = Sigmoid()
linear = Linear()

# Registry for loading activations by name (used by checkpoint restore)
ACTIVATION_REGISTRY = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "linear": Linear,
}


def get_activation(name: str):
    """Return an activation instance by string name."""
    name = name.lower()
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(ACTIVATION_REGISTRY.keys())}"
        )
    return ACTIVATION_REGISTRY[name]()
