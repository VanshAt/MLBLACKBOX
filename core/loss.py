"""
loss.py — Loss functions for MLBlackBox.

Loss functions measure the difference between network predictions and ground truth.
They produce a scalar value (lower = better).

The derivative of the loss is the starting gradient for backpropagation.
If the loss derivative is wrong, ALL training is wrong.

Available:
  MSE                  — Mean Squared Error (regression)
  BinaryCrossEntropy   — Binary Cross Entropy (binary classification, sigmoid output)
"""

import math
from typing import List, Union


# Type alias for scalar or list predictions
Prediction = Union[float, List[float]]


class MSE:
    """
    Mean Squared Error Loss.

    Forward:    (1/n) * Σ (predicted_i - actual_i)²
    Derivative: (2/n) * (predicted_i - actual_i)   [per sample]

    Use for regression problems where the output is a continuous value.
    The output neuron should use Linear activation.
    """

    name = "mse"

    def forward(self, predicted: List[float], actual: List[float]) -> float:
        """
        Compute MSE loss over a batch.

        Args:
            predicted: list of predicted values
            actual   : list of true target values

        Returns:
            float: scalar loss value
        """
        if len(predicted) != len(actual):
            raise ValueError(
                f"MSE: predicted length {len(predicted)} != actual length {len(actual)}"
            )
        n = len(predicted)
        import math
        total = 0.0
        for p, a in zip(predicted, actual):
            try:
                diff_sq = (p - a) ** 2
                if math.isinf(diff_sq) or math.isnan(diff_sq):
                    return float('nan')
                total += diff_sq
            except OverflowError:
                # Extreme outlier values — signal NaN so trainer's guard catches it
                return float('nan')
        return total / n

    def derivative(self, predicted: List[float], actual: List[float]) -> List[float]:
        """
        Gradient of MSE w.r.t. each predicted value.

        Returns:
            list of floats: one gradient per predicted value
        """
        n = len(predicted)
        return [(2.0 / n) * (p - a) for p, a in zip(predicted, actual)]

    def __repr__(self):
        return "MSE()"


class BinaryCrossEntropy:
    """
    Binary Cross Entropy Loss.

    Forward:    -( y * log(p + ε) + (1-y) * log(1 - p + ε) )
    Derivative: -(y/(p+ε)) + (1-y)/(1-p+ε)   [per sample]

    Use for binary classification where the output neuron uses Sigmoid activation.
    The ε (epsilon) term prevents log(0) which would produce NaN.

    WARNING: Using this with non-sigmoid output (values outside [0,1]) will
    produce NaN. The fault detector will catch this as NaN propagation.
    """

    name = "bce"
    EPSILON = 1e-8  # prevents log(0) and division by zero

    def forward(self, predicted: List[float], actual: List[float]) -> float:
        """
        Compute BCE loss over a batch.

        Args:
            predicted: list of predicted probabilities (should be in [0,1])
            actual   : list of true binary labels (0 or 1)

        Returns:
            float: scalar loss value
        """
        if len(predicted) != len(actual):
            raise ValueError(
                f"BCE: predicted length {len(predicted)} != actual length {len(actual)}"
            )
        eps = self.EPSILON
        total = 0.0
        for p, y in zip(predicted, actual):
            # Clamp p to prevent log(0)
            p_safe = max(eps, min(1.0 - eps, p))
            total += -(y * math.log(p_safe) + (1.0 - y) * math.log(1.0 - p_safe))
        return total / len(predicted)

    def derivative(self, predicted: List[float], actual: List[float]) -> List[float]:
        """
        Gradient of BCE w.r.t. each predicted value.

        Returns:
            list of floats: one gradient per predicted value
        """
        eps = self.EPSILON
        grads = []
        n = len(predicted)
        for p, y in zip(predicted, actual):
            p_safe = max(eps, min(1.0 - eps, p))
            grad = (-(y / p_safe) + (1.0 - y) / (1.0 - p_safe)) / n
            grads.append(grad)
        return grads

    def __repr__(self):
        return "BinaryCrossEntropy()"


# Convenience instances
mse = MSE()
bce = BinaryCrossEntropy()

# Registry for loading by name
LOSS_REGISTRY = {
    "mse": MSE,
    "bce": BinaryCrossEntropy,
    "binary_cross_entropy": BinaryCrossEntropy,
}


def get_loss(name: str):
    """Return a loss function instance by string name."""
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name]()
