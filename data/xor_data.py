"""
data/xor_data.py — XOR problem dataset for MLBlackBox tests.

The XOR function is the canonical test for neural networks:
  Input (0,0) → Output 0
  Input (0,1) → Output 1
  Input (1,0) → Output 1
  Input (1,1) → Output 0

It is NOT linearly separable — a single-layer network cannot learn it.
A network with at least one hidden layer and a non-linear activation can.

If your network's loss converges on XOR, your forward pass and backpropagation
are both implemented correctly.
"""

from typing import List, Tuple


def get_xor_data() -> Tuple[List[List[float]], List[List[float]]]:
    """
    Return the four XOR training samples.

    Returns:
        X: list of input vectors [[0,0], [0,1], [1,0], [1,1]]
        y: list of target vectors [[0], [1], [1], [0]]
           (wrapped in lists for compatibility with the trainer)
    """
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    y = [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]
    return X, y
