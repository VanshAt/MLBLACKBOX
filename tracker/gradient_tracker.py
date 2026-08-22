"""
gradient_tracker.py — Per-layer gradient magnitude statistics for MLBlackBox.

After every backward pass, this module inspects all weight gradients
across every neuron in every layer and computes health statistics.

Per-layer tracking is critical: gradient explosion often starts in one
specific layer. Global averages hide this. If layer 2 explodes while
layers 1, 3, 4 are fine, the average masks the fault.

Output structure:
    {
        "global": {"max": ..., "min": ..., "mean": ..., "std": ...},
        "per_layer": [
            {"layer": 0, "max": ..., "min": ..., "mean": ..., "std": ...},
            {"layer": 1, ...},
            ...
        ]
    }
"""

import math
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.network import Network


def compute_gradient_stats(network: "Network") -> Dict[str, Any]:
    """
    Compute gradient magnitude statistics globally and per layer.

    Args:
        network: the Network instance immediately after a backward pass
                 (before clear_gradients() is called)

    Returns:
        dict with 'global' and 'per_layer' keys
    """
    all_grads: List[float] = []
    per_layer_stats = []

    for layer_idx, layer in enumerate(network.layers):
        layer_grads: List[float] = []

        for neuron in layer.neurons:
            # Collect weight gradients (absolute values = magnitude)
            for g in neuron.weight_gradients:
                if not (math.isnan(g) or math.isinf(g)):
                    layer_grads.append(abs(g))
                else:
                    # Use a sentinel that the fault detector will flag
                    layer_grads.append(float("nan"))

            # Include bias gradient
            bg = neuron.bias_gradient
            if not (math.isnan(bg) or math.isinf(bg)):
                layer_grads.append(abs(bg))
            else:
                layer_grads.append(float("nan"))

        # Filter out NaN for statistics computation
        clean_grads = [g for g in layer_grads if not math.isnan(g)]
        has_nan = len(clean_grads) < len(layer_grads)

        layer_stats = _compute_stats(clean_grads)
        layer_stats["layer"] = layer_idx
        layer_stats["has_nan"] = has_nan
        per_layer_stats.append(layer_stats)

        all_grads.extend(layer_grads)

    clean_all = [g for g in all_grads if not math.isnan(g)]
    global_stats = _compute_stats(clean_all)
    global_stats["has_nan"] = len(clean_all) < len(all_grads)

    return {
        "global": global_stats,
        "per_layer": per_layer_stats,
    }


def compute_weight_stats(network: "Network") -> Dict[str, float]:
    """
    Compute weight magnitude statistics across the entire network.
    Dangerously large weights are a precursor to gradient explosion.

    Returns:
        dict with max, min, mean
    """
    all_weights: List[float] = []

    for layer in network.layers:
        for neuron in layer.neurons:
            all_weights.extend(abs(w) for w in neuron.weights)
            all_weights.append(abs(neuron.bias))

    if not all_weights:
        return {"max": 0.0, "min": 0.0, "mean": 0.0}

    return {
        "max": max(all_weights),
        "min": min(all_weights),
        "mean": sum(all_weights) / len(all_weights),
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute max, min, mean, std for a list of floats."""
    if not values:
        return {"max": 0.0, "min": 0.0, "mean": 0.0, "std": 0.0}

    n = len(values)
    try:
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance) if variance >= 0 else 0.0
    except (OverflowError, ValueError):
        # Catastrophically large gradients — return inf to trigger fault detection
        return {"max": float("inf"), "min": 0.0, "mean": float("inf"), "std": 0.0}

    return {
        "max": max(values),
        "min": min(values),
        "mean": mean,
        "std": std,
    }
