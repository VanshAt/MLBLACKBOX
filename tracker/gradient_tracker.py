"""
gradient_tracker.py — Per-layer gradient magnitude statistics for MLBlackBox.

After every backward pass, this module inspects all weight gradients
across every parameter in every layer and computes health statistics.

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
import torch
from typing import List, Dict, Any


def compute_gradient_stats(network: torch.nn.Module) -> Dict[str, Any]:
    """
    Compute gradient magnitude statistics globally and per layer.

    Args:
        network: the PyTorch Module instance immediately after a backward pass

    Returns:
        dict with 'global' and 'per_layer' keys
    """
    all_grads: List[float] = []
    per_layer_stats = []

    layer_idx = 0
    for name, param in network.named_parameters():
        layer_grads: List[float] = []
        
        if param.grad is not None:
            # Flatten to 1D and convert to python floats
            grads = param.grad.flatten().tolist()
            for g in grads:
                if not (math.isnan(g) or math.isinf(g)):
                    layer_grads.append(abs(g))
                else:
                    layer_grads.append(float("nan"))
        
        # We group by parameter. In a typical nn.Linear there is weight and bias.
        # For simplicity, we consider each parameter tensor as its own "layer" 
        # or we could group by module. Let's just track each parameter matrix.

        clean_grads = [g for g in layer_grads if not math.isnan(g)]
        has_nan = len(clean_grads) < len(layer_grads)

        layer_stats = _compute_stats(clean_grads)
        layer_stats["layer"] = layer_idx
        layer_stats["name"] = name
        layer_stats["has_nan"] = has_nan
        per_layer_stats.append(layer_stats)

        all_grads.extend(layer_grads)
        layer_idx += 1

    clean_all = [g for g in all_grads if not math.isnan(g)]
    global_stats = _compute_stats(clean_all)
    global_stats["has_nan"] = len(clean_all) < len(all_grads)

    return {
        "global": global_stats,
        "per_layer": per_layer_stats,
    }


def compute_weight_stats(network: torch.nn.Module) -> Dict[str, float]:
    """
    Compute weight magnitude statistics across the entire network.
    Dangerously large weights are a precursor to gradient explosion.

    Returns:
        dict with max, min, mean
    """
    all_weights: List[float] = []

    for name, param in network.named_parameters():
        weights = param.data.flatten().tolist()
        all_weights.extend(abs(w) for w in weights)

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
