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
from typing import List, Dict, Any, Optional


class GradientAccumulator:
    """
    Accumulates per-layer and global gradient statistics immediately after
    loss.backward() and before optimizer.step() / zero_grad().
    At epoch end, computes mean, max, std, and latest gradient norm for each layer.
    """

    def __init__(self, network: torch.nn.Module):
        self.network = network
        self.reset()

    def reset(self):
        # Maps param_name or layer_idx -> list of norm values across batches
        self.batch_layer_norms: Dict[int, List[float]] = {}
        self.layer_names: Dict[int, str] = {}
        self.batch_global_norms: List[float] = []
        self.latest_layer_norms: Dict[int, float] = {}
        self.latest_global_norm: float = 0.0
        self.pre_clip_global_norm: Optional[float] = None
        self.post_clip_global_norm: Optional[float] = None
        self.has_nan: bool = False

    def capture_batch(
        self,
        pre_clip_norm: Optional[float] = None,
        post_clip_norm: Optional[float] = None,
    ):
        """
        Record gradient norms for current batch immediately after backward().
        """
        layer_idx = 0
        total_sq_norm = 0.0

        for name, param in self.network.named_parameters():
            self.layer_names[layer_idx] = name
            if layer_idx not in self.batch_layer_norms:
                self.batch_layer_norms[layer_idx] = []

            if param.grad is not None:
                grad_data = param.grad.detach()
                if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                    self.has_nan = True
                    norm = float("nan")
                else:
                    norm = float(torch.linalg.norm(grad_data).item())
                    total_sq_norm += norm ** 2
                self.batch_layer_norms[layer_idx].append(norm)
                self.latest_layer_norms[layer_idx] = norm
            else:
                self.batch_layer_norms[layer_idx].append(0.0)
                self.latest_layer_norms[layer_idx] = 0.0

            layer_idx += 1

        global_norm = float(math.sqrt(total_sq_norm)) if not self.has_nan else float("nan")
        self.batch_global_norms.append(global_norm)
        self.latest_global_norm = global_norm

        if pre_clip_norm is not None:
            self.pre_clip_global_norm = pre_clip_norm
        if post_clip_norm is not None:
            self.post_clip_global_norm = post_clip_norm

    def compute_epoch_stats(self) -> Dict[str, Any]:
        """
        Produce aggregated gradient stats for the entire epoch.
        """
        per_layer = []
        for layer_idx, norms in sorted(self.batch_layer_norms.items()):
            clean = [n for n in norms if not (math.isnan(n) or math.isinf(n))]
            stats = _compute_stats(clean)
            stats["layer"] = layer_idx
            stats["name"] = self.layer_names.get(layer_idx, f"layer_{layer_idx}")
            stats["has_nan"] = len(clean) < len(norms)
            stats["latest"] = self.latest_layer_norms.get(layer_idx, 0.0)
            per_layer.append(stats)

        clean_global = [g for g in self.batch_global_norms if not (math.isnan(g) or math.isinf(g))]
        global_stats = _compute_stats(clean_global)
        global_stats["has_nan"] = self.has_nan or (len(clean_global) < len(self.batch_global_norms))
        global_stats["latest"] = self.latest_global_norm

        result = {
            "global": global_stats,
            "per_layer": per_layer,
        }
        if self.pre_clip_global_norm is not None:
            result["pre_clip_global_norm"] = self.pre_clip_global_norm
        if self.post_clip_global_norm is not None:
            result["post_clip_global_norm"] = self.post_clip_global_norm

        return result


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
