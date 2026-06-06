"""
Layer selector: evicts Tier 3 layers from past_kv before compression,
and reconstructs them via interpolation on the receiving side.

past_key_values is a tuple of (k, v) pairs — one per transformer layer.
The tuple length equals num_layers (e.g. 28 for Qwen3-4B, 40 for Qwen3-14B).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def evict_tier3(past_kv: Tuple, layer_profile: dict) -> Tuple:
    """
    Remove Tier 3 layers from past_kv entirely.
    Returns a tuple containing only Tier 1 and Tier 2 layer pairs,
    in original index order.
    """
    tier_map: dict = layer_profile["tier_map"]
    surviving = []
    for layer_idx, layer_tensors in enumerate(past_kv):
        if tier_map.get(layer_idx, 1) != 3:
            surviving.append(layer_tensors)
    return tuple(surviving)


def reconstruct_tier3(
    decompressed: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    layer_profile: dict,
    num_layers: int,
) -> Tuple:
    """
    Reconstruct the full past_key_values tuple of length num_layers.

    - Tier 1 & 2 layers: use the decompressed tensors directly.
    - Tier 3 layers: reconstruct via arithmetic mean of adjacent surviving layers.
      If no left neighbor exists, copy right neighbor; if no right, copy left.

    Returns a tuple of (k, v) pairs for all num_layers.
    """
    tier_map: dict = layer_profile["tier_map"]

    # Build ordered list of surviving layer indices
    surviving_indices = sorted(
        idx for idx, tier in tier_map.items() if tier != 3
    )

    full: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

    # Fill in surviving layers first
    for layer_idx, kv in decompressed.items():
        full[layer_idx] = kv

    # Reconstruct dropped layers
    for layer_idx in range(num_layers):
        if full[layer_idx] is not None:
            continue  # already have it

        # Find nearest surviving neighbors on each side
        left_idx = _find_nearest_left(layer_idx, surviving_indices)
        right_idx = _find_nearest_right(layer_idx, surviving_indices, num_layers)

        if left_idx is not None and right_idx is not None:
            lk, lv = full[left_idx]
            rk, rv = full[right_idx]
            k = _interpolate(lk, rk, layer_idx, left_idx, right_idx)
            v = _interpolate(lv, rv, layer_idx, left_idx, right_idx)
        elif left_idx is not None:
            k, v = full[left_idx]
        elif right_idx is not None:
            k, v = full[right_idx]
        else:
            raise RuntimeError(f"Layer {layer_idx}: no surviving neighbors to interpolate from.")

        full[layer_idx] = (k, v)

    return tuple(full)


def _interpolate(
    left: torch.Tensor,
    right: torch.Tensor,
    target_idx: int,
    left_idx: int,
    right_idx: int,
) -> torch.Tensor:
    """Linear interpolation between two tensors based on relative position."""
    span = right_idx - left_idx
    if span == 0:
        return left.clone()
    alpha = (target_idx - left_idx) / span  # 0.0 → left, 1.0 → right
    return ((1.0 - alpha) * left + alpha * right).to(left.dtype)


def _find_nearest_left(idx: int, surviving: List[int]) -> Optional[int]:
    result = None
    for s in surviving:
        if s < idx:
            result = s
        else:
            break
    return result


def _find_nearest_right(idx: int, surviving: List[int], num_layers: int) -> Optional[int]:
    for s in surviving:
        if s > idx:
            return s
    return None
