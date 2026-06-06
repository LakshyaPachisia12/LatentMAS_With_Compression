"""
Per-head asymmetric min-max quantization for KV cache tensors.

Tier 1 layers → INT8  (b=8, range [0, 255])
Tier 2 layers → INT4  (b=4, range [0, 15])

Each K/V tensor has shape [batch, num_kv_heads, seq_len, head_dim].
We compute (min, scale) per head so outlier activations in one head
don't corrupt the quantization range of the others.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class QuantizedLayer:
    layer_idx: int
    tier: int  # 1 or 2
    bits: int  # 8 or 4
    # Keys
    k_quant: torch.Tensor   # uint8, shape [B, H, S, D]  (4-bit packed for tier-2)
    k_min: torch.Tensor     # float16, shape [B, H, 1, 1]
    k_scale: torch.Tensor   # float16, shape [B, H, 1, 1]
    # Values
    v_quant: torch.Tensor
    v_min: torch.Tensor
    v_scale: torch.Tensor


def _quantize_tensor(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize x (shape [..., H, S, D]) per head (dim -3).
    Returns (x_quant uint8, min_val float16, scale float16).
    For 4-bit: two values are packed into one uint8 byte.
    """
    max_int = (1 << bits) - 1  # 255 for INT8, 15 for INT4

    # Reduce over seq_len and head_dim, keep head axis
    # x: [B, H, S, D]
    x_min = x.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values  # [B, H, 1, 1]
    x_max = x.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values  # [B, H, 1, 1]

    scale = (x_max - x_min).clamp(min=1e-8) / max_int  # [B, H, 1, 1]
    x_quant_f = ((x - x_min) / scale).round().clamp(0, max_int)

    if bits == 8:
        x_quant = x_quant_f.to(torch.uint8)
    else:
        # Pack two 4-bit values into one uint8 byte along the last dim.
        # Pad last dim to even length if needed.
        D = x_quant_f.shape[-1]
        if D % 2 != 0:
            pad = torch.zeros(*x_quant_f.shape[:-1], 1, device=x_quant_f.device, dtype=x_quant_f.dtype)
            x_quant_f = torch.cat([x_quant_f, pad], dim=-1)
        x_uint8 = x_quant_f.to(torch.uint8)
        lo = x_uint8[..., 0::2]  # even indices
        hi = x_uint8[..., 1::2]  # odd indices
        x_quant = (lo | (hi << 4))  # pack

    return x_quant, x_min.to(torch.float16), scale.to(torch.float16)


def _dequantize_tensor(
    x_quant: torch.Tensor,
    x_min: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    original_last_dim: int,
) -> torch.Tensor:
    """Inverse of _quantize_tensor. Returns bfloat16 tensor."""
    if bits == 8:
        x_f = x_quant.to(torch.float32)
    else:
        # Unpack 4-bit values
        lo = (x_quant & 0x0F).to(torch.float32)
        hi = ((x_quant >> 4) & 0x0F).to(torch.float32)
        # Interleave: lo goes to even positions, hi to odd
        B, H, S, D_packed = lo.shape
        x_f = torch.empty(B, H, S, D_packed * 2, dtype=torch.float32, device=x_quant.device)
        x_f[..., 0::2] = lo
        x_f[..., 1::2] = hi
        # Trim any padding we added during quantization
        x_f = x_f[..., :original_last_dim]

    x_min_f = x_min.to(torch.float32)
    scale_f = scale.to(torch.float32)
    return (x_f * scale_f + x_min_f).to(torch.bfloat16)


def quantize_kv_cache(
    past_kv: Tuple,
    layer_profile: dict,
) -> List[QuantizedLayer]:
    """
    Compress a past_key_values tuple.
    Only Tier 1 and Tier 2 layers are included (Tier 3 already dropped by layer_selector).

    past_kv: tuple of (k, v) per layer — each k/v is [B, H, S, D] in bfloat16.
    layer_profile: dict with 'tier_map' {layer_idx: tier} for all layers.
    Returns list of QuantizedLayer (one per surviving layer).
    """
    tier_map: dict = layer_profile["tier_map"]
    compressed: List[QuantizedLayer] = []

    for layer_idx, (k, v) in enumerate(past_kv):
        tier = tier_map.get(layer_idx, 1)
        if tier == 3:
            # Tier 3 should already be dropped, skip defensively
            continue
        bits = 8 if tier == 1 else 4
        original_d = k.shape[-1]

        k_quant, k_min, k_scale = _quantize_tensor(k, bits)
        v_quant, v_min, v_scale = _quantize_tensor(v, bits)

        compressed.append(QuantizedLayer(
            layer_idx=layer_idx,
            tier=tier,
            bits=bits,
            k_quant=k_quant, k_min=k_min, k_scale=k_scale,
            v_quant=v_quant, v_min=v_min, v_scale=v_scale,
        ))

    # Store original head_dim in each entry for safe unpacking
    for entry in compressed:
        entry._original_d = past_kv[entry.layer_idx][0].shape[-1]

    return compressed


def dequantize_kv_cache(compressed: List[QuantizedLayer]) -> dict:
    """
    Decompress a list of QuantizedLayer back to bfloat16 tensors.
    Returns dict: {layer_idx: (k_bf16, v_bf16)}.
    """
    result = {}
    for entry in compressed:
        original_d = getattr(entry, "_original_d", entry.k_quant.shape[-1])
        k = _dequantize_tensor(entry.k_quant, entry.k_min, entry.k_scale, entry.bits, original_d)
        v = _dequantize_tensor(entry.v_quant, entry.v_min, entry.v_scale, entry.bits, original_d)
        result[entry.layer_idx] = (k, v)
    return result
