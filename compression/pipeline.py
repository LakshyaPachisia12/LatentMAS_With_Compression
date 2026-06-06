"""
KVCompressionPipeline — public API for latent_mas.py.

Usage:
    pipeline = KVCompressionPipeline.from_profile("calibration_artifacts/layer_profile_Qwen3-8B.json")

    # On sender side (after generate_latent_batch):
    compressed = pipeline.compress(past_kv)

    # On receiver side (before next generate_latent_batch):
    past_kv = pipeline.decompress(compressed)

compression_mode values:
    "none"          — passthrough, no-op (original LatentMAS behaviour)
    "uniform_int8"  — all Tier 1+2 layers quantized to INT8, nothing dropped
    "adaptive"      — full LAKV: INT8/INT4/DROP per calibration profile
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from .quantizer import QuantizedLayer, quantize_kv_cache, dequantize_kv_cache
from .layer_selector import evict_tier3, reconstruct_tier3
from . import metrics as _metrics


class KVCompressionPipeline:
    def __init__(self, layer_profile: dict, compression_mode: str = "adaptive"):
        assert compression_mode in ("none", "uniform_int8", "adaptive"), \
            f"Unknown compression_mode: {compression_mode}"
        self.layer_profile = layer_profile
        self.compression_mode = compression_mode
        self.num_layers: int = layer_profile["num_layers"]

        # For uniform_int8 mode, rewrite the tier_map so everything is Tier 1
        if compression_mode == "uniform_int8":
            self.layer_profile = dict(layer_profile)
            self.layer_profile["tier_map"] = {
                idx: 1 for idx in range(self.num_layers)
            }

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_profile(cls, profile_path: Union[str, Path], compression_mode: str = "adaptive") -> "KVCompressionPipeline":
        with open(profile_path, "r") as f:
            profile = json.load(f)
        # JSON keys are strings — convert to ints
        profile["tier_map"] = {int(k): v for k, v in profile["tier_map"].items()}
        return cls(profile, compression_mode)

    @classmethod
    def make_uniform_profile(cls, num_layers: int, compression_mode: str = "adaptive") -> "KVCompressionPipeline":
        """Fallback: if no calibration file exists, assign tiers by fixed percentile of layer index."""
        tier_map = {}
        for idx in range(num_layers):
            frac = idx / max(num_layers - 1, 1)
            if frac >= 0.70:          # top 30% by score → but we have no scores, use last layers
                tier_map[idx] = 3     # drop last 30%
            elif frac >= 0.30:        # mid 40%
                tier_map[idx] = 2     # INT4
            else:
                tier_map[idx] = 1     # INT8
        profile = {"num_layers": num_layers, "tier_map": tier_map}
        return cls(profile, compression_mode)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compress(self, past_kv: Optional[Tuple]) -> Optional[Union[Tuple, List[QuantizedLayer]]]:
        """
        Compress past_kv for inter-agent transfer.
        Returns the same object untouched when compression_mode == "none".
        """
        if self.compression_mode == "none" or past_kv is None:
            return past_kv

        # Normalise to plain tuple of (k, v) pairs
        past_kv = _to_legacy_tuple(past_kv)

        # Record raw size before compression
        _metrics.record_kv_before(self.log_size(past_kv))

        # Step 1: drop Tier 3 layers
        selected = evict_tier3(past_kv, self.layer_profile)

        # Step 2: quantize surviving layers
        compressed = quantize_kv_cache(
            selected,
            self._surviving_profile(past_kv),
        )

        # Record compressed size
        after_mb = sum(
            e.k_quant.element_size() * e.k_quant.nelement() +
            e.v_quant.element_size() * e.v_quant.nelement()
            for e in compressed
        ) / (1024 ** 2)
        _metrics.record_kv_after(after_mb)

        return compressed

    def decompress(self, compressed: Optional[Union[Tuple, List[QuantizedLayer]]]) -> Optional[Tuple]:
        """
        Decompress back to a full past_key_values tuple ready for the model.
        Returns the same object untouched when compression_mode == "none".
        """
        if self.compression_mode == "none" or compressed is None:
            return compressed

        if not isinstance(compressed, list):
            # Already a plain tuple (shouldn't happen, but be safe)
            return compressed

        # Step 1: dequantize → {layer_idx: (k, v)}
        decompressed_dict = dequantize_kv_cache(compressed)

        # Step 2: reconstruct Tier 3 layers via interpolation
        full_tuple = reconstruct_tier3(
            decompressed_dict,
            self.layer_profile,
            self.num_layers,
        )
        return full_tuple

    def log_size(self, past_kv: Optional[Tuple], label: str = "") -> float:
        """Return and print the size in MB of a past_kv object."""
        if past_kv is None:
            return 0.0
        past_kv = _to_legacy_tuple(past_kv)
        total_bytes = sum(
            t.element_size() * t.nelement()
            for k, v in past_kv
            for t in (k, v)
        )
        mb = total_bytes / (1024 ** 2)
        if label:
            print(f"[LAKV] {label}: {mb:.1f} MB")
        return mb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _surviving_profile(self, full_past_kv: Tuple) -> dict:
        """
        Build a layer_profile that re-indexes surviving layers sequentially,
        because evict_tier3 removed the Tier 3 entries and the quantizer
        iterates over whatever tuple it receives.
        """
        tier_map = self.layer_profile["tier_map"]
        surviving_tiers = {
            new_idx: tier_map[orig_idx]
            for new_idx, orig_idx in enumerate(
                idx for idx in range(len(full_past_kv))
                if tier_map.get(idx, 1) != 3
            )
        }
        return {"num_layers": len(surviving_tiers), "tier_map": surviving_tiers}


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _to_legacy_tuple(past_kv) -> Tuple:
    """Convert HuggingFace Cache objects to plain tuple-of-tuples."""
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kv, Cache):
            return past_kv.to_legacy_cache()
    except ImportError:
        pass
    return past_kv
