"""
Offline Calibration Profiler for LAKV compression.

Runs N samples from a dataset through the model with latent steps active,
measures per-layer attention importance, offset variance, and effective rank,
then assigns each layer to a compression tier.

Run as a script:
    python -m compression.calibrate \
        --model_name Qwen/Qwen3-8B \
        --n_samples 50 \
        --latent_steps 20 \
        --output calibration_artifacts/layer_profile_Qwen3-8B.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_model(model_name: str, device: str) -> Tuple:
    print(f"[Calibrate] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        output_attentions=False,  # we enable per-call below
    ).to(device).eval()
    return model, tokenizer


def _load_calibration_samples(n: int) -> List[str]:
    """Load n questions from the GSM8K training split."""
    ds = load_dataset("gsm8k", "main", split="train")
    questions = [row["question"] for row in ds]
    return questions[:n]


def _tokenize(text: str, tokenizer, device: str) -> dict:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return {k: v.to(device) for k, v in enc.items()}


# -----------------------------------------------------------------------
# Signal computation
# -----------------------------------------------------------------------

@torch.no_grad()
def _compute_attention_importance(model, tokenizer, samples: List[str], device: str) -> torch.Tensor:
    """
    A_l = mean L2-norm of attention weight matrices across all heads and samples.
    Returns tensor of shape [num_layers].
    """
    num_layers = model.config.num_hidden_layers
    layer_scores = torch.zeros(num_layers, device="cpu")
    count = 0

    for text in samples:
        enc = _tokenize(text, tokenizer, device)
        out = model(**enc, output_attentions=True, use_cache=False)
        # out.attentions: tuple of [B, num_heads, S, S] per layer
        for l, attn in enumerate(out.attentions):
            if l >= num_layers:
                break
            # mean entropy proxy: L2 norm of attention distribution
            layer_scores[l] += attn.float().norm(dim=-1).mean().item()
        count += 1

    layer_scores /= max(count, 1)
    return layer_scores  # [num_layers]


@torch.no_grad()
def _compute_offset_variance(
    model, tokenizer, samples: List[str], device: str, latent_steps: int
) -> torch.Tensor:
    """
    V_l = mean (1 - cosine_sim) of hidden states at layer l
          when running with empty prefix vs. with latent_steps fake tokens prepended.

    A high V_l means the layer is sensitive to the context length — i.e.,
    it would break if a misaligned KV cache were injected.

    Returns tensor of shape [num_layers].
    """
    num_layers = model.config.num_hidden_layers
    variances = torch.zeros(num_layers, device="cpu")
    count = 0

    for text in samples[:min(len(samples), 20)]:  # 20 samples is enough for variance
        enc_base = _tokenize(text, tokenizer, device)

        # Baseline: normal forward pass
        out_base = model(**enc_base, output_hidden_states=True, use_cache=False)
        hidden_base = out_base.hidden_states  # tuple of [B, S, H] per layer (0 = embed)

        # Perturbed: prepend `latent_steps` copies of a fixed "pad" token to simulate
        # the position shift introduced by passing a KV cache from a previous agent.
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        prefix = torch.full((1, latent_steps), pad_id, dtype=torch.long, device=device)
        perturbed_ids = torch.cat([prefix, enc_base["input_ids"]], dim=1)
        perturbed_mask = torch.ones_like(perturbed_ids)

        out_pert = model(
            input_ids=perturbed_ids,
            attention_mask=perturbed_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_pert = out_pert.hidden_states  # layers 0..num_layers

        # Compare last token hidden state at each layer
        for l in range(1, min(len(hidden_base), num_layers + 1)):
            h_b = hidden_base[l][:, -1, :].float()   # [B, H]
            h_p = hidden_pert[l][:, -1, :].float()   # [B, H]
            sim = F.cosine_similarity(h_b, h_p, dim=-1).mean().item()
            variances[l - 1] += (1.0 - sim)

        count += 1

    variances /= max(count, 1)
    return variances  # [num_layers]


@torch.no_grad()
def _compute_effective_rank(
    model, tokenizer, samples: List[str], device: str
) -> torch.Tensor:
    """
    R_l = ratio of singular values required to capture 90% of total variance
          in the Key matrices at layer l.

    Returns tensor of shape [num_layers] (values in [0, 1]).
    """
    num_layers = model.config.num_hidden_layers
    ranks = torch.zeros(num_layers, device="cpu")
    count = 0

    # We hook into the model to extract K matrices without modifying it
    k_buffers: List[Optional[torch.Tensor]] = [None] * num_layers

    hooks = []
    for l, layer_module in enumerate(_iter_attention_layers(model)):
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                # 'out' for self-attention is usually (attn_output, attn_weights, present_kv)
                # We want the KV cache; use use_cache=True
                pass
            return hook_fn
        # We'll collect KV via use_cache=True instead of hooks — simpler
        hooks.append(None)

    for text in samples[:min(len(samples), 20)]:
        enc = _tokenize(text, tokenizer, device)
        out = model(**enc, use_cache=True, output_hidden_states=False)
        pkv = out.past_key_values  # tuple of (k, v) per layer
        if pkv is None:
            continue
        for l, (k, v) in enumerate(pkv):
            if l >= num_layers:
                break
            # k: [B, H, S, D] — reshape to [H*S, D] and run SVD
            B, H, S, D = k.shape
            k_mat = k.float().reshape(B * H * S, D)
            try:
                sv = torch.linalg.svd(k_mat, full_matrices=False).S  # [min(rows, D)]
                cum = sv.cumsum(0) / sv.sum().clamp(min=1e-8)
                r = (cum < 0.90).sum().item() + 1  # how many SVs to reach 90%
                ranks[l] += r / max(sv.shape[0], 1)
            except Exception:
                ranks[l] += 1.0
        count += 1

    ranks /= max(count, 1)
    return ranks  # [num_layers]


def _iter_attention_layers(model):
    """Yield the self-attention sub-modules for Qwen3 / LLaMA-style models."""
    try:
        return model.model.layers  # Qwen3, LLaMA
    except AttributeError:
        pass
    try:
        return model.transformer.h  # GPT-style
    except AttributeError:
        return []


# -----------------------------------------------------------------------
# Tier assignment
# -----------------------------------------------------------------------

def _normalize(t: torch.Tensor) -> torch.Tensor:
    lo, hi = t.min(), t.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(t)
    return (t - lo) / (hi - lo)


def _assign_tiers(
    attn_importance: torch.Tensor,
    offset_variance: torch.Tensor,
    effective_rank: torch.Tensor,
    tier1_frac: float = 0.30,
    tier3_frac: float = 0.30,
) -> dict:
    """
    Compute joint score and assign tiers.
    Higher score = more important = keep at higher precision.

    Score_l = A_l_norm * (1 - V_l_norm) / (R_l_norm + eps)

    Tier 1 (top tier1_frac by score): INT8
    Tier 3 (bottom tier3_frac by score): DROPPED
    Tier 2 (middle): INT4
    """
    A = _normalize(attn_importance)
    V = _normalize(offset_variance)
    R = _normalize(effective_rank)

    scores = A * (1.0 - V) / (R + 1e-6)
    num_layers = len(scores)

    # Sort indices by score descending
    order = scores.argsort(descending=True).tolist()

    n_tier1 = max(1, math.floor(tier1_frac * num_layers))
    n_tier3 = max(1, math.floor(tier3_frac * num_layers))

    tier_map = {}
    for rank, layer_idx in enumerate(order):
        if rank < n_tier1:
            tier_map[layer_idx] = 1
        elif rank >= num_layers - n_tier3:
            tier_map[layer_idx] = 3
        else:
            tier_map[layer_idx] = 2

    return tier_map


# -----------------------------------------------------------------------
# Main calibration entry point
# -----------------------------------------------------------------------

class CalibrationProfiler:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        n_samples: int = 50,
        latent_steps: int = 20,
        tier1_frac: float = 0.30,
        tier3_frac: float = 0.30,
    ):
        self.model_name = model_name
        self.device = device
        self.n_samples = n_samples
        self.latent_steps = latent_steps
        self.tier1_frac = tier1_frac
        self.tier3_frac = tier3_frac

    def run(self, output_path: Optional[str] = None) -> dict:
        model, tokenizer = _load_model(self.model_name, self.device)
        num_layers = model.config.num_hidden_layers

        print(f"[Calibrate] Model has {num_layers} layers.")
        samples = _load_calibration_samples(self.n_samples)
        print(f"[Calibrate] Loaded {len(samples)} calibration samples.")

        print("[Calibrate] Computing attention importance ...")
        A = _compute_attention_importance(model, tokenizer, samples, self.device)

        print("[Calibrate] Computing offset variance ...")
        V = _compute_offset_variance(model, tokenizer, samples, self.device, self.latent_steps)

        print("[Calibrate] Computing effective rank ...")
        R = _compute_effective_rank(model, tokenizer, samples, self.device)

        print("[Calibrate] Assigning tiers ...")
        tier_map = _assign_tiers(A, V, R, self.tier1_frac, self.tier3_frac)

        tier_counts = {1: 0, 2: 0, 3: 0}
        for t in tier_map.values():
            tier_counts[t] += 1
        print(f"[Calibrate] Tier distribution: Tier1={tier_counts[1]}, Tier2={tier_counts[2]}, Tier3(dropped)={tier_counts[3]}")

        profile = {
            "model_name": self.model_name,
            "num_layers": num_layers,
            "tier_map": {str(k): v for k, v in tier_map.items()},  # JSON requires string keys
            "scores": {str(i): float(s) for i, s in enumerate((_normalize(A) * (1.0 - _normalize(V)) / (_normalize(R) + 1e-6)).tolist())},
            "tier1_frac": self.tier1_frac,
            "tier3_frac": self.tier3_frac,
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(profile, f, indent=2)
            print(f"[Calibrate] Saved layer profile to {output_path}")

        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return profile


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAKV offline calibration profiler")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--tier1_frac", type=float, default=0.30)
    parser.add_argument("--tier3_frac", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Path to save JSON profile")
    args = parser.parse_args()

    if args.output is None:
        safe_name = args.model_name.replace("/", "_")
        args.output = f"calibration_artifacts/layer_profile_{safe_name}.json"

    profiler = CalibrationProfiler(
        model_name=args.model_name,
        device=args.device,
        n_samples=args.n_samples,
        latent_steps=args.latent_steps,
        tier1_frac=args.tier1_frac,
        tier3_frac=args.tier3_frac,
    )
    profiler.run(output_path=args.output)
