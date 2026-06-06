"""
Run all method/compression configurations on a shared model and print
a comparison table.

Usage:
    python benchmark_all.py --model_name Qwen/Qwen3-8B --task gsm8k --n_samples 50

    # With a pre-built calibration profile (recommended for adaptive mode):
    python benchmark_all.py --model_name Qwen/Qwen3-8B --task gsm8k --n_samples 50 \
        --calibration_file calibration_artifacts/layer_profile_Qwen3-8B.json

    # Skip certain configs:
    python benchmark_all.py --model_name Qwen/Qwen3-8B --n_samples 20 \
        --skip baseline --skip text_mas

    # Save results to JSON:
    python benchmark_all.py --model_name Qwen/Qwen3-8B --n_samples 50 \
        --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import time
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from data import (
    load_gsm8k, load_aime2024, load_aime2025, load_arc_easy,
    load_arc_challenge, load_gpqa_diamond, load_mbppplus,
    load_humanevalplus, load_medqa,
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
from compression.metrics import MetricsCollector, activate, deactivate
from compression.pipeline import KVCompressionPipeline


# -----------------------------------------------------------------------
# Config definitions
# -----------------------------------------------------------------------

@dataclass
class RunConfig:
    label: str               # display name in table
    method: str              # baseline | text_mas | latent_mas
    latent_steps: int = 0
    compression_mode: str = "none"
    calibration_file: Optional[str] = None
    skip_key: str = ""       # matched against --skip flags


CONFIGS: List[RunConfig] = [
    RunConfig("Baseline (single agent)",      "baseline",   skip_key="baseline"),
    RunConfig("TextMAS",                       "text_mas",   skip_key="text_mas"),
    RunConfig("LatentMAS  (no compression)",  "latent_mas", latent_steps=20, compression_mode="none",         skip_key="latent_none"),
    RunConfig("LatentMAS  + uniform INT8",    "latent_mas", latent_steps=20, compression_mode="uniform_int8", skip_key="uniform_int8"),
    RunConfig("LatentMAS  + adaptive LAKV",   "latent_mas", latent_steps=20, compression_mode="adaptive",     skip_key="adaptive"),
]


# -----------------------------------------------------------------------
# Dataset loader
# -----------------------------------------------------------------------

def load_dataset(task: str, split: str, n: int) -> List[Dict]:
    loaders = {
        "gsm8k":         lambda: load_gsm8k(split=split),
        "aime2024":      lambda: load_aime2024(split="train"),
        "aime2025":      lambda: load_aime2025(split="train"),
        "gpqa":          lambda: load_gpqa_diamond(split="test"),
        "arc_easy":      lambda: load_arc_easy(split="test"),
        "arc_challenge": lambda: load_arc_challenge(split="test"),
        "mbppplus":      lambda: load_mbppplus(split="test"),
        "humanevalplus": lambda: load_humanevalplus(split="test"),
        "medqa":         lambda: load_medqa(split="test"),
    }
    if task not in loaders:
        raise ValueError(f"Unknown task: {task}")
    items = list(itertools.islice(loaders[task](), n))
    return items


# -----------------------------------------------------------------------
# Method factory
# -----------------------------------------------------------------------

def build_method(cfg: RunConfig, model: ModelWrapper, args: argparse.Namespace):
    common = dict(temperature=args.temperature, top_p=args.top_p)

    # Attach compression pipeline to args so LatentMASMethod picks it up
    args.kv_pipeline = None
    if cfg.method == "latent_mas" and cfg.compression_mode != "none":
        cal_file = cfg.calibration_file or args.calibration_file
        if cal_file and Path(cal_file).exists():
            args.kv_pipeline = KVCompressionPipeline.from_profile(
                cal_file, compression_mode=cfg.compression_mode
            )
        else:
            try:
                num_layers = model.model.config.num_hidden_layers
            except AttributeError:
                num_layers = 28
            args.kv_pipeline = KVCompressionPipeline.make_uniform_profile(
                num_layers, compression_mode=cfg.compression_mode
            )

    if cfg.method == "baseline":
        return BaselineMethod(
            model, max_new_tokens=args.max_new_tokens,
            generate_bs=1, use_vllm=False, args=args, **common,
        )
    elif cfg.method == "text_mas":
        return TextMASMethod(
            model, max_new_tokens_each=args.max_new_tokens,
            generate_bs=1, args=args, **common,
        )
    elif cfg.method == "latent_mas":
        args.latent_steps = cfg.latent_steps
        return LatentMASMethod(
            model, latent_steps=cfg.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            generate_bs=1, args=args, **common,
        )
    raise ValueError(cfg.method)


# -----------------------------------------------------------------------
# Single run
# -----------------------------------------------------------------------

def run_config(
    cfg: RunConfig,
    items: List[Dict],
    model: ModelWrapper,
    args: argparse.Namespace,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  Running: {cfg.label}")
    print(f"{'='*60}")

    collector = MetricsCollector()
    activate(collector)
    collector.start_run()

    method = build_method(cfg, model, args)

    t_start = time.perf_counter()
    all_results = []

    for item in tqdm(items, desc=cfg.label, leave=False):
        collector.start_sample()
        result = method.run_batch([item])[0]
        collector.end_sample(correct=result.get("correct", False))
        all_results.append(result)

    t_total = time.perf_counter() - t_start
    collector.end_run()
    deactivate()

    stats = collector.summarise()
    correct = sum(1 for r in all_results if r.get("correct", False))
    n = len(all_results)

    return {
        "label":               cfg.label,
        "method":              cfg.method,
        "compression_mode":    cfg.compression_mode,
        "latent_steps":        cfg.latent_steps,
        "n_samples":           n,
        "accuracy_pct":        round(correct / n * 100, 2) if n else 0,
        "total_time_s":        round(t_total, 1),
        "time_per_sample_s":   round(t_total / n, 2) if n else 0,
        "avg_kv_before_mb":    stats.get("avg_kv_before_mb", 0),
        "avg_kv_after_mb":     stats.get("avg_kv_after_mb", 0),
        "compression_ratio":   stats.get("compression_ratio", 1.0),
        "avg_judger_setup_ms": stats.get("avg_judger_setup_ms", 0),
        "peak_gpu_mb":         stats.get("peak_gpu_mb", 0),
    }


# -----------------------------------------------------------------------
# Table printer
# -----------------------------------------------------------------------

def print_table(rows: List[Dict]) -> None:
    columns = [
        ("Config",                 "label",               "<", 38),
        ("Accuracy",               "accuracy_pct",        ">",  9),
        ("Time/sample (s)",        "time_per_sample_s",   ">", 16),
        ("KV before (MB)",         "avg_kv_before_mb",    ">", 15),
        ("KV after (MB)",          "avg_kv_after_mb",     ">", 14),
        ("Ratio",                  "compression_ratio",   ">",  7),
        ("Judger setup (ms)",      "avg_judger_setup_ms", ">", 18),
        ("Peak GPU (MB)",          "peak_gpu_mb",         ">", 14),
    ]

    # Try to use tabulate for a nicer look
    try:
        from tabulate import tabulate as _tabulate
        headers = [c[0] for c in columns]
        table_rows = []
        for r in rows:
            row = []
            for _, key, _, _ in columns:
                v = r.get(key, "—")
                if key == "accuracy_pct":
                    row.append(f"{v}%")
                elif key == "compression_ratio":
                    row.append(f"{v}x" if v != 1.0 else "—")
                elif v == 0 and key in ("avg_kv_before_mb", "avg_kv_after_mb", "avg_judger_setup_ms"):
                    row.append("—")
                else:
                    row.append(v)
            table_rows.append(row)
        print("\n" + _tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))
        return
    except ImportError:
        pass

    # Fallback: manual ASCII table
    header_parts = [f"{name:{align}{width}}" for name, _, align, width in columns]
    separator = "-+-".join("-" * w for _, _, _, w in columns)
    header_line = " | ".join(header_parts)

    print("\n" + header_line)
    print(separator)
    for r in rows:
        parts = []
        for _, key, align, width in columns:
            v = r.get(key, "—")
            if key == "accuracy_pct":
                cell = f"{v}%"
            elif key == "compression_ratio":
                cell = f"{v}x" if v != 1.0 else "—"
            elif v == 0 and key in ("avg_kv_before_mb", "avg_kv_after_mb", "avg_judger_setup_ms"):
                cell = "—"
            else:
                cell = str(v)
            parts.append(f"{cell:{align}{width}}")
        print(" | ".join(parts))
    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LAKV benchmark — all configs in one run")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task", default="gsm8k",
                        choices=["gsm8k","aime2024","aime2025","gpqa",
                                 "arc_easy","arc_challenge","mbppplus","humanevalplus","medqa"])
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Number of samples to evaluate per config")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--latent_steps", type=int, default=20,
                        help="Latent steps used for all latent_mas configs")
    parser.add_argument("--calibration_file", type=str, default=None,
                        help="Path to layer_profile JSON for adaptive mode")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip config by skip_key (repeatable). "
                             "Keys: baseline, text_mas, latent_none, uniform_int8, adaptive")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full results to this JSON file")
    # Required by LatentMASMethod even when not using vLLM
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--use_second_HF_model", action="store_true", default=False)
    parser.add_argument("--latent_space_realign", action="store_true", default=False)
    parser.add_argument("--latent_only", action="store_true", default=False)
    parser.add_argument("--sequential_info_only", action="store_true", default=False)
    parser.add_argument("--think", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default="sequential")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # Set dummy latent_steps so LatentMASMethod.__init__ doesn't crash
    args.latent_steps = args.latent_steps

    set_seed(args.seed)
    device = auto_device(args.device)

    print(f"\nLoading model {args.model_name} ...")
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    print("Model loaded.")

    print(f"\nLoading {args.n_samples} samples from {args.task} ...")
    items = load_dataset(args.task, args.split, args.n_samples)
    print(f"Loaded {len(items)} samples.")

    configs_to_run = [c for c in CONFIGS if c.skip_key not in args.skip]
    # Propagate shared latent_steps into configs
    for c in configs_to_run:
        if c.method == "latent_mas":
            c.latent_steps = args.latent_steps

    all_row_results = []
    for cfg in configs_to_run:
        row = run_config(cfg, items, model, args)
        all_row_results.append(row)
        # Quick per-run summary
        print(f"  -> Accuracy: {row['accuracy_pct']}%  |  "
              f"Time/sample: {row['time_per_sample_s']}s  |  "
              f"Ratio: {row['compression_ratio']}x  |  "
              f"Peak GPU: {row['peak_gpu_mb']} MB")

    print_table(all_row_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_row_results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
