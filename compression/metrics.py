"""
Lightweight metrics collector for LAKV benchmarking.

A single global collector instance is activated at the start of each
benchmark run and deactivated at the end. Both the compression pipeline
and latent_mas.py report into it via module-level calls that are no-ops
when no collector is active.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch


# -----------------------------------------------------------------------
# Per-sample record
# -----------------------------------------------------------------------

@dataclass
class SampleMetrics:
    # KV cache sizes at each inter-agent hop (one entry per non-judger agent)
    kv_mb_before: List[float] = field(default_factory=list)  # before compress
    kv_mb_after:  List[float] = field(default_factory=list)  # after compress / before decompress
    # Time spent in judger decompression + KV injection setup (ms)
    judger_setup_ms: float = 0.0
    # Whether this sample was answered correctly
    correct: Optional[bool] = None


# -----------------------------------------------------------------------
# Collector — one per benchmark run
# -----------------------------------------------------------------------

class MetricsCollector:
    def __init__(self) -> None:
        self.samples: List[SampleMetrics] = []
        self._current: Optional[SampleMetrics] = None
        self._judger_t0: Optional[float] = None
        self.peak_gpu_mb: float = 0.0
        self._run_t0: float = 0.0

    # -- lifecycle --------------------------------------------------------

    def start_run(self) -> None:
        self.samples.clear()
        self.peak_gpu_mb = 0.0
        self._run_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_run(self) -> None:
        if torch.cuda.is_available():
            self.peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    def start_sample(self) -> None:
        self._current = SampleMetrics()

    def end_sample(self, correct: bool) -> None:
        if self._current is not None:
            self._current.correct = correct
            self.samples.append(self._current)
        self._current = None

    # -- KV size hooks (called from pipeline.py) --------------------------

    def record_kv_before(self, mb: float) -> None:
        if self._current is not None:
            self._current.kv_mb_before.append(mb)

    def record_kv_after(self, mb: float) -> None:
        if self._current is not None:
            self._current.kv_mb_after.append(mb)

    # -- judger setup timing (called from latent_mas.py) ------------------

    def start_judger_setup(self) -> None:
        self._judger_t0 = time.perf_counter()

    def end_judger_setup(self) -> None:
        if self._judger_t0 is not None and self._current is not None:
            self._current.judger_setup_ms = (time.perf_counter() - self._judger_t0) * 1000
        self._judger_t0 = None

    # -- aggregated stats -------------------------------------------------

    def summarise(self) -> dict:
        n = len(self.samples)
        if n == 0:
            return {}

        accuracy = sum(1 for s in self.samples if s.correct) / n * 100

        all_before = [v for s in self.samples for v in s.kv_mb_before]
        all_after  = [v for s in self.samples for v in s.kv_mb_after]
        setup_times = [s.judger_setup_ms for s in self.samples]

        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        avg_before = mean(all_before)
        avg_after  = mean(all_after)
        ratio = avg_before / avg_after if avg_after > 0 else 1.0

        return {
            "n_samples":        n,
            "accuracy_pct":     round(accuracy, 2),
            "avg_kv_before_mb": round(avg_before, 2),
            "avg_kv_after_mb":  round(avg_after, 2),
            "compression_ratio": round(ratio, 2),
            "avg_judger_setup_ms": round(mean(setup_times), 1),
            "peak_gpu_mb":      round(self.peak_gpu_mb, 1),
        }


# -----------------------------------------------------------------------
# Global singleton — module-level no-op API
# -----------------------------------------------------------------------

_active: Optional[MetricsCollector] = None


def activate(collector: MetricsCollector) -> None:
    global _active
    _active = collector


def deactivate() -> None:
    global _active
    _active = None


def record_kv_before(mb: float) -> None:
    if _active is not None:
        _active.record_kv_before(mb)


def record_kv_after(mb: float) -> None:
    if _active is not None:
        _active.record_kv_after(mb)


def start_judger_setup() -> None:
    if _active is not None:
        _active.start_judger_setup()


def end_judger_setup() -> None:
    if _active is not None:
        _active.end_judger_setup()
