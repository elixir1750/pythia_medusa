from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

from pythia_medusa.eval.metrics import summarize_numeric_series


@contextmanager
def timed_section():
    start = time.perf_counter()
    payload = {"elapsed_sec": 0.0}
    try:
        yield payload
    finally:
        payload["elapsed_sec"] = time.perf_counter() - start


def measure_repeated(
    fn: Callable[[], dict],
    *,
    repeat: int,
    warmup: int = 0,
) -> list[dict]:
    for _ in range(max(warmup, 0)):
        fn()
    return [fn() for _ in range(max(repeat, 0))]


def summarize_benchmark_runs(runs: list[dict], *, latency_key: str, throughput_key: str) -> dict:
    latency_values = [float(run[latency_key]) for run in runs]
    throughput_values = [float(run[throughput_key]) for run in runs]
    return {
        "avg_latency_sec": summarize_numeric_series(latency_values)["mean"],
        "avg_tokens_per_sec": summarize_numeric_series(throughput_values)["mean"],
        "latency_stats": summarize_numeric_series(latency_values),
        "throughput_stats": summarize_numeric_series(throughput_values),
        "num_runs": len(runs),
    }
