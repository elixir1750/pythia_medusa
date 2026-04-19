from __future__ import annotations

import math
import statistics
from typing import Iterable


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def summarize_numeric_series(values: Iterable[float]) -> dict[str, float]:
    series = list(values)
    if not series:
        return {
            "count": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }

    sorted_values = sorted(series)
    return {
        "count": float(len(sorted_values)),
        "mean": float(statistics.fmean(sorted_values)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "p50": float(_percentile(sorted_values, 0.50)),
        "p90": float(_percentile(sorted_values, 0.90)),
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
