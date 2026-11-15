"""Explicit float32 storage contract, shared by batch and streaming features.

Only finite float64 values can be saturated. NaN/infinity in raw features is an
upstream numerical error, never an imputation opportunity. No fitted state.
"""
from dataclasses import dataclass

import numpy as np

FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass
class RangeStats:
    calls: int = 0
    values: int = 0
    saturated_values: int = 0


def finite_float32(values, *, stats: RangeStats | None = None) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite raw feature before float32 conversion')
    outside = (raw > FLOAT32_MAX) | (raw < -FLOAT32_MAX)
    saturated = int(np.count_nonzero(outside))
    if stats is not None:
        stats.calls += 1
        stats.values += raw.size
        stats.saturated_values += saturated
    if saturated:
        raw = np.clip(raw, -FLOAT32_MAX, FLOAT32_MAX)
    # Preserve legacy casting, including signed zero, for all in-range inputs.
    return raw.astype(np.float32)
