"""Causal multi-scale moving-average crossing-rate features.

This is the streaming deployment of ``gen_mac_xrate_dyadic`` from the frozen
``stack_tg_mac`` discovery artifact.  The four columns compare the online
crossing rate of moving-average pairs (4,8), (8,16), (16,32), and (32,64)
against the corresponding historical rate.  All state is causal.
"""

from __future__ import annotations

from collections import deque
from typing import List, Sequence

import numpy as np


MAC_XRATE_NAMES = [
    "mac_xrate_4_8",
    "mac_xrate_8_16",
    "mac_xrate_16_32",
    "mac_xrate_32_64",
]
_WINDOWS = (4, 8, 16, 32, 64)
_PAIRS = ((4, 8), (8, 16), (16, 32), (32, 64))
_MAX_WINDOW = 64


def _historical_state(history: np.ndarray):
    """Return frozen history rates, last signs, and prefix-sum state."""
    n = len(history)
    csum = np.concatenate(([0.0], np.cumsum(history, dtype=np.float64)))
    index = np.arange(n)
    means = {}
    for window in _WINDOWS:
        lo = np.maximum(0, index - window + 1)
        count = (index - lo + 1).astype(np.float64)
        means[window] = (csum[index + 1] - csum[lo]) / count

    rates = []
    signs = []
    for fast, slow in _PAIRS:
        if n:
            sign = np.sign(means[fast] - means[slow])
            cross = np.zeros(n, dtype=np.float64)
            cross[1:] = (sign[1:] * sign[:-1] < 0.0).astype(np.float64)
            lo, hi = min(slow, n), n
            rate = float(cross[lo:hi].mean()) if hi - lo >= 8 else 0.0
            last_sign = float(sign[-1])
        else:
            rate = 0.0
            last_sign = 0.0
        rates.append(rate)
        signs.append(last_sign)

    # Prefix sums after 0..n observations.  Sixty-five entries are sufficient
    # to reproduce every rolling sum after future points are appended.
    prefixes = deque(csum[-(_MAX_WINDOW + 1) :].tolist(), maxlen=_MAX_WINDOW + 1)
    return np.asarray(rates, dtype=np.float64), np.asarray(signs, dtype=np.float64), prefixes


class MACXRateStream:
    """Streaming four-column port of the certified vectorized generator."""

    __slots__ = ("_n_total", "_t", "_total", "_prefixes", "_hist_rate", "_prev_sign", "_crosses")

    def __init__(self, x_historical: Sequence[float]):
        history = np.asarray(x_historical, dtype=np.float64).ravel()
        self._hist_rate, self._prev_sign, self._prefixes = _historical_state(history)
        self._n_total = len(history)
        self._t = 0
        self._total = float(self._prefixes[-1])
        self._crosses = np.zeros(len(_PAIRS), dtype=np.float64)

    def step(self, value: float) -> List[float]:
        x = float(value)
        self._n_total += 1
        self._t += 1
        self._total += x
        self._prefixes.append(self._total)

        means = {}
        available = len(self._prefixes) - 1
        for window in _WINDOWS:
            count = min(window, self._n_total)
            if count <= available:
                rolling_sum = self._prefixes[-1] - self._prefixes[-1 - count]
            else:
                # Only reachable when history + online contains fewer than 64
                # observations; the deque still includes the zero prefix.
                rolling_sum = self._prefixes[-1] - self._prefixes[0]
            means[window] = rolling_sum / float(count)

        result = np.empty(len(_PAIRS), dtype=np.float64)
        for column, (fast, slow) in enumerate(_PAIRS):
            sign = float(np.sign(means[fast] - means[slow]))
            if sign * self._prev_sign[column] < 0.0:
                self._crosses[column] += 1.0
            self._prev_sign[column] = sign
            online_rate = self._crosses[column] / float(self._t)
            contrast = online_rate - self._hist_rate[column] if self._t >= fast else 0.0
            result[column] = np.tanh(contrast * 6.0) * 3.0
        return result.astype(np.float32).tolist()


def batch_mac_xrate(x_historical: Sequence[float], x_online: Sequence[float]) -> np.ndarray:
    """Return a float32 ``[online_time, 4]`` matrix using streaming semantics."""
    stream = MACXRateStream(x_historical)
    rows = [stream.step(float(value)) for value in x_online]
    if not rows:
        return np.zeros((0, len(MAC_XRATE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
