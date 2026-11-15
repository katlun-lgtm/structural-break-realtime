"""Streaming vol-whitening components (deployment for the consol101 family).

EWMAVolStream : fits lambda on the historical innovations at init (quasi-ML over
                a fixed grid), then streams v_t = e_t / sigma_t with the EWMA
                variance recursion continuing causally from the historical tail.
VStats7Stream : the 7 vol-whitened statistics from the audited batch code
                (_vstats in frontier9): v_cusum_var (k=.5), v_cusum_var025,
                v_var_all, v_logvar64 (refresh 4), v_ks64 (every 8, vs the
                historical-v reference), sigma_ratio, lambda_hat.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

VOL_NAMES = [
    "v_cusum_var",
    "v_cusum_var025",
    "v_var_all",
    "v_logvar64",
    "v_ks64",
    "sigma_ratio",
    "lambda_hat",
]
_LAMBDAS = [0.85, 0.90, 0.94, 0.97, 0.99]


def _fast_ks_presorted(a, b_sorted):
    """KS statistic with b already sorted."""
    na, nb = len(a), len(b_sorted)
    a_s = np.sort(a)
    combined = np.concatenate([a_s, b_sorted])
    combined.sort()
    cdf_a = np.searchsorted(a_s, combined, side='right') / na
    cdf_b = np.searchsorted(b_sorted, combined, side='right') / nb
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ewma_fit(e_hist) -> float:
    """Vectorized EWMA lambda selection over a grid."""
    e = np.asarray(e_hist, dtype=np.float64)
    e2 = e * e
    best_lam, best_ll = 0.94, -1e18
    for lam in _LAMBDAS:
        # Compute EWMA variance series: s2[0]=1, s2[i+1] = lam*s2[i] + (1-lam)*e[i]^2
        # This is a linear recurrence; unroll via cumulative power weights
        n = len(e)
        s2 = np.empty(n, dtype=np.float64)
        s2_val = 1.0
        ll = 0.0
        for i in range(n):
            ll -= 0.5 * (math.log(s2_val + 1e-12) + e2[i] / (s2_val + 1e-12))
            s2_val = lam * s2_val + (1 - lam) * e2[i]
        if ll > best_ll:
            best_ll, best_lam = ll, lam
    return best_lam


class EWMAVolStream:
    """v_t = e_t / sigma_t; exposes v_hist and lam. Feed e_t via step()."""

    def __init__(self, e_hist: np.ndarray) -> None:
        self.lam = _ewma_fit(e_hist)
        lam = self.lam
        v_hist = np.empty_like(e_hist)
        s2 = 1.0
        for i, x in enumerate(e_hist):
            v_hist[i] = x / math.sqrt(s2 + 1e-12)
            s2 = lam * s2 + (1 - lam) * x * x
        self.v_hist = v_hist
        self._s2 = s2
        self.sigma = math.sqrt(s2 + 1e-12)

    def step(self, e_t: float) -> float:
        sd = math.sqrt(self._s2 + 1e-12)
        self.sigma = sd
        v = e_t / sd
        self._s2 = self.lam * self._s2 + (1 - self.lam) * e_t * e_t
        return v


class VStats7Stream:
    """Streaming twin of frontier9._vstats (feed v_t + the stream's sigma/lam)."""

    def __init__(self, v_hist: np.ndarray, lam: float) -> None:
        self.lam = lam
        if len(v_hist):
            idx = np.linspace(0, len(v_hist) - 1, min(len(v_hist), 250)).astype(int)
            self.ref = v_hist[idx]
        else:
            self.ref = np.zeros(1)
        self._ref_sorted = np.sort(self.ref)
        self._n = 0
        self._cp = self._cn = self._cp2 = self._cn2 = 0.0
        self._s1 = self._s2s = 0.0
        self._ks = self._logvar = 0.0
        self._buf: deque[float] = deque(maxlen=64)

    def step(self, v_t: float, sigma_t: float) -> list[float]:
        self._n += 1
        n = self._n
        x = float(v_t)
        u2 = x * x - 1.0
        self._cp = max(0.0, self._cp + u2 - 0.5)
        self._cn = max(0.0, self._cn - u2 - 0.5)
        self._cp2 = max(0.0, self._cp2 + u2 - 0.25)
        self._cn2 = max(0.0, self._cn2 - u2 - 0.25)
        self._s1 += x
        self._s2s += x * x
        var_all = (
            (self._s2s / n - (self._s1 / n) ** 2 - 1.0) * math.sqrt(n)
            if n >= 2
            else 0.0
        )
        self._buf.append(x)
        if n <= 8 or n % 4 == 0:
            w = np.asarray(self._buf)
            if len(w) >= 2:
                self._logvar = math.log(float(np.var(w)) + 1e-12)
        if n % 8 == 0 and len(self._buf) >= 8:
            w = np.asarray(self._buf)
            self._ks = _fast_ks_presorted(w, self._ref_sorted)
        return [
            max(self._cp, self._cn),
            max(self._cp2, self._cn2),
            var_all,
            self._logvar,
            self._ks,
            float(sigma_t),
            self.lam,
        ]
