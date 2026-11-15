"""Causal online structural-break detectors + a CrunchDAO-compatible infer().

Everything here is strictly causal: the score after online observation t depends
only on the historical (reference) segment and online[:t+1] — never on future
points — so it is valid for the streaming Real-Time edition.

The unsupervised scorer combines self-resetting CUSUM statistics (which stay ~0
under the null and ratchet up after a break) for mean and variance shifts, plus a
windowed KS distance vs a historical reference sample for distribution/shape
changes. This is the strong baseline; a supervised LightGBM layer over the same
per-step features is the next step once the real 10k-series dataset is available.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def _robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(x))
    if not math.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return med, scale


def _acf1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation (shift/scale invariant)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


class OnlineBreakDetector:
    """Maintains streaming state and emits a break-confidence score in [0, 1]."""

    def __init__(
        self,
        historical: np.ndarray,
        k_mean: float = 0.5,
        k_var: float = 0.5,
        scale_stat: float = 8.0,
        ks_window: int = 64,
        ks_every: int = 10,
        ks_refsize: int = 200,
        use_acf: bool = False,
        k_acf: float = 0.5,
        w_acf: float = 0.5,
    ) -> None:
        h = np.asarray(historical, dtype=np.float64)
        h = h[np.isfinite(h)]
        if h.size == 0:
            h = np.zeros(1)
        self.mu, self.sd = _robust_center_scale(h)
        self.var_h = max(float(np.var(h)), 1e-12)

        # Reference sample for KS (deterministic, evenly spaced subsample).
        if h.size > ks_refsize:
            idx = np.linspace(0, h.size - 1, ks_refsize).astype(int)
            self.ref_sample = h[idx]
        else:
            self.ref_sample = h

        self.k_mean = k_mean
        self.k_var = k_var
        self.scale_stat = scale_stat
        self.ks_window = ks_window
        self.ks_every = ks_every
        self.use_acf = use_acf
        self.k_acf = k_acf
        self.w_acf = w_acf
        self.acf1_h = _acf1(h)  # historical lag-1 autocorrelation

        self._ref_sorted = np.sort(self.ref_sample)

        # Streaming state.
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._v_pos = 0.0
        self._v_neg = 0.0
        self._a_pos = 0.0
        self._a_neg = 0.0
        self._prev_z: float | None = None
        self._s3_pos = self._s3_neg = 0.0  # skew CUSUM on z^3
        self._s4_pos = self._s4_neg = 0.0  # kurtosis CUSUM on z^4 - 3
        self._n = 0
        self._buf: deque[float] = deque(maxlen=ks_window)
        self._ks_val = 0.0

    def update(self, x: float) -> float:
        if not math.isfinite(x):
            x = self.mu
        self._n += 1
        z = (x - self.mu) / self.sd

        # Mean-shift CUSUM (two-sided, self-resetting under the null).
        self._s_pos = max(0.0, self._s_pos + z - self.k_mean)
        self._s_neg = max(0.0, self._s_neg - z - self.k_mean)
        mean_cusum = max(self._s_pos, self._s_neg)

        # Variance-shift CUSUM on (z^2 - 1), which has zero mean under the null.
        u = z * z - 1.0
        self._v_pos = max(0.0, self._v_pos + u - self.k_var)
        self._v_neg = max(0.0, self._v_neg - u - self.k_var)
        var_cusum = max(self._v_pos, self._v_neg)

        # Skew CUSUM on z^3 and kurtosis CUSUM on z^4-3 (both zero-mean under a
        # standard-normal null; larger slack for their heavier tails). Target
        # distribution-shape breaks that the mean/variance CUSUMs miss.
        u3 = z * z * z
        self._s3_pos = max(0.0, self._s3_pos + u3 - 1.5)
        self._s3_neg = max(0.0, self._s3_neg - u3 - 1.5)
        u4 = z * z * z * z - 3.0
        self._s4_pos = max(0.0, self._s4_pos + u4 - 3.0)
        self._s4_neg = max(0.0, self._s4_neg - u4 - 3.0)

        # Windowed KS vs the historical reference (recomputed periodically).
        self._buf.append(x)
        if self._n % self.ks_every == 0 and len(self._buf) >= 8:
            w = np.asarray(self._buf)
            w_sorted = np.sort(w)
            combined = np.concatenate([w_sorted, self._ref_sorted])
            combined.sort()
            na, nb = len(w), len(self._ref_sorted)
            cdf_a = np.searchsorted(w_sorted, combined, side='right') / na
            cdf_b = np.searchsorted(self._ref_sorted, combined, side='right') / nb
            self._ks_val = float(np.max(np.abs(cdf_a - cdf_b)))

        # Autocorrelation-shift CUSUM on the lag-1 product (z_t * z_{t-1}), whose
        # null mean is the historical lag-1 autocorrelation. Catches dependence-
        # structure breaks that leave the marginal mean/variance unchanged.
        acf_cusum = 0.0
        if self.use_acf:
            if self._prev_z is not None:
                d = z * self._prev_z - self.acf1_h
                self._a_pos = max(0.0, self._a_pos + d - self.k_acf)
                self._a_neg = max(0.0, self._a_neg - d - self.k_acf)
                acf_cusum = max(self._a_pos, self._a_neg)
            self._prev_z = z

        combined = mean_cusum + 0.5 * var_cusum + self.w_acf * acf_cusum
        cusum_score = 1.0 - math.exp(-combined / self.scale_stat)
        # Soft-OR the CUSUM score with the KS distance.
        score = 1.0 - (1.0 - cusum_score) * (1.0 - self._ks_val)
        return float(min(max(score, 0.0), 1.0))


def score_series(historical: np.ndarray, online: np.ndarray, **kw) -> np.ndarray:
    """Return the per-online-step score array for one series (causal)."""
    det = OnlineBreakDetector(historical, **kw)
    out = np.empty(len(online), dtype=np.float64)
    for i, x in enumerate(online):
        out[i] = det.update(float(x))
    return out


# --- CrunchDAO Real-Time submission interface -------------------------------
# infer() is a generator: yield once to signal readiness, then for each dataset
# (historical_list, online_iterable) yield one score after consuming each online
# point. Matches the platform's baseline template. train() is a no-op for the
# unsupervised detector (the supervised LightGBM layer will populate it later).


def train(X_train, y_train, model_directory_path: str) -> None:  # noqa: ANN001
    """Unsupervised baseline needs no training. Kept for template compatibility."""
    import os

    import joblib

    joblib.dump(
        {"version": "cusum-ks-v1"}, os.path.join(model_directory_path, "model.joblib")
    )


def infer(datasets, model_directory_path: str):  # noqa: ANN001
    yield  # signal ready
    for historical, online_stream in datasets:
        det = OnlineBreakDetector(np.asarray(historical, dtype=np.float64))
        for x in online_stream:
            yield det.update(float(x))
