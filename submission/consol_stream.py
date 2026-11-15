"""Streaming components for the consolidated model (deployment).

InnovStream : per-step innovation producer (PIT via historical ECDF -> AR(p)
              whitening fitted on historical -> e_t), exposing e_hist for use
              as the 'historical' of a second RichFeaturizerV4 view.
Deep8Stream : the 8 winning calibrated innovation statistics from the deep bank
              (exact formulas/constants from the audited batch implementation):
              e_var_all, e_mean_all, e_cusum_var_k025, e_cusum_kurt,
              e_cusum_skew, e_maxabs64, e_absmean64, e_tailfrac64.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.stats import norm

DEEP8_NAMES = [
    "e_var_all",
    "e_mean_all",
    "e_cusum_var_k025",
    "e_cusum_kurt",
    "e_cusum_skew",
    "e_maxabs64",
    "e_absmean64",
    "e_tailfrac64",
]

_WIN = 64
_REFRESH = 4
_K_VAR_025 = 0.25
_K_SKEW = 1.0
_K_KURT = 2.0
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_TAIL_25 = 2.0 * (1.0 - norm.cdf(2.5))

# Precomputed norm.ppf lookup table for PIT transform (avoids per-call scipy overhead)
_PPF_N = 65536
_ppf_u = np.linspace(0.0001, 0.9999, _PPF_N)
_ppf_v = norm.ppf(_ppf_u).astype(np.float64)


def _ppf_interp(u):
    """Fast norm.ppf via linear interpolation on precomputed table."""
    idx_f = np.clip(u, 0.0001, 0.9999) * (_PPF_N - 1) / 0.9998 - 0.0001 * (_PPF_N - 1) / 0.9998
    # Simpler: map u in [0.0001, 0.9999] to [0, _PPF_N-1]
    idx_f = (np.clip(u, 0.0001, 0.9999) - 0.0001) / (0.9999 - 0.0001) * (_PPF_N - 1)
    idx_lo = np.clip(idx_f.astype(np.int64), 0, _PPF_N - 2)
    frac = idx_f - idx_lo
    return _ppf_v[idx_lo] * (1.0 - frac) + _ppf_v[idx_lo + 1] * frac


def _ppf_scalar(u):
    """Fast norm.ppf for a single scalar."""
    u = max(0.0001, min(0.9999, u))
    idx_f = (u - 0.0001) / (0.9999 - 0.0001) * (_PPF_N - 1)
    idx_lo = min(int(idx_f), _PPF_N - 2)
    frac = idx_f - idx_lo
    return _ppf_v[idx_lo] * (1.0 - frac) + _ppf_v[idx_lo + 1] * frac


def _fit_ar(z: np.ndarray, pmax: int = 10):
    n = z.size
    rss0 = float(np.dot(z, z))
    best_aic = n * math.log(rss0 / n + 1e-12)
    best_p, best_phi, best_resid = 0, np.zeros(0), z.copy()
    for p in range(1, pmax + 1):
        if n <= p + 5:
            break
        Y = z[p:]
        X = np.column_stack([z[p - 1 - i : n - 1 - i] for i in range(p)])
        try:
            phi, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if np.max(np.abs(phi)) > 0.98:
            continue
        resid = Y - X @ phi
        neff = resid.size
        rss = float(np.dot(resid, resid))
        aic = neff * math.log(rss / neff + 1e-12) + 2.0 * p
        if aic < best_aic:
            best_aic, best_p, best_phi, best_resid = aic, p, phi, resid
    sigma = float(best_resid.std())
    if not math.isfinite(sigma) or sigma < 1e-9:
        sigma = 1.0
    return best_p, best_phi, sigma, best_resid / sigma


class InnovStream:
    """Per-step innovation e_t; exposes e_hist (standardized historical residuals)."""

    def __init__(self, historical) -> None:
        h = np.asarray(historical, dtype=np.float64)
        h = h[np.isfinite(h)]
        if h.size < 8:
            h = np.zeros(8)
        self._med = float(np.median(h))
        self._nh = h.size
        self._sh = np.sort(h)
        self._u_lo = 1.0 / (self._nh + 1)
        self._u_hi = self._nh / (self._nh + 1)
        self._u_scale = 1.0 / (self._nh + 1)
        zh = self._pit_batch(h)
        self.p, self.phi, self.sigma, self.e_hist = _fit_ar(zh, pmax=10)
        self._inv_sigma = 1.0 / self.sigma
        if self.p > 0:
            seed = zh[-self.p :] if zh.size >= self.p else np.zeros(self.p)
            self._lags = deque(seed, maxlen=self.p)
            self._phi = [float(phi_i) for phi_i in self.phi]
        else:
            self._lags = deque()
            self._phi = []

    def _pit_batch(self, x: np.ndarray) -> np.ndarray:
        """PIT for arrays (init-time) — uses fast lookup table."""
        u = np.searchsorted(self._sh, x, side="right") * self._u_scale
        u = np.clip(u, self._u_lo, self._u_hi)
        return _ppf_interp(u)

    def step(self, x) -> float:
        xv = float(x) if math.isfinite(x) else self._med
        # Scalar PIT: searchsorted + fast ppf lookup
        u = float(np.searchsorted(self._sh, xv, side="right")) * self._u_scale
        u = max(self._u_lo, min(self._u_hi, u))
        z = float(_ppf_scalar(u))
        if self.p > 0:
            pred = 0.0
            lags = self._lags
            phi = self._phi
            for i in range(self.p):
                pred += phi[i] * lags[-1 - i]
            resid = z - pred
            lags.append(z)
        else:
            resid = z
        return resid * self._inv_sigma


class Deep8Stream:
    """Streaming winners of the deep innovation bank (feed e_t per step)."""

    def __init__(self) -> None:
        self._n = 0
        self._rmean = 0.0
        self._rM2 = 0.0
        self._sv_pos = self._sv_neg = 0.0
        self._ku_pos = self._ku_neg = 0.0
        self._sk_pos = self._sk_neg = 0.0
        self._buf: deque[float] = deque(maxlen=_WIN)
        self._maxabs64 = self._absmean64 = self._tailfrac64 = 0.0

    def step(self, et: float) -> list[float]:
        self._n += 1
        n = self._n
        et2 = et * et
        u2 = et2 - 1.0
        u3 = et2 * et
        u4 = et2 * et2 - 3.0
        self._sv_pos = max(0.0, self._sv_pos + u2 - _K_VAR_025)
        self._sv_neg = max(0.0, self._sv_neg - u2 - _K_VAR_025)
        self._sk_pos = max(0.0, self._sk_pos + u3 - _K_SKEW)
        self._sk_neg = max(0.0, self._sk_neg - u3 - _K_SKEW)
        self._ku_pos = max(0.0, self._ku_pos + u4 - _K_KURT)
        self._ku_neg = max(0.0, self._ku_neg - u4 - _K_KURT)

        delta = et - self._rmean
        self._rmean += delta / n
        self._rM2 += delta * (et - self._rmean)
        mean_all = self._rmean * math.sqrt(n)
        var_all = (self._rM2 / (n - 1) - 1.0) * math.sqrt(n) if n >= 2 else 0.0

        self._buf.append(et)
        if n <= 8 or n % _REFRESH == 0:
            w = np.asarray(self._buf)
            if w.size >= 2:
                aw = np.abs(w)
                self._absmean64 = float(aw.mean()) - _SQRT_2_OVER_PI
                self._tailfrac64 = float(np.mean(aw > 2.5)) - _TAIL_25
                self._maxabs64 = float(aw.max())
        return [
            var_all,
            mean_all,
            max(self._sv_pos, self._sv_neg),
            max(self._ku_pos, self._ku_neg),
            max(self._sk_pos, self._sk_neg),
            self._maxabs64,
            self._absmean64,
            self._tailfrac64,
        ]
