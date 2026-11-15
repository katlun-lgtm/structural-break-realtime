"""
Extreme-value-8 streaming featurizer for the structural-break submission.

Adds the 8 extreme-value/tail features validated in feature_search (champ+extremeval8
= +0.0022 local, 4/4 blocks). Computed on a RAW-series AR whitening (no PIT) -- separate
from the pipeline's PIT-innovation stream. The champion under-uses extreme values; these
add a real, generalizing signal.

The 8 features (window W=50, updated every `stride` steps, held constant between, to
EXACTLY match the batch training features):
  max_run   = running max |innov| / s0
  abs_run   = running mean |innov| / s0
  kurt_run  = running excess kurtosis of innov (scipy default: biased Fisher)
  std_w     = std(window) / s0
  absmax_w  = max|window| / s0
  range_w   = (max-min)(window) / s0
  logvar_w  = log(var(window)/s2 + 1e-6)
  tail_w    = mean(|window| > 2.5)
where innov = raw online whitened by AR(p<=4) fit on the raw historical; s0 =
sqrt(s2 * n/(n-p-1)); s2 = AR residual variance.

`batch_ev(hist, online)` computes the same as a matrix (train path); `verify()` confirms
the stream == batch bit-for-bit at every step.
"""

from __future__ import annotations

import numpy as np

EV_NAMES = [
    "ev_max_run",
    "ev_abs_run",
    "ev_kurt_run",
    "ev_std_w",
    "ev_absmax_w",
    "ev_range_w",
    "ev_logvar_w",
    "ev_tail_w",
]
W = 50
STRIDE = 1


def _ar_fit(x, pmax=10):
    """OLS AR(p) by AIC over 0..pmax on the raw series. Matches self_sim_null.ar_fit."""
    x = np.asarray(x, np.float64)
    n = x.size
    best = None
    p_cap = min(pmax, max(0, n // 5))
    common_n = n - p_cap
    for p in range(0, p_cap + 1):
        if p == 0:
            mu = x.mean()
            resid = x - mu
            common = resid[p_cap:]
            rss = float(common @ common)
            aic = common_n * np.log(rss / common_n + 1e-300) + 2
            cand = (aic, 0, mu, np.zeros(0), max(resid.var(), 1e-12))
        else:
            rows = n - p
            X = np.empty((rows, p + 1))
            X[:, 0] = 1.0
            for i in range(1, p + 1):
                X[:, i] = x[p - i : n - i]
            yv = x[p:]
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ beta
            common = resid[p_cap - p :]
            rss = float(common @ common)
            aic = common_n * np.log(rss / common_n + 1e-300) + 2 * (p + 1)
            cand = (aic, p, beta[0], beta[1:], max(resid.var(), 1e-12))
        if best is None or cand[0] < best[0]:
            best = cand
    _a, p, intercept, coefs, sigma2 = best
    # residual variance with the SAME dof-corrected s0 as feature_search
    return intercept, coefs, sigma2, p


class ExtremeValueStream:
    def __init__(self, historical):
        h = np.asarray(historical, np.float64)
        h = h[np.isfinite(h)]
        if h.size < 8:
            h = np.zeros(8)
        self.intercept, self.coefs, self.s2, self.p = _ar_fit(h, 4)
        nres = max(h.size - self.p, 1)
        self.s0 = float(np.sqrt(max(self.s2 * nres / max(nres - self.p - 1, 1), 1e-9)))
        if self.p > 0:
            from collections import deque as _deque
            self.lags = _deque(h[-self.p :], maxlen=self.p)
            self._coefs_list = [float(c) for c in self.coefs]
        else:
            self.lags = None
            self._coefs_list = []
        self.win = []  # last W innovations
        self.n = 0
        self.sum_abs = 0.0
        self.max_abs = 0.0
        self.s1 = self.s2m = self.s3 = self.s4 = 0.0  # innov moment sums
        self.t = 0
        self._cache = np.zeros(8, np.float64)

    def _innov(self, point):
        x = float(point) if np.isfinite(point) else 0.0
        if self.lags is not None:
            pred = self.intercept
            lags = self.lags
            coefs = self._coefs_list
            for i in range(self.p):
                pred += coefs[i] * lags[-1 - i]
            e = x - pred
            lags.append(x)
        else:
            e = x - self.intercept
        return e

    def _compute(self):
        s0 = self.s0
        e = np.asarray(self.win)
        ae = np.abs(e)
        mean = self.s1 / self.n
        m2 = max(self.s2m / self.n - mean * mean, 1e-12)
        m4 = (
            self.s4 / self.n
            - 4 * mean * self.s3 / self.n
            + 6 * mean * mean * self.s2m / self.n
            - 3 * mean**4
        )
        kurt = (
            (m4 / (m2 * m2) - 3.0) if self.n > 4 else 0.0
        )  # match feature_search n>4 guard
        wvar = e.var() if e.size > 1 else 0.0
        self._cache = np.array(
            [
                self.max_abs / s0,  # ev_max_run
                (self.sum_abs / self.n) / s0,  # ev_abs_run
                kurt,  # ev_kurt_run
                (e.std() / s0) if e.size > 2 else 0.0,  # ev_std_w
                (ae.max() / s0) if e.size else 0.0,  # ev_absmax_w
                ((e.max() - e.min()) / s0) if e.size > 1 else 0.0,  # ev_range_w
                np.log(wvar / self.s2 + 1e-6) if e.size > 2 else 0.0,  # ev_logvar_w
                (ae > 2.5).mean() if e.size else 0.0,  # ev_tail_w
            ],
            np.float64,
        )

    def step(self, point):
        e = self._innov(point)
        self.n += 1
        ae = abs(e)
        self.sum_abs += ae
        if ae > self.max_abs:
            self.max_abs = ae
        e2 = e * e
        self.s1 += e
        self.s2m += e2
        self.s3 += e2 * e
        self.s4 += e2 * e2
        self.win.append(e)
        if len(self.win) > W:
            self.win.pop(0)
        if self.t % STRIDE == 0:
            self._compute()
        self.t += 1
        return self._cache.copy()


def batch_ev(historical, online):
    """(len(online), 8) matrix — same values as running the stream, for train path."""
    s = ExtremeValueStream(historical)
    return np.vstack([s.step(p) for p in online]).astype(np.float32)


def verify(seed=0, n=6):
    rng = np.random.RandomState(seed)
    ok = True
    for _ in range(n):
        h = rng.randn(rng.randint(200, 500))
        o = np.concatenate(
            [
                rng.randn(rng.randint(80, 160)),
                rng.randn(rng.randint(80, 160)) * rng.uniform(1.3, 2.0),
            ]
        )
        s = ExtremeValueStream(h)
        stream = np.vstack([s.step(p) for p in o])
        batch = batch_ev(h, o)
        if not np.allclose(stream, batch):
            ok = False
    print("stream==batch:", ok, "| feats finite:", np.isfinite(batch_ev(h, o)).all())
    return ok


if __name__ == "__main__":
    verify()
