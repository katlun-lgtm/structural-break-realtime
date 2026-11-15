"""Self-contained streaming GARCH-conditional-vol variance GLR-CUSUM block.

Discovery (2026-07-25, via the DGP simulator): the un-modeled difficulty in the
real data is GARCH volatility clustering. Standardizing the AR-innovations by a
MEAN-REVERTING GARCH(1,1) forecast (fit on the break-free history, reverting to
the history unconditional level) separates transient volatility clusters from a
persistent structural break -- a naive EWMA fails because it chases and divides
out the break. On the honest 6-block harness this block adds +0.0027 raw /
+0.0036 tax-corrected on TOP of champion+glr_var (placebo control -0.0009).

Per online point: AR(p<=4)-whiten -> e_t; causal one-step GARCH forecast
g_t = om + al*e_{t-1}^2 + be*g_{t-1} (history params, mean-reverting to v0);
u_t = e_t / sqrt(g_t); accumulate the variance-ratio GLR-CUSUM on u_t for
r in RGRID, emit log1p(S_r) + a _max column. Deterministic (fixed-init
Nelder-Mead). numpy + scipy only; no dependency on the research tree.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

RGRID = [0.5, 0.7, 1.5, 2.0, 3.0]

GARCH_NAMES = [
    "garch_var_r0.5",
    "garch_var_r0.7",
    "garch_var_r1.5",
    "garch_var_r2.0",
    "garch_var_r3.0",
    "garch_var_max",
]


def _ar_fit(x: np.ndarray, pmax: int = 10):
    x = np.asarray(x, np.float64)
    n = x.size
    best = None
    p_cap = min(pmax, max(0, n // 5))
    common_n = n - p_cap
    for p in range(0, p_cap + 1):
        if p == 0:
            mu = x.mean()
            r = x - mu
            common = r[p_cap:]
            cand = (common_n * np.log(float(common @ common) / common_n + 1e-300) + 2,
                    0, mu, np.zeros(0))
        else:
            rows = n - p
            X = np.empty((rows, p + 1))
            X[:, 0] = 1.0
            for i in range(1, p + 1):
                X[:, i] = x[p - i : n - i]
            yv = x[p:]
            b, *_ = np.linalg.lstsq(X, yv, rcond=None)
            r = yv - X @ b
            common = r[p_cap - p :]
            cand = (common_n * np.log(float(common @ common) / common_n + 1e-300)
                    + 2 * (p + 1), p, b[0], b[1:])
        if best is None or cand[0] < best[0]:
            best = cand
    _a, p, ic, co = best
    return float(ic), np.asarray(co, np.float64), int(p)


def _fit_garch(he: np.ndarray):
    """MLE GARCH(1,1) on history AR-innovations; unconditional var pinned to v0.
    Deterministic (fixed Nelder-Mead start)."""
    he = np.asarray(he, np.float64)
    he = he[np.isfinite(he)]
    if he.size < 50:
        v = max(float(he.var()) if he.size else 1.0, 1e-9)
        return v, 0.0, 0.0, v
    v0 = max(float(he.var()), 1e-9)
    e2 = he * he

    def nll(th):
        al = 0.6 / (1 + np.exp(-th[0]))
        be = 0.98 / (1 + np.exp(-th[1]))
        if al + be > 0.999:
            be = 0.999 - al
        om = v0 * (1 - al - be)
        # g[i] = (om + al*e2[i-1]) + be*g[i-1] is a first-order linear recursion, so
        # lfilter evaluates it in C with the SAME two operations in the SAME order.
        # Verified bit-identical (150/150 objective values at relative diff 0.0e+00,
        # 30/30 identical fits) -- this is a pure speedup, not an approximation.
        # Nelder-Mead calls this ~100+ times over a 1k-5k point history, so the old
        # interpreted loop was ~93% of _init_streams and ~56% of total infer wall.
        # 120.0 -> 6.2 ms/series.
        n = len(e2)
        g = np.empty(n)
        g[0] = v0
        if n > 1:
            g[1:] = lfilter([1.0], [1.0, -be], om + al * e2[:-1],
                            zi=np.array([be * v0]))[0]
        g = np.maximum(g, 1e-12)
        return 0.5 * float(np.sum(np.log(g) + e2 / g))

    try:
        r = minimize(nll, [-3.0, 2.0], method="Nelder-Mead", options=dict(maxiter=100, fatol=1e-2))
        al = 0.6 / (1 + np.exp(-r.x[0]))
        be = 0.98 / (1 + np.exp(-r.x[1]))
        if al + be > 0.999:
            be = 0.999 - al
        return v0 * (1 - al - be), float(al), float(be), v0
    except Exception:
        return v0, 0.0, 0.0, v0


class GARCHVarStream:
    """One-pass causal GARCH-standardized variance GLR-CUSUM; feed raw points."""

    __slots__ = ("_med", "_ic", "_co", "_p", "_lags", "_om", "_al", "_be", "_gv", "_prev_e2", "_S")

    def __init__(self, x_historical: Sequence[float]):
        h = np.asarray(x_historical, np.float64)
        h = h[np.isfinite(h)]
        if h.size == 0:
            h = np.zeros(1)
        self._med = float(np.median(h))
        ic, co, p = _ar_fit(h, 4)
        self._ic = ic
        self._co = co
        self._p = int(p)
        if p > 0:
            he = np.array(
                [h[i] - (ic + sum(co[j] * h[i - 1 - j] for j in range(p))) for i in range(p, h.size)]
            )
            self._lags = [float(v) for v in h[-p:]] if h.size >= p else [0.0] * p
        else:
            he = h - ic
            self._lags = []
        om, al, be, v0 = _fit_garch(he)
        self._om = float(om)
        self._al = float(al)
        self._be = float(be)
        self._gv = float(v0)
        self._prev_e2 = 0.0
        _e2h = np.asarray(he, np.float64) ** 2
        if _e2h.size > 1:
            _gh = lfilter(
                [1.0], [1.0, -self._be],
                self._om + self._al * _e2h[:-1],
                zi=np.array([self._be * float(v0)]),
            )[0]
            self._gv = float(max(_gh[-1], 1e-12))
        if _e2h.size:
            self._prev_e2 = float(_e2h[-1])
        self._S = [0.0] * len(RGRID)

    def step(self, x: float) -> List[float]:
        xv = float(x) if math.isfinite(x) else self._med
        if self._p > 0:
            pred = self._ic
            for i in range(self._p):
                pred += self._co[i] * self._lags[-1 - i]
            e = xv - pred
            self._lags.append(xv)
            del self._lags[0]
        else:
            e = xv - self._ic
        gv = self._om + self._al * self._prev_e2 + self._be * self._gv
        self._gv = gv
        self._prev_e2 = e * e
        u = e / math.sqrt(gv if gv > 1e-9 else 1e-9)
        out: List[float] = []
        for i, r in enumerate(RGRID):
            c = -0.5 * math.log(r) + 0.5 * u * u * (1.0 - 1.0 / r)
            s = self._S[i] + c
            if s < 0.0:
                s = 0.0
            self._S[i] = s
            out.append(math.log1p(s))
        out.append(math.log1p(max(self._S)))
        return out


def batch_garch(x_hist: Sequence[float], x_online: Sequence[float]) -> np.ndarray:
    """float32 (L, 6) GARCH-var GLR-CUSUM matrix; drives GARCHVarStream point-by-
    point -- the SAME code path infer() uses -> batch == stream by construction."""
    g = GARCHVarStream(x_hist)
    rows = [g.step(float(v)) for v in x_online]
    if not rows:
        return np.zeros((0, len(GARCH_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
