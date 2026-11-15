"""Conditional-volatility standardizer for the innovation stream feeding `iv`.

WHY. `garch_stream` computes u_t = e_t/sqrt(g_t) INTERNALLY and emits 6 CUSUM columns
from it — but the ~43+43 `iv` features were built from the plain AR innovations e_t,
so most of the model saw volatility-contaminated input. Post-pipeline diagnostic on
466 real series: after PIT + AR(<=10) + GARCH, the largest remaining structure is
still ARCH (acf1(u^2) > 0.1 in 17.8%), essentially unchanged from the forensics'
pre-garch 17%. Leverage is ABSENT (median +0.0006); skew (1.5%) and kurtosis (3.4%)
are already handled by the PIT. So conditional volatility is the one live nuisance and
it was being removed in one block rather than from the representation.

Measured on the honest 6-block harness (cache rebuild + `run_gstd_compare.py`):
champion @ p10 = 0.62353 -> **gstd = 0.62595, +0.00242, 4/6 blocks**. Same column
count, zero added tax. Pre-registered in PREREG_gstd_2026-07-26.md.

MEAN-REVERTING, fit on BREAK-FREE history — never adaptive. An adaptive estimator
chases the break and divides it out, which is exactly how the original GARCH lead was
falsely refuted at -0.038. Verified: a 2.5x variance break survives standardization as
a ~2.07x variance ratio while the null stays near 1.

Deterministic. numpy + the shared _fit_garch only.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# buildH SCOPE: gstd keeps the LEGACY garch fitter (its standardization was
# trained around it; the certified footprint fixes garch_stream ONLY).



class GStdStream:
    """Standardize AR-innovations by a causal one-step conditional-variance forecast.

    __init__ fits GARCH(1,1) on the historical innovations and runs the historical
    pass (exposing `uh`, the standardized history the featurizer uses as its frozen
    reference). step() continues the SAME recurrence over online innovations, so the
    online state carries on from where history ended — batch and stream agree by
    construction.
    """

    __slots__ = ("_om", "_al", "_be", "_g", "_prev", "uh", "_ok")

    def __init__(self, e_hist: Sequence[float]):
        eh = np.asarray(e_hist, dtype=np.float64)
        eh = eh[np.isfinite(eh)]
        self._ok = eh.size >= 50
        if not self._ok:
            self.uh = eh
            self._om = self._al = self._be = 0.0
            self._g = 1.0
            self._prev = 1.0
            return
        om, al, be, v0 = _fit_garch_legacy(eh)
        self._om, self._al, self._be = float(om), float(al), float(be)
        g = float(v0)
        prev = float(np.mean(eh * eh))
        uh = np.empty(eh.size, dtype=np.float64)
        for t in range(eh.size):
            g = self._om + self._al * prev + self._be * g
            uh[t] = eh[t] / math.sqrt(max(g, 1e-12))
            prev = eh[t] * eh[t]
        if not np.isfinite(uh).all():
            self._ok = False
            self.uh = eh
            self._g, self._prev = 1.0, 1.0
            return
        self.uh = uh
        self._g, self._prev = g, prev

    def step(self, e: float) -> float:
        ev = float(e)
        if not math.isfinite(ev):
            ev = 0.0
        if not self._ok:
            return ev
        self._g = self._om + self._al * self._prev + self._be * self._g
        u = ev / math.sqrt(max(self._g, 1e-12))
        self._prev = ev * ev
        return u if math.isfinite(u) else 0.0


def batch_gstd(e_hist: Sequence[float], e_online: Sequence[float]):
    """(uh, u) for the batch path. Drives GStdStream point-by-point — the same code
    path infer() uses — so batch and stream are bit-identical by construction."""
    s = GStdStream(e_hist)
    u = np.asarray(
        [s.step(float(v)) for v in np.asarray(e_online, dtype=np.float64)],
        dtype=np.float64,
    )
    return s.uh, u


from scipy.signal import lfilter
from scipy.optimize import minimize

def _fit_garch_legacy(he: np.ndarray):
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
        r = minimize(nll, [0.0, 2.0], method="Nelder-Mead", options=dict(maxiter=100, fatol=1e-2))
        al = 0.6 / (1 + np.exp(-r.x[0]))
        be = 0.98 / (1 + np.exp(-r.x[1]))
        if al + be > 0.999:
            be = 0.999 - al
        return v0 * (1 - al - be), float(al), float(be), v0
    except Exception:
        return v0, 0.0, 0.0, v0
