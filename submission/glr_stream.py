"""Self-contained streaming GLR-CUSUM (variance-shift) feature block for the
production #12-candidate submission.

This is the ONE-PASS-CAUSAL twin of glr_block.new_feats(..., cfg="glr_var")
(/root/ml-comp/structural_break/glr_block.py). It reproduces the exact same
math -- AR(p)-whitened, sigma-standardized innovations u_t, then for each
hypothesized post-change variance ratio r in R_GRID the per-step log-LR of
u_t ~ N(0, r) vs N(0, 1):

    c_r(u) = -0.5*log(r) + 0.5*(1 - 1/r)*u^2

accumulated as a one-sided CUSUM S_r,t = max(0, S_r,t-1 + c_r(u_t)), emitted
as log1p(S_r,t), plus a "_max" column = log1p(max_r S_r,t) -- the running GLR
statistic itself.

The ONLY numerical deviation from glr_block.new_feats is the CUSUM
accumulation order: glr_block vectorizes the whole online segment at once via
the telescoping identity S_t = C_t - min_{0<=j<=t} C_j; this module instead
runs the literal sequential recurrence S_t = max(0, S_{t-1} + c_t), point by
point, so it can be driven by infer()'s one-pass generator loop exactly like
every other stream object in this package (Deep8Stream, EWMAVolStream, ...).
Both forms are mathematically identical, and on real data bit-identical at
float32 (verified 500/500 series, max diff 0.0). They CAN diverge on
pathological inputs (a |u| >~ 4e7-sigma spike makes the reference's running
cumsum absorb later increments; measured real-data max |u| is ~25) — in that
regime the sequential recurrence here is the numerically CORRECT side, and
train+infer both use it, so they can never disagree with each other.

No imports from /root/ml-comp/structural_break -- ar_fit is inlined verbatim
from self_sim_null.py so this module (and sb_prod generally) stays
self-contained for the cloud push. numpy only.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

R_GRID = [0.5, 0.7, 1.5, 2.0, 3.0]  # variance-ratio grid (post-change / null)

GLR_NAMES = [
    "glr_var_r0.5",
    "glr_var_r0.7",
    "glr_var_r1.5",
    "glr_var_r2.0",
    "glr_var_r3.0",
    "glr_var_max",
]


def ar_fit(
    x: np.ndarray, pmax: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, float, int]:
    """OLS AR(p) with p chosen by AIC over 0..pmax. Returns (intercept, coefs[p],
    residuals, sigma2, p). residuals align to x[p:].

    Verbatim copy of self_sim_null.ar_fit (/root/ml-comp/structural_break/self_sim_null.py)
    -- inlined so this module has zero dependency on the structural_break tree.
    """
    x = np.asarray(x, dtype=np.float64)
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
            k = 1
            aic = common_n * np.log(rss / common_n + 1e-300) + 2 * k
            cand = (aic, 0, mu, np.zeros(0), x - mu, max(resid.var(), 1e-12))
        else:
            # design: rows t=p..n-1 ; cols [1, x[t-1..t-p]]
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
            k = p + 1
            aic = common_n * np.log(rss / common_n + 1e-300) + 2 * k
            cand = (aic, p, beta[0], beta[1:], resid, max(resid.var(), 1e-12))
        if best is None or cand[0] < best[0]:
            best = cand
    _aic, p, intercept, coefs, resid, sigma2 = best
    return intercept, coefs, resid, sigma2, p


class GLRVarStream:
    """One-pass causal variance-shift GLR-CUSUM stream.

    __init__ fits the AR(p) sieve + sigma0 on x_historical (matches
    glr_block._whitened_u's setup exactly); step(x) whitens one online point
    (matches self_sim_null.whiten's per-step recurrence exactly, run
    incrementally instead of via the vectorized concatenate-and-loop), updates
    the 5 variance-ratio CUSUM accumulators, and returns the 6 GLR_NAMES
    columns as a list of floats.
    """

    __slots__ = ("_med", "_intercept", "_coefs", "_p", "_s0", "_lags", "_S")

    def __init__(self, x_historical: Sequence[float]):
        h = np.asarray(x_historical, dtype=np.float64)
        h = h[np.isfinite(h)]
        if h.size == 0:
            h = np.zeros(1)
        self._med = float(np.median(h))

        intercept, coefs, resid, sigma2, p = ar_fit(h, pmax=10)
        self._intercept = float(intercept)
        self._coefs = np.asarray(coefs, dtype=np.float64)
        self._p = int(p)

        # out-of-sample variance inflation -- matches glr_block._whitened_u
        # (in-sample OLS residuals are deflated by the fitted dof).
        dof = max(resid.size - p - 1, 1)
        sigma2_adj = sigma2 * (resid.size / dof) if resid.size > 0 else sigma2
        if not np.isfinite(sigma2_adj) or sigma2_adj < 1e-9:
            sigma2_adj = max(sigma2, 1e-9)
        self._s0 = float(np.sqrt(sigma2_adj))

        # lag buffer = last p RAW (sanitized) hist values, oldest->newest, so
        # self._lags[-1] is lag-1, self._lags[-2] is lag-2, ... matching
        # self_sim_null.whiten's warmup seeding (ext = warmup[-p:] + x).
        if self._p > 0:
            if h.size >= self._p:
                self._lags = [float(v) for v in h[-self._p :]]
            else:
                self._lags = [0.0] * self._p
        else:
            self._lags = []

        self._S = [0.0] * len(R_GRID)

    def step(self, x: float) -> List[float]:
        xv = float(x)
        if not math.isfinite(xv):
            xv = self._med

        if self._p > 0:
            pred = self._intercept
            for i in range(self._p):
                pred += self._coefs[i] * self._lags[-1 - i]
            innov = xv - pred
        else:
            innov = xv - self._intercept

        u = innov / self._s0

        out: List[float] = []
        for i, r in enumerate(R_GRID):
            c = -0.5 * math.log(r) + 0.5 * (1.0 - 1.0 / r) * u * u
            s = self._S[i] + c
            if s < 0.0:
                s = 0.0
            self._S[i] = s
            out.append(math.log1p(s))
        out.append(math.log1p(max(self._S)))

        if self._p > 0:
            self._lags.append(xv)
            del self._lags[0]

        return out


def batch_glr(x_hist: Sequence[float], x_online: Sequence[float]) -> np.ndarray:
    """float32 (L, 6) GLR-var feature matrix for one series. Constructs a
    GLRVarStream and drives it point-by-point -- the SAME code path infer()
    uses -- so batch and stream outputs are bit-identical by construction."""
    g = GLRVarStream(x_hist)
    rows = [g.step(float(v)) for v in x_online]
    if not rows:
        return np.zeros((0, len(GLR_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
