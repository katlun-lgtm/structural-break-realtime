"""Bayesian changepoint posterior as a streaming block: P(break has already occurred by t).

WHY IT IS HERE. Every other variance detector in this build MAXIMISES a GAUSSIAN log-LR over
candidate split points (glr_var, garch_var). Two measured weaknesses of that construction:
  * max-over-split is dominated by noise when the signal is weak -- an unwindowed classical GLR
    scored 0.4764 on faint breaks, BELOW chance;
  * the Gaussian likelihood is misspecified -- the calibrated DGP is ~35% Student-t(df 5).
Marginalising over the split point with a Normal-Inverse-Gamma prior fixes both by construction:
the marginal likelihood IS a Student-t, and evidence is integrated rather than maximised. The
competition docs list "Bayesian sequential models" as a recommended approach and this build had
no run-length/posterior model of any kind.

THE SCORE IS THE TARGET QUANTITY. The task asks for P(a break has already occurred anywhere in
the online segment by t). This computes exactly that posterior, using the prior the docs state
(50% of series contain a break):

    log ML(x | a, b) = -n/2 log(2pi) + a log b - lgamma(a) + lgamma(a + n/2)
                       - (a + n/2) log(b + S/2)          S = sum x^2, zero-mean innovations
    no break : the whole online prefix continues the HISTORY regime (strong prior from history)
    break@s  : x[0:s] continues history, x[s:t] is a NEW regime (weak prior, variance refit)

MEASURED (deployment scale, train 9000 / hold 1000): univariate within-t AUC 0.5756 (vs
garch_var_max 0.5977, glr_var_max 0.5869, ev_max_run 0.5607); incremental over the 121-col stack
+0.0011 on the primary split and +0.0014 on three FRESH folds -- the only candidate in two days
that did not shrink out of sample, but still small. Cost 60 us/pt (~+6% infer).

Batch and stream are IDENTICAL BY CONSTRUCTION: `batch_bocpd` drives this same object.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.special import gammaln

BOCPD_NAMES = ["bocpd_p", "bocpd_p4", "bocpd_loc", "bocpd_conc"]

_A_WEAK = (1.0, 4.0)   # new-regime prior strengths: eager vs conservative
_MIN_SEG = 8
_PRIOR_BREAK = 0.5     # the task states 50% of series contain a break
_CHUNK = 256


class BOCPDStream:
    """O(t) per step: recompute the split-point posterior from cumulative sums of squares.

    Not the O(K) run-length recursion -- the exact sum is affordable here (60 us/pt measured)
    and avoids the pruning approximation, which matters precisely for the weak-evidence
    regime this block exists to serve.
    """

    __slots__ = ("_ah", "_bh", "_S", "_n", "_GA", "_GB", "_lg_ah", "_lg_a0", "_cap",
                 "_sd")

    def __init__(self, e_hist, span: int = 0):
        """`span` trims ONLY the PRIOR'S EVIDENCE, never the scaling.

        The NIG prior is ah = A0 + nh/2, bh = B0 + sum(eh^2)/2 with nh the history length, so a
        1000-5000 point history makes the prior worth thousands of observations and no amount of
        online evidence can move it. Trimming nh is the whole intervention. But `sd` MUST stay on
        the full history: location/scale wants all the data, only the reference wants trimming.
        Deriving sd from the trimmed window instead is a DIFFERENT statistic -- it produced a
        6.2e-01 mismatch against the measured cache before this parameter existed.
        """
        eh = np.asarray(e_hist, dtype=np.float64)
        eh = eh[np.isfinite(eh)]
        if eh.size < 8:
            eh = np.zeros(8)
        sd = float(eh.std())
        if not (sd > 1e-12):
            sd = 1.0
        eh = eh / sd
        if span and eh.size > span:
            eh = eh[-span:]              # AFTER scaling -- trim the reference only
        nh = eh.size
        ss = float(np.dot(eh, eh))
        # One posterior per new-regime prior strength; the history is the SAME evidence for all.
        self._ah = [a + nh / 2.0 for a in _A_WEAK]
        self._bh = [a + ss / 2.0 for a in _A_WEAK]
        self._lg_ah = [math.lgamma(v) for v in self._ah]
        self._lg_a0 = [math.lgamma(a) for a in _A_WEAK]
        # Scaling lives INSIDE the object so callers cannot forget it -- step() takes a raw
        # innovation and standardizes by the history sd, exactly as the batch path does.
        self._sd = sd
        self._S = np.zeros(_CHUNK + 1, dtype=np.float64)
        self._n = 0
        self._cap = _CHUNK
        self._grow_tables(_CHUNK)

    def _grow_tables(self, cap):
        k = np.arange(cap + 1, dtype=np.float64)
        self._GA = [gammaln(ah + k / 2.0) for ah in self._ah]
        self._GB = [gammaln(a + k / 2.0) for a in _A_WEAK]
        self._cap = cap

    def step(self, e_raw: float) -> List[float]:
        """Feed ONE RAW innovation; returns the 4 BOCPD features. Scaling is internal."""
        x = float(e_raw) / self._sd
        if not math.isfinite(x):
            x = 0.0
        if self._n + 1 > self._cap:
            new = max(self._cap * 2, self._n + 2)
            S2 = np.zeros(new + 1, dtype=np.float64)
            S2[: self._n + 1] = self._S[: self._n + 1]
            self._S = S2
            self._grow_tables(new)
        self._S[self._n + 1] = self._S[self._n] + x * x
        self._n += 1
        t = self._n
        if t < 2 * _MIN_SEG:
            return [0.0, 0.0, 0.0, 0.0]

        s = np.arange(_MIN_SEG, t - _MIN_SEG + 1)
        n1 = s.astype(np.float64)
        n2 = (t - s).astype(np.float64)
        Ss = self._S[s]
        St = self._S[t]
        out = []
        wn = None
        for i, A0 in enumerate(_A_WEAK):
            ah, bh = self._ah[i], self._bh[i]
            ln_no = (ah * math.log(bh) - self._lg_ah[i] + self._GA[i][t]
                     - (ah + t / 2.0) * math.log(bh + St / 2.0))
            ln_br = ((ah * math.log(bh) - self._lg_ah[i] + self._GA[i][s]
                      - (ah + n1 / 2.0) * np.log(bh + Ss / 2.0))
                     + (A0 * math.log(A0) - self._lg_a0[i] + self._GB[i][t - s]
                        - (A0 + n2 / 2.0) * np.log(A0 + (St - Ss) / 2.0))
                     - math.log(len(s)))
            m = max(ln_no, float(ln_br.max()))
            w = np.exp(ln_br - m)
            pb = _PRIOR_BREAK * float(w.sum())
            pn = (1.0 - _PRIOR_BREAK) * math.exp(ln_no - m)
            out.append(pb / (pb + pn) if (pb + pn) > 0 else 0.0)
            if i == 0:
                tot = float(w.sum())
                wn = w / tot if tot > 0 else None
        if wn is None:
            out.extend([0.0, 0.0])
        else:
            out.append(float((s * wn).sum()) / t)
            out.append(float(wn.max()))
        return [float(v) for v in out]


def batch_bocpd(e_hist, e_online, span: int = 0) -> np.ndarray:
    """Batch path — drives the SAME streaming object, so parity is by construction."""
    eh = np.asarray(e_hist, dtype=np.float64)
    eh = eh[np.isfinite(eh)]
    if eh.size < 8:
        eh = np.zeros(8)
    sd = float(eh.std())
    if not (sd > 1e-12):
        sd = 1.0
    st = BOCPDStream(eh, span=span)
    o = np.asarray(e_online, dtype=np.float64)
    rows = [st.step(float(v)) for v in o]
    return np.asarray(rows, dtype=np.float32).reshape(len(o), len(BOCPD_NAMES))
