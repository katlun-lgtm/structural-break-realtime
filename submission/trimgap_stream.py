"""trimgap_core streaming block — build K's certified 4-column family.

Certified 2026-08-20 against the build-H controls (f16h2_lgb_oof + mlparch_s2_ctrl_oof)
that match pinned #79:  FULL +0.00226 [+0.00111,+0.00353] FUND
                     LGBonly +0.00159 [+0.00060,+0.00252] FUND

Mechanism: the running gap between the ordinary cumulative mean of the online prefix and a
TRIMMED mean of the same prefix, where the trim is evaluated on a 16-bin histogram whose
edges are FROZEN from the history quantiles. A break that shifts or fattens one tail moves
the ordinary mean more than the trimmed mean, so the gap opens. State-contrast, which is the
only mechanism that has ever funded on this problem.

Columns: gap_z_mild (gap vs the history's own mean-minus-trimmed gap), gap_z_deep (alpha
0.5), gap_raw_mild, gap_growth (rolling-64 mean of the gap minus its expanding mean).
All divided by the history std and hard-clipped to +-20.

BOUNDEDNESS (the law that cost build I a board slot): there is NO sqrt(t) or other
length-scaling here — these are mean DIFFERENCES, so magnitude does not grow with the
stream. Measured max|v| over L=510..20000: 0.14-0.30 on nulls, 2.50 under a +10sd mean
shift, 1.03 under a 20x variance change, 0.60 under AR(0.8) — nowhere near the clip. Only a
degenerate near-constant history (std ~ 0) reaches the clip, and there it saturates rather
than diverging.

Math is a 1:1 running port of fe_candidates_v199.gen_trimgap_core: the batch version
cumsums per-bin counts/sums over the prefix, which is exactly the running histogram this
object maintains, so batch == stream by construction. batch_trimgap() drives THIS object.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

TRIMGAP_NAMES = ["tg_z_mild", "tg_z_deep", "tg_raw_mild", "tg_growth"]
_EPS = 1e-8
_B = 16  # histogram bins, edges frozen from history quantiles
_W = 64  # rolling window for the growth column
_CLIP = 20.0


def _trim_mean_from_bins(
    bcount: np.ndarray, bsum: np.ndarray, n: float, alpha: float
) -> float:
    """Trimmed mean over the prefix using frozen-edge bin counts/sums.

    Identical arithmetic to the batch path, evaluated on running totals rather than on a
    cumsum of one-hots. Partial bins are kept fractionally, which is what makes the batch
    and stream agree exactly instead of only approximately.
    """
    rankcum = np.cumsum(bcount)
    prefix_before = rankcum - bcount
    k = (alpha / 2.0) * n
    overlap_lo = np.maximum(prefix_before, k)
    overlap_hi = np.minimum(rankcum, n - k)
    kept = np.clip(overlap_hi - overlap_lo, 0.0, None)
    kept = np.minimum(kept, bcount)
    frac = np.where(bcount > 0, kept / np.maximum(bcount, _EPS), 0.0)
    return float((bsum * frac).sum() / max(float(kept.sum()), _EPS))


class TrimGapStream:
    __slots__ = (
        "_edges",
        "_inner",
        "_scale",
        "_hgap",
        "_t",
        "_sum",
        "_bc",
        "_bs",
        "_gsum",
        "_ring",
        "_ri",
    )

    def __init__(self, x_historical: Sequence[float]):
        h = np.asarray(x_historical, np.float64).ravel()
        h = h[np.isfinite(h)]
        hn = len(h)
        if hn >= 8:
            edges = np.quantile(h, np.linspace(0.0, 1.0, _B + 1))
        elif hn >= 2:
            hs = float(np.std(h))
            hs = hs if (math.isfinite(hs) and hs > _EPS) else 1.0
            edges = float(np.mean(h)) + hs * np.linspace(-3.0, 3.0, _B + 1)
        else:
            edges = np.linspace(-3.0, 3.0, _B + 1)
        edges = np.maximum.accumulate(edges) + np.arange(_B + 1) * _EPS
        self._edges = edges
        self._inner = edges[1:_B]

        scale = float(np.std(h)) if hn > 1 else 1.0
        self._scale = scale if (math.isfinite(scale) and scale >= _EPS) else _EPS

        # the history's OWN mean-minus-trimmed gap, used to centre gap_z_mild
        if hn >= 2:
            bidx = np.clip(np.searchsorted(self._inner, h, side="right"), 0, _B - 1)
            bc = np.bincount(bidx, minlength=_B).astype(np.float64)
            bs = np.bincount(bidx, weights=h, minlength=_B).astype(np.float64)
            self._hgap = float(np.mean(h)) - _trim_mean_from_bins(
                bc, bs, float(hn), 0.2
            )
        else:
            self._hgap = 0.0

        self._t = 0
        self._sum = 0.0
        self._bc = np.zeros(_B, np.float64)
        self._bs = np.zeros(_B, np.float64)
        self._gsum = 0.0  # expanding sum of gap_mild
        self._ring = np.zeros(_W, np.float64)  # last W gap_mild values
        self._ri = 0

    def step(self, x: float) -> List[float]:
        xv = float(x) if math.isfinite(x) else 0.0
        self._t += 1
        t = float(self._t)
        self._sum += xv
        cm = self._sum / t

        b = int(np.clip(np.searchsorted(self._inner, xv, side="right"), 0, _B - 1))
        self._bc[b] += 1.0
        self._bs[b] += xv

        tm_mild = _trim_mean_from_bins(self._bc, self._bs, t, 0.2)
        tm_deep = _trim_mean_from_bins(self._bc, self._bs, t, 0.5)
        gap_mild = cm - tm_mild
        gap_deep = cm - tm_deep

        self._gsum += gap_mild
        self._ring[self._ri % _W] = gap_mild
        self._ri += 1
        nwin = min(self._t, _W)
        if self._t >= _W:
            roll = float(self._ring.sum()) / nwin
        else:
            roll = float(self._ring[: self._t].sum()) / nwin
        growth = roll - self._gsum / t

        warm = self._t >= 8
        s = self._scale
        out = [
            ((gap_mild - self._hgap) / s) if warm else 0.0,
            (gap_deep / s) if warm else 0.0,
            (gap_mild / s) if warm else 0.0,
            (growth / s) if self._t >= _W else 0.0,
        ]
        return [min(max(v, -_CLIP), _CLIP) if math.isfinite(v) else 0.0 for v in out]


def batch_trimgap(x_hist: Sequence[float], x_online: Sequence[float]) -> np.ndarray:
    """float32 (L, 4); drives TrimGapStream point-by-point -> batch == stream."""
    s = TrimGapStream(x_hist)
    rows = [s.step(float(v)) for v in x_online]
    if not rows:
        return np.zeros((0, len(TRIMGAP_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
