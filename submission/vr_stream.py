"""Streaming Lo-MacKinlay variance-ratio view (16 columns) -- the ONE implementation used for training features AND for
serving, so train/serve parity is by construction.

Per online point p, for channel a in {x, |x - mean_hist|}, q in (2, 4, 8, 16), w in (128, 512):
    VR_w,q(p) = var(last w q-sums ending at p-w+1..p) / (q * var(last w values))
    col = log(VR_w,q(p) + 1e-9) - log(VR_hist,q + 1e-9),  VR_hist,q = var(all history q-sums) / (q * var(history))
Column order: channel, then q, then w (x: q2w128, q2w512, q4w128, ..., q16w512; then |x-m| the same).
Causal: step(point) uses only the history and online points up to and including this one.
O(1) per point: running sums / sums of squares over each window (recomputed exactly every RESYNC steps so float
drift cannot accumulate). Parity with the vectorised offline view (selfimprove/cands/raw_b3_vratio.py) ~1e-7."""

from __future__ import annotations
import collections
import math
import numpy as np

QS = (2, 4, 8, 16)
WS = (128, 512)
N_COLS = 2 * len(QS) * len(WS)
RESYNC = 2048


def _vr_whole(a):
    c = np.cumsum(np.r_[0.0, a])
    out = []
    for q in QS:
        sq = c[q:] - c[:-q]
        out.append(math.log(float(sq.var()) / (q * float(a.var()) + 1e-12) + 1e-9))
    return out


class _Win:
    """sum and sum of squares of the last w values pushed."""

    __slots__ = ("w", "d", "s", "s2")

    def __init__(self, w, init):
        self.w = w
        self.d = collections.deque(init[-w:], maxlen=w)
        self.s = float(sum(self.d))
        self.s2 = float(sum(v * v for v in self.d))

    def push(self, v):
        if len(self.d) == self.w:
            o = self.d[0]
            self.s -= o
            self.s2 -= o * o
        self.d.append(v)
        self.s += v
        self.s2 += v * v

    def var(self):
        n = len(self.d)
        m = self.s / n
        return max(self.s2 / n - m * m, 0.0)

    def resync(self):
        self.s = float(sum(self.d))
        self.s2 = float(sum(v * v for v in self.d))


class _Chan:
    def __init__(self, tail, lh):
        self.lh = lh
        self.raw = collections.deque(tail[-max(QS) :], maxlen=max(QS))
        self.v1 = {w: _Win(w, tail) for w in WS}
        # q-sums ending at each of the last max(WS) history positions
        c = np.cumsum(np.r_[0.0, np.asarray(tail, np.float64)])
        self.vq = {}
        for q in QS:
            sq = (c[q:] - c[:-q]).tolist()
            for w in WS:
                self.vq[(q, w)] = _Win(w, sq)
        self.qs = {q: float(sum(list(self.raw)[-q:])) for q in QS}

    def push(self, v):
        full = len(self.raw) == self.raw.maxlen
        for q in QS:  # running q-sum: add v, drop the value q back
            drop = self.raw[-q] if len(self.raw) >= q else 0.0
            self.qs[q] += v - drop
        self.raw.append(v)
        for w in WS:
            self.v1[w].push(v)
        for q in QS:
            for w in WS:
                self.vq[(q, w)].push(self.qs[q])
        return full

    def resync(self):
        r = list(self.raw)
        for q in QS:
            self.qs[q] = float(sum(r[-q:]))
        for w in WS:
            self.v1[w].resync()
        for k in self.vq:
            self.vq[k].resync()

    def cols(self, out, k):
        for qi, q in enumerate(QS):
            for w in WS:
                out[k] = (
                    math.log(
                        self.vq[(q, w)].var() / (q * self.v1[w].var() + 1e-12) + 1e-9
                    )
                    - self.lh[qi]
                )
                k += 1
        return k


class VrStream:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, np.float64)
        self.m = float(h.mean())
        tail_n = max(WS) + max(QS)
        ah = np.abs(h - self.m)
        self.ch = (
            _Chan(h[-tail_n:].tolist(), _vr_whole(h)),
            _Chan(ah[-tail_n:].tolist(), _vr_whole(ah)),
        )
        self.n = 0

    def step(self, point):
        x = float(point)
        self.ch[0].push(x)
        self.ch[1].push(abs(x - self.m))
        self.n += 1
        if self.n % RESYNC == 0:
            self.ch[0].resync()
            self.ch[1].resync()
        out = np.empty(N_COLS)
        k = self.ch[0].cols(out, 0)
        self.ch[1].cols(out, k)
        return out


def series_view(x_hist, x_on):
    """[n_o, 16] for one series, computed by the streaming class (training-side use)."""
    s = VrStream(x_hist)
    return np.stack([s.step(v) for v in x_on]).astype(np.float32)
