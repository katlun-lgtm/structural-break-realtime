"""Streaming twin of evo/genes.py for a FIXED genome (serving side). One step per online point, causal.
Per gene: the base series (x, absdev, sign, rank, sq, diff, innov) is built from history-only constants; the trailing-window
statistic (vr / acf / mean / var) is recomputed over a small ring buffer each step with the SAME formulas as genes.py
(population variance via E[a^2] - E[a]^2); the history reference (median / mean / sd of the statistic over history
positions) is computed once at the start by genes.py itself on the history alone -- those positions only use history
values, so this is identical to the offline reference. Supported stats: vr, acf, mean, var; contrasts: logratio, z, diff."""

from __future__ import annotations
import collections
import numpy as np
import genes as G


class _Gene:
    def __init__(self, gene, x_hist, cache):
        b, st, p, w, c = gene
        assert st in ("vr", "acf", "mean", "var", "kurt", "qhi", "qlo", "runs") and c in (
            "logratio",
            "z",
            "diff",
            "sr",
            "cusum_up",
            "cusum_dn",
        ), gene
        self.acc = 0.0  # accumulator state for the sr / cusum contrasts
        self.b, self.st, self.p, self.w, self.c = b, st, int(p), int(w), c
        h = np.asarray(x_hist, np.float64)
        nh = len(h)
        self.m = h.mean()
        self.sd = h.std() + 1e-12
        self.med = np.median(h)
        self.hs = np.sort(h)
        self.nh = nh
        if b == "innov":
            P = 5
            hh = h - self.m
            X = np.stack([hh[P - k - 1 : nh - k - 1] for k in range(P)], 1)
            self.phi = np.linalg.lstsq(X, hh[P:], rcond=None)[0]
            self.zbuf = collections.deque((h[-P:] - self.m).tolist(), maxlen=P)
        # history reference from genes.py on the history alone (identical to the offline reference positions [w, nh))
        a_h = G.base_series(b, h, np.empty(0), {})
        r = G._stat_roll(a_h, st, self.p, self.w)
        hist = r[self.w : nh]
        hist = hist[np.isfinite(hist)]
        if len(hist) == 0:
            hist = np.array([0.0])
        self.ref_med, self.ref_mean, self.ref_sd = (
            float(np.median(hist)),
            float(hist.mean()),
            float(hist.std()),
        )
        ref = a_h[: G.REF_N]                                   # the SAME fixed history prefix as genes.py (leak-fixed)
        self.th = float(np.percentile(ref, 90 if st == "qhi" else 10)) if st in ("qhi", "qlo") else 0.0
        self.rmed = float(np.median(ref)) if st == "runs" else 0.0
        need = self.w + max(self.p, 1) + 1
        self.buf = collections.deque(a_h[-need:].tolist(), maxlen=need)
        self.prev_x = float(h[-1])

    def _base(self, x):
        b = self.b
        if b == "x":
            return (x - self.m) / self.sd
        if b == "absdev":
            return abs(x - self.m) / self.sd
        if b == "sign":
            return float(np.sign(x - self.med))
        if b == "rank":
            return np.searchsorted(self.hs, x, side="right") / (self.nh + 1.0) - 0.5
        if b == "sq":
            return ((x - self.m) / self.sd) ** 2
        if b == "diff":
            v = (x - self.prev_x) / self.sd
            return v
        if b == "innov":
            z = x - self.m
            e = z - sum(self.phi[k] * self.zbuf[-1 - k] for k in range(len(self.phi)))
            self.zbuf.append(z)
            return e / self.sd
        raise ValueError(b)

    def step(self, x):
        v_base = self._base(float(x))
        self.prev_x = float(x)
        self.buf.append(v_base)
        a = np.fromiter(self.buf, np.float64, len(self.buf))
        w, p = self.w, self.p

        def var(u):
            m = u.mean()
            return max(float((u * u).mean() - m * m), 0.0)

        if self.st == "vr":
            c = np.cumsum(np.r_[0.0, a])
            sq = (c[p:] - c[:-p])[-w:]
            val = var(sq) / (p * var(a[-w:]) + 1e-12)
        elif self.st == "acf":
            aw = a[-w:]
            bw = a[-w - p : len(a) - p]
            val = ((aw * bw).mean() - aw.mean() * bw.mean()) / (var(aw) + 1e-12)
        elif self.st == "kurt":
            u = a[-w:]; m1, m2, m3, m4 = (float((u ** k).mean()) for k in (1, 2, 3, 4)); v = max(m2 - m1 ** 2, 1e-18)
            val = min(max((m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / v ** 2 - 3, -50.0), 50.0)
        elif self.st in ("qhi", "qlo"):
            u = a[-w:]; val = float((u > self.th).mean() if self.st == "qhi" else (u < self.th).mean())
        elif self.st == "runs":
            sv = (a[-(w + 1):] > self.rmed).astype(float); val = float((np.diff(sv) != 0).mean())
        elif self.st == "mean":
            val = float(a[-w:].mean())
        else:
            val = var(a[-w:])
        if self.c == "logratio":
            out = np.log(abs(val) + 1e-9) - np.log(abs(self.ref_med) + 1e-9)
        elif self.c == "z":
            out = (val - self.ref_mean) / (self.ref_sd + 1e-9)
        elif self.c == "diff":
            out = val - self.ref_med
        else:  # accumulation contrasts, identical to genes.gene_column (z clipped to +-10; CUSUM_K; SR shift 0.5)
            z = (val - self.ref_mean) / (self.ref_sd + 1e-9)
            z = 0.0 if not np.isfinite(z) else min(max(z, -10.0), 10.0)
            if self.c == "sr":
                self.acc = np.logaddexp(0.0, self.acc) + (z - 0.5)
            else:
                sgn = 1.0 if self.c == "cusum_up" else -1.0
                self.acc = max(0.0, self.acc + sgn * z - G.CUSUM_K)
            out = self.acc
        return 0.0 if not np.isfinite(out) else float(out)


class GenomeStream:
    def __init__(self, genome, x_hist):
        cache = {}
        self.genes = [_Gene(tuple(g), x_hist, cache) for g in genome]

    def step(self, x):
        return np.array([g.step(x) for g in self.genes], np.float64)


def series_view(genome, x_hist, x_on):
    s = GenomeStream(genome, x_hist)
    return np.stack([s.step(v) for v in x_on]).astype(np.float32)
