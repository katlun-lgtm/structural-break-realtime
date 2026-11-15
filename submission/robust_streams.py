"""Causal streaming implementations of the four frozen robust-view feature blocks.

The batch builders that produced the training caches are preserved under
``recovered_full_compound_20260910/source_builders``.  This module keeps their
arithmetic and float32 boundary, while exposing one row at a time for Crunch's
real-time generator.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, ndtri


FLIGNER_NAMES = ("flig_z", "flig_z_w", "flig_cusum", "flig_max",
                 "flig_cusum_dc", "flig_max_dc")
ARBOCPD_NAMES = ("arbocpd_p_c0.25", "arbocpd_p_c1", "arbocpd_loc", "arbocpd_conc")
_DB_STATS = ("ks", "w1", "ad", "ent", "skew", "kurt", "qratio")
DISTBATTERY_NAMES = tuple(f"{s}_{w}" for w in ("pre", "win") for s in _DB_STATS)
_SB_CH = ("mean", "var", "dep", "vol", "skew", "tail")
SPECBASIS_NAMES = tuple(f"sb_{c}_w{w}" for w in (64, 256) for c in _SB_CH)

_FLIG_WIN = 128
_FLIG_DC_N = 64
_q = (np.arange(1, 200001) - 0.5) / 200000.0
_a = ndtri(0.5 + _q / 2.0)
_A0_MU, _A0_SD = float(_a.mean()), float(_a.std())
del _q, _a


def _ar_fit(x, pmax=4):
    x = np.asarray(x, float)
    n = x.size
    best = (np.inf, 0, float(x.mean()) if n else 0.0, np.zeros(0))
    for p in range(pmax + 1):
        if n <= p + 5:
            break
        if p == 0:
            r = x - x.mean()
            rss = float(r @ r)
            cand = (n * np.log(rss / n + 1e-300) + 2, 0, float(x.mean()), np.zeros(0))
        else:
            z = np.column_stack([np.ones(n - p)] +
                                [x[p-k-1:n-k-1] for k in range(p)])
            b, *_ = np.linalg.lstsq(z, x[p:], rcond=None)
            r = x[p:] - z @ b
            rss = float(r @ r)
            cand = (n * np.log(rss / n + 1e-300) + 2*(p+1),
                    p, float(b[0]), b[1:])
        if cand[0] < best[0]:
            best = cand
    return best[1], best[2], best[3]


class FlignerStream:
    """Frozen-history Fligner scores with the causal prefix-replay repair."""

    def __init__(self, history):
        h = np.asarray(history, dtype=np.float64)
        hf = h[-2000:]
        self.p, self.intercept, self.coef = _ar_fit(hf)
        if self.p:
            hr = np.array([
                hf[i] - (self.intercept + sum(self.coef[j] * hf[i-1-j]
                                              for j in range(self.p)))
                for i in range(self.p, len(hf))
            ])
        else:
            hr = hf - self.intercept
        self.valid = len(hr) >= 32
        self.med = float(np.median(hr)) if self.valid else 0.0
        self.ref = np.sort(np.abs(hr - self.med)) if self.valid else np.zeros(0)
        self.raw = list(h[-self.p:]) if self.p else []
        self.scores = []
        self.sum_score = 0.0
        self.raw_cs = self.raw_max = 0.0
        self.dc_cs = self.dc_max = 0.0
        self.drift = None

    def step(self, point):
        x = float(point)
        if not self.valid:
            self.raw.append(x)
            return np.zeros(6, dtype=np.float32)
        pred = self.intercept
        for k in range(self.p):
            pred += self.coef[k] * self.raw[-1-k]
        self.raw.append(x)
        d = abs(x - pred - self.med)
        q = np.searchsorted(self.ref, d, side="right") / (len(self.ref) + 1.0)
        q = min(max(float(q), 1e-6), 1.0 - 1e-6)
        a = float((ndtri(0.5 + q / 2.0) - _A0_MU) / _A0_SD)
        self.scores.append(a)
        self.sum_score += a
        n = len(self.scores)
        z = self.sum_score / math.sqrt(n)
        lo = max(0, n - _FLIG_WIN)
        zw = sum(self.scores[lo:]) / math.sqrt(n - lo)
        self.raw_cs = max(0.0, self.raw_cs + a)
        self.raw_max = max(self.raw_max, self.raw_cs)

        if n <= _FLIG_DC_N:
            # Match the cache builder's NumPy reduction (not Python's running
            # sum, which differs by a few ulps on degenerate prefixes).
            drift = float(np.asarray(self.scores, dtype=np.float64).mean())
            cs = mx = 0.0
            for value in self.scores:
                cs = max(0.0, cs + value - drift)
                mx = max(mx, cs)
            self.dc_cs, self.dc_max = cs, mx
            if n == _FLIG_DC_N:
                self.drift = drift
        else:
            self.dc_cs = max(0.0, self.dc_cs + a - self.drift)
            self.dc_max = max(self.dc_max, self.dc_cs)
        return np.asarray((z, zw, math.log1p(self.raw_cs), math.log1p(self.raw_max),
                           math.log1p(self.dc_cs), math.log1p(self.dc_max)),
                          dtype=np.float32)


def _seg_ml(n, syy, sxy, sxx, c, a0, b0, lg_a0):
    ic = 1.0 / c
    a11 = sxx[0] + ic
    a22 = sxx[2] + ic
    a12 = sxx[1]
    det = np.maximum(a11*a22 - a12*a12, 1e-300)
    v1, v2 = sxy[0], sxy[1]
    quad = (v1*v1*a22 - 2.0*v1*v2*a12 + v2*v2*a11) / det
    bn = b0 + 0.5*np.maximum(syy - quad, 1e-12)
    an = a0 + n/2.0
    return (-0.5*n*math.log(2*math.pi) - 0.5*np.log(det) - math.log(c)
            + a0*math.log(b0) - an*np.log(bn) + gammaln(an) - lg_a0)


class ArbocpdStream:
    """Incremental P=2 NIG-marginal AR break posterior; O(t) per point."""

    _CS = (0.25, 1.0)
    _MIN = 16

    def __init__(self, e_hist):
        eh = np.asarray(e_hist, dtype=np.float64)
        s2h = float(np.var(eh)) if len(eh) > 8 else 1.0
        if not np.isfinite(s2h) or s2h <= 0:
            s2h = 1.0
        self.a0 = 4.0
        self.b0 = self.a0*s2h
        self.lg_a0 = float(gammaln(self.a0))
        tail = list(eh[-2:])
        self.prev = ([0.0] * (2-len(tail))) + tail
        self.pyy = [0.0]
        self.pxy0 = [0.0]
        self.pxy1 = [0.0]
        self.pxx00 = [0.0]
        self.pxx01 = [0.0]
        self.pxx11 = [0.0]

    def step(self, value):
        y = float(value)
        x0, x1 = float(self.prev[-1]), float(self.prev[-2])
        self.prev.append(y)
        self.pyy.append(self.pyy[-1] + y*y)
        self.pxy0.append(self.pxy0[-1] + x0*y)
        self.pxy1.append(self.pxy1[-1] + x1*y)
        self.pxx00.append(self.pxx00[-1] + x0*x0)
        self.pxx01.append(self.pxx01[-1] + x0*x1)
        self.pxx11.append(self.pxx11[-1] + x1*x1)
        n_t = len(self.pyy)-1
        if n_t <= 2*self._MIN:
            return np.zeros(4, dtype=np.float32)
        # Batch builder uses t=n_t-1 and arange(MIN, t-MIN+1), whose
        # exclusive upper bound is n_t-MIN.
        s = np.arange(self._MIN, n_t-self._MIN)
        pyy = np.asarray(self.pyy)
        pxy = (np.asarray(self.pxy0), np.asarray(self.pxy1))
        pxx = (np.asarray(self.pxx00), np.asarray(self.pxx01), np.asarray(self.pxx11))
        out = np.zeros(4, dtype=np.float64)
        best = None
        for ci, c in enumerate(self._CS):
            n1 = s.astype(np.float64)
            n2 = float(n_t)-n1
            ml1 = _seg_ml(n1, pyy[s], [q[s] for q in pxy], [q[s] for q in pxx],
                          c, self.a0, self.b0, self.lg_a0)
            ml2 = _seg_ml(n2, pyy[n_t]-pyy[s],
                          [q[n_t]-q[s] for q in pxy],
                          [q[n_t]-q[s] for q in pxx],
                          c, self.a0, self.b0, self.lg_a0)
            ln_br = ml1 + ml2 - math.log(len(s))
            ln_no = _seg_ml(np.array([float(n_t)]), np.array([pyy[n_t]]),
                            [np.array([q[n_t]]) for q in pxy],
                            [np.array([q[n_t]]) for q in pxx],
                            c, self.a0, self.b0, self.lg_a0)[0]
            m = max(float(ln_no), float(ln_br.max()))
            weights = np.exp(ln_br-m)
            pb = 0.5*float(weights.sum())
            pn = 0.5*math.exp(float(ln_no)-m)
            out[ci] = pb/(pb+pn) if pb+pn > 0 else 0.0
            if ci == 0:
                total = float(weights.sum())
                best = weights/total if total > 0 else None
        if best is not None:
            out[2] = float((s*best).sum())/n_t
            out[3] = float(best.max())
        return out.astype(np.float32)


_NBIN, _NGRID, _DWIN, _DMIN = 32, 96, 256, 16
_GLO, _GHI = -12.0, 12.0
_EDGES = np.linspace(0.0, 1.0, _NBIN+1)
_AD_W = 1.0/np.maximum(_EDGES[1:-1]*(1.0-_EDGES[1:-1]), 1e-6)


def _cdf_stats(cnt, n):
    if n < _DMIN:
        return 0.0, 0.0, 0.0, 0.0
    p = cnt/n
    f = np.cumsum(p)[:-1]
    d = f-_EDGES[1:-1]
    nz = p[p > 0]
    return (float(np.abs(d).max()),
            float(np.abs(np.concatenate([d, [0.0]])).sum()/_NBIN),
            float((d*d*_AD_W).sum()/(_NBIN-1)),
            float(-(nz*np.log(nz)).sum()))


def _shape_stats(n, s1, s2, s3, s4, gcnt):
    if n < _DMIN:
        return 0.0, 0.0, 0.0
    m = s1/n
    v = s2/n-m*m
    if not v > 1e-12:
        return 0.0, 0.0, 0.0
    sd = math.sqrt(v)
    m3 = s3/n-3*m*s2/n+2*m**3
    m4 = s4/n-4*m*s3/n+6*m*m*s2/n-3*m**4
    skew, kurt = m3/sd**3, m4/(v*v)-3.0
    total = gcnt.sum()
    if total < _DMIN:
        return float(skew), float(kurt), 0.0
    c = np.cumsum(gcnt)/total
    edges = np.linspace(_GLO, _GHI, _NGRID+1)[1:]
    q = np.interp([0.10, 0.25, 0.75, 0.90], c, edges)
    inner = q[2]-q[1]
    return float(skew), float(kurt), float((q[3]-q[0])/inner) if inner > 1e-9 else 0.0


class DistBatteryStream:
    def __init__(self, e_hist):
        eh = np.asarray(e_hist, dtype=np.float64)
        eh = eh[np.isfinite(eh)]
        self.ref = np.sort(eh if len(eh) >= 8 else np.zeros(8))
        self.cp = np.zeros(_NBIN); self.cw = np.zeros(_NBIN)
        self.gp = np.zeros(_NGRID); self.gw = np.zeros(_NGRID)
        self.rb = np.zeros(_DWIN, dtype=np.int64)
        self.rg = np.zeros(_DWIN, dtype=np.int64)
        self.rx = np.zeros(_DWIN)
        self.pm = np.zeros(4); self.wm = np.zeros(4); self.t = 0

    def step(self, value):
        x = float(value) if math.isfinite(float(value)) else 0.0
        u = (np.searchsorted(self.ref, x, side="right")+0.5)/(len(self.ref)+1.0)
        b = min(int(u*_NBIN), _NBIN-1)
        gi = min(int((min(max(x, _GLO), _GHI)-_GLO)/(_GHI-_GLO)*_NGRID), _NGRID-1)
        self.cp[b] += 1; self.gp[gi] += 1
        powers = np.array((x, x*x, x**3, x**4))
        self.pm += powers
        slot = self.t % _DWIN
        if self.t >= _DWIN:
            ob, og, ox = self.rb[slot], self.rg[slot], self.rx[slot]
            self.cw[ob] -= 1; self.gw[og] -= 1
            self.wm -= np.array((ox, ox*ox, ox**3, ox**4))
        self.rb[slot], self.rg[slot], self.rx[slot] = b, gi, x
        self.cw[b] += 1; self.gw[gi] += 1; self.wm += powers
        self.t += 1
        np_, nw = self.t, min(self.t, _DWIN)
        row = (*_cdf_stats(self.cp, np_), *_shape_stats(np_, *self.pm, self.gp),
               *_cdf_stats(self.cw, nw), *_shape_stats(nw, *self.wm, self.gw))
        return np.nan_to_num(np.asarray(row, dtype=np.float32))


def _spear1(v):
    if len(v) < 8:
        return 0.0
    r = np.argsort(np.argsort(v)).astype(np.float64)
    a, b = r[:-1], r[1:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(((a-a.mean())*(b-b.mean())).mean()/(sa*sb))


def _spec_stats(v):
    q = np.percentile(v, [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
    o1, q1, o3, med, o5, q3, o7 = q
    mad = np.median(np.abs(v-med)); iqr = q3-q1
    bowley = (q3+q1-2*med)/iqr if iqr > 1e-9 else 0.0
    moors = ((o7-o5)+(o3-o1))/iqr if iqr > 1e-9 else 0.0
    return med, mad, _spear1(v), _spear1(np.abs(v-med)), bowley, moors


class SpecBasisStream:
    def __init__(self, e_hist):
        self.x = []
        self.scale = float(np.std(e_hist)) if len(e_hist) > 8 else 1.0
        if not np.isfinite(self.scale) or self.scale <= 0:
            self.scale = 1.0

    def step(self, value):
        self.x.append(float(value))
        row = np.zeros(12, dtype=np.float64)
        for wi, w in enumerate((64, 256)):
            if len(self.x) < 2*w:
                continue
            a = np.asarray(self.x[-2*w:-w]); b = np.asarray(self.x[-w:])
            ma, da, pa, va, ka, ta = _spec_stats(a)
            mb, db, pb, vb, kb, tb = _spec_stats(b)
            pooled = 0.5*(da+db)*1.4826
            if pooled < 1e-9:
                pooled = self.scale
            base = wi*6
            row[base:base+6] = ((mb-ma)/(pooled*math.sqrt(2.0/w)),
                                math.log(max(db, 1e-9)/max(da, 1e-9)),
                                pb-pa, vb-va, kb-ka, tb-ta)
        return np.nan_to_num(row.astype(np.float32))
