"""v4 featurizer = v3 + moment-targeted CUSUMs (skew z^3, kurtosis z^4-3, read
from the detector) + transform-expanded features (first-difference higher
moments, absolute-value series mean/var/KS). Widens the feature bank toward the
'breadth across transforms' strategy the winning solutions use.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import anderson_ksamp

from detector import OnlineBreakDetector, _acf1

warnings.filterwarnings("ignore")


def _fast_ks(a, b, b_sorted=None):
    """KS statistic via numpy (~10x faster than scipy for small arrays).
    Pass b_sorted to avoid re-sorting b on every call."""
    na, nb = len(a), len(b)
    a_s = np.sort(a)
    b_s = b_sorted if b_sorted is not None else np.sort(b)
    combined = np.concatenate([a_s, b_s])
    combined.sort()
    cdf_a = np.searchsorted(a_s, combined, side='right') / na
    cdf_b = np.searchsorted(b_s, combined, side='right') / nb
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _fast_mwu(a, b, b_sorted=None):
    """Mann-Whitney U statistic (normalized) via numpy."""
    na, nb = len(a), len(b)
    b_s = b_sorted if b_sorted is not None else np.sort(b)
    u = float(np.sum(np.searchsorted(b_s, a, side='left')))
    return abs(u / (na * nb) - 0.5) * 2.0

FEATURE_NAMES = (
    [
        "n",
        "log_n",
        "mean_cusum",
        "var_cusum",
        "skew_cusum",
        "kurt_cusum",
        "ks",
        "ad",
        "mwu",
        "m64",
        "v64",
        "sk64",
        "ku64",
        "m16",
        "v16",
        "mall",
        "vall",
    ]
    + [f"dq{i}" for i in range(5)]
    + [
        "acf_w",
        "slope",
        "diff_mean",
        "diff_var",
        "sign_chg",
        "spec_hilo",
        "rank_shift",
        "js_div",
        "dvar2",
        "dvar4",
        "ar1c",
        "ar2c",
        "m32",
        "v32",
        "cumsum_rng",
        "diff_skew",
        "diff_kurt",
        "abs_mean",
        "abs_var",
        "abs_ks",
        "last_z",
    ]
)
_N_WIN = 33


def _q(a):
    return np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9])


def _spec_hilo(a):
    if len(a) < 8:
        return 0.0
    a = a - a.mean()
    p = np.abs(np.fft.rfft(a)) ** 2
    if p.sum() < 1e-12:
        return 0.0
    half = len(p) // 2
    return math.log((p[half:].sum() + 1e-12) / (p[1:half].sum() + 1e-12))


def _js(w, ref, edges):
    pw = np.histogram(w, bins=edges)[0] / (len(w) + 1e-12) + 1e-9
    pr = np.histogram(ref, bins=edges)[0] / (len(ref) + 1e-12) + 1e-9
    m = 0.5 * (pw + pr)
    return float(0.5 * np.sum(pw * np.log(pw / m)) + 0.5 * np.sum(pr * np.log(pr / m)))


def _dvar(a, k):
    return float(np.var(a[k:] - a[:-k])) if len(a) > k else 0.0


def _ar2(a):
    a = a - a.mean()
    if len(a) < 8:
        return 0.0, 0.0
    try:
        c = np.linalg.lstsq(np.column_stack([a[1:-1], a[:-2]]), a[2:], rcond=None)[0]
        return float(c[0]), float(c[1])
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def _moms(a):
    if len(a) < 2:
        return 0.0, 0.0
    s = a.std() + 1e-12
    return (((a - a.mean()) ** 3).mean() / s**3, ((a - a.mean()) ** 4).mean() / s**4)


# ⛔ REFERENCE TRIM (REF_SPAN=500) TESTED AND REJECTED 2026-07-30. Adopted from Katlun's build on
# the belief it drove his .6313; his own ledger attributes that arc to `garch` +0.74, `pmax 4->10`
# +0.348 and `refresh=1` +0.117 -- the trim is NOT credited. Our deployment-scale harness scored
# it NEGATIVE in all four folds: -0.0012 primary, fresh folds -0.0032/-0.0015/-0.0017, mean
# -0.0021. Sanity-checked as a real change (14/121 columns differ). His oracle diagnostic showed
# KS(window, ref) break-vs-null AUC improving monotonically with trimming (0.5251 -> 0.5324), so
# the TWO-SAMPLE STATISTIC does sharpen -- it just does not survive end-to-end, presumably
# because a 500-point reference is noisier and the model already reads level via mu/sd.
# ⇒ Reverted to the full-history reference. Do not re-adopt without an end-to-end test.
class RichFeaturizerV4:
    def __init__(self, historical, refresh: int = 4, heavy_every: int = 8) -> None:
        h = np.asarray(historical, dtype=np.float64)
        h = h[np.isfinite(h)]
        if h.size < 6:
            h = np.zeros(6)
        self.mu = float(np.median(h))
        mad = float(np.median(np.abs(h - self.mu)))
        self.sd = 1.4826 * mad if mad > 1e-9 else max(float(h.std()), 1e-9)
        self.var_h = max(float(h.var()), 1e-12)
        hstd = h.std() + 1e-12
        self.skew_h = float(((h - h.mean()) ** 3).mean() / hstd**3)
        self.kurt_h = float(((h - h.mean()) ** 4).mean() / hstd**4)
        self.acf_h = _acf1(h)
        self.qh = _q(h)
        self.ref = h[np.linspace(0, h.size - 1, min(h.size, 250)).astype(int)]
        self.ref_sorted = np.sort(self.ref)
        dh = np.diff(h)
        self.hdiff_mu, self.hdiff_sd = float(dh.mean()), float(dh.std() + 1e-9)
        self.hdiff_var = max(float(dh.var()), 1e-12)
        self.hdiff_sk, self.hdiff_ku = _moms(dh)
        self.h_sign = float(np.mean(np.diff(np.sign(h - self.mu)) != 0))
        self.h_spec = _spec_hilo(h)  # buildH: full-history baseline (was 250-pt decimated ref)
        self.hdvar2 = max(_dvar(h, 2), 1e-12)
        self.hdvar4 = max(_dvar(h, 4), 1e-12)
        self.har1, self.har2 = _ar2(h)
        absh = np.abs((h - self.mu) / self.sd)
        self.habs_mu, self.habs_var = float(absh.mean()), max(float(absh.var()), 1e-12)
        self.habs_ref = absh[
            np.linspace(0, absh.size - 1, min(absh.size, 250)).astype(int)
        ]
        self.habs_ref_sorted = np.sort(self.habs_ref)
        self.edges = np.quantile(h, np.linspace(0, 1, 11))
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        self.refresh, self.heavy_every = refresh, heavy_every
        self.det = OnlineBreakDetector(h)
        self.online: list[float] = []
        self._sum = 0.0  # running sum of online (for O(1) full-series mean/var)
        self._sumsq = 0.0  # running sum of squares of online
        self.win = [0.0] * _N_WIN
        self.ks = self.ad = self.mwu = self.abs_ks = 0.0

    def _st(self, a):
        if len(a) < 2:
            return 0.0, 0.0, 0.0, 0.0
        astd = a.std() + 1e-12
        return (
            (a.mean() - self.mu) / self.sd,
            math.log((a.var() + 1e-12) / self.var_h),
            ((a - a.mean()) ** 3).mean() / astd**3 - self.skew_h,
            ((a - a.mean()) ** 4).mean() / astd**4 - self.kurt_h,
        )

    def update(self, x) -> list[float]:
        x = float(x) if math.isfinite(x) else self.mu
        d = self.det
        d.update(x)
        self.online.append(x)
        n = len(self.online)
        # Running full-series moments, updated every point in O(1). Replaces the
        # per-refresh np.asarray(self.online) + full-array var, which made update()
        # O(L^2) per series and blew up on long (cloud full-set) series.
        self._sum += x
        self._sumsq += x * x
        if n <= 8 or n % self.refresh == 0:
            # Only the last 64 samples feed any windowed feature, so slice the tail
            # (O(1)) instead of rebuilding the whole online array (O(n)).
            w = np.asarray(self.online[-64:], dtype=np.float64)
            wl = w[-16:]
            w32 = w[-32:]
            m64, v64, sk64, ku64 = self._st(w)
            m16, v16, _, _ = self._st(wl)
            # mall/vall == self._st(full online): (mean-mu)/sd, log((var+eps)/var_h),
            # with population var = E[x^2]-E[x]^2 from the running sums.
            if n >= 2:
                mean_all = self._sum / n
                var_all = self._sumsq / n - mean_all * mean_all
                if var_all < 0.0:  # guard tiny negative from float cancellation
                    var_all = 0.0
                mall = (mean_all - self.mu) / self.sd
                vall = math.log((var_all + 1e-12) / self.var_h)
            else:
                mall = vall = 0.0
            m32, v32, _, _ = self._st(w32)
            dq = ((_q(w) - self.qh) / self.sd) if len(w) >= 5 else np.zeros(5)
            acf_w = _acf1(w) - self.acf_h if len(w) >= 4 else 0.0
            slope = (
                float(np.polyfit(np.arange(len(w)), w, 1)[0]) / self.sd
                if len(w) >= 4
                else 0.0
            )
            if len(w) >= 4:
                dw = np.diff(w)
                diff_mean = (dw.mean() - self.hdiff_mu) / self.hdiff_sd
                diff_var = math.log((dw.var() + 1e-12) / self.hdiff_var)
                dsk, dku = _moms(dw)
                diff_skew, diff_kurt = dsk - self.hdiff_sk, dku - self.hdiff_ku
                sign_chg = (
                    float(np.mean(np.diff(np.sign(w - self.mu)) != 0)) - self.h_sign
                )
            else:
                diff_mean = diff_var = diff_skew = diff_kurt = sign_chg = 0.0
            spec = _spec_hilo(w) - self.h_spec if len(w) >= 8 else 0.0
            rank_shift = (
                float(
                    np.mean(np.searchsorted(np.sort(self.ref), w)) / len(self.ref) - 0.5
                )
                if len(w)
                else 0.0
            )
            js = _js(w, self.ref, self.edges) if len(w) >= 8 else 0.0
            dvar2 = (
                math.log(max(_dvar(w, 2), 1e-12) / self.hdvar2) if len(w) > 2 else 0.0
            )
            dvar4 = (
                math.log(max(_dvar(w, 4), 1e-12) / self.hdvar4) if len(w) > 4 else 0.0
            )
            a1, a2 = _ar2(w)
            ar1c, ar2c = a1 - self.har1, a2 - self.har2
            cs = np.cumsum(w - self.mu)
            cumsum_rng = (
                (cs.max() - cs.min()) / (self.sd * math.sqrt(len(w))) if len(w) else 0.0
            )
            absw = np.abs((w - self.mu) / self.sd)
            abs_mean = absw.mean() - self.habs_mu
            abs_var = math.log((absw.var() + 1e-12) / self.habs_var)
            self.win = [
                m64,
                v64,
                sk64,
                ku64,
                m16,
                v16,
                mall,
                vall,
                *dq.tolist(),
                acf_w,
                slope,
                diff_mean,
                diff_var,
                sign_chg,
                spec,
                rank_shift,
                js,
                dvar2,
                dvar4,
                ar1c,
                ar2c,
                m32,
                v32,
                cumsum_rng,
                diff_skew,
                diff_kurt,
                abs_mean,
                abs_var,
                self.abs_ks,
            ]
            if n % self.heavy_every == 0 and len(w) >= 8:
                self.ks = _fast_ks(w, self.ref, self.ref_sorted)
                try:
                    self.ad = float(
                        np.clip(anderson_ksamp([w, self.ref]).statistic, -2, 20)
                    )
                except (ValueError, IndexError):
                    pass
                self.mwu = _fast_mwu(w, self.ref, self.ref_sorted)
                self.abs_ks = _fast_ks(absw, self.habs_ref, self.habs_ref_sorted)
                self.win[-1] = self.abs_ks
        return [
            float(n),
            math.log1p(n),
            max(d._s_pos, d._s_neg),
            max(d._v_pos, d._v_neg),
            max(d._s3_pos, d._s3_neg),
            max(d._s4_pos, d._s4_neg),
            self.ks,
            self.ad,
            self.mwu,
            *self.win,
            (x - self.mu) / self.sd,
        ]


def rich_series_features_v4(historical, online, **kw):
    f = RichFeaturizerV4(historical, **kw)
    return np.asarray([f.update(x) for x in online], dtype=np.float64)
