"""Gene grammar for the evolving raw-view search. A gene = (base, stat, param, window, contrast) -> one causal column
per online point, contrasted with the series' OWN history. Everything is vectorised per series (running sums), so a
genome of <= 8 genes costs a few ms per series. Causal by construction (trailing windows ending at each point)."""

from __future__ import annotations
import numpy as np

BASES = ("x", "absdev", "sign", "rank", "sq", "diff", "innov")
STATS = ("vr", "acf", "mean", "var", "kurt", "runs", "qhi", "qlo")
PARAMS = {
    "vr": (2, 4, 8, 16, 32),
    "acf": (1, 2, 3, 5, 8),
    "mean": (0,),
    "var": (0,),
    "kurt": (0,),
    "runs": (0,),
    "qhi": (0,),
    "qlo": (0,),
}
REF_N = 256  # 2026-09-24 LEAK FIX: fixed history prefix (histories >= 1,000 pts); the old quarter-of-total-length reference encoded the online length
WINDOWS = (16, 32, 64, 128, 256, 512, 1024)
CONTRASTS = ("logratio", "z", "diff", "cusum_up", "cusum_dn", "sr")
# Accumulation contrasts (2026-09-23): z_t = (stat_t - mean_hist) / sd_hist over the ONLINE points, then
#   cusum_up: C_t = max(0, C_{t-1} + z_t - K)   cusum_dn: the same on -z   (Page's CUSUM, started at the online start)
#   sr:       log R_t, R_t = (1 + R_{t-1}) * exp(z_t - 0.5)   (Shiryaev-Roberts on a unit-shift likelihood ratio)
# These integrate evidence ACCUMULATED SINCE a break (the part of the hindsight gap a causal model can still reach).
CUSUM_K = 0.5


def roll_mean(a, w):
    c = np.cumsum(np.r_[0.0, a])
    i = np.arange(1, len(a) + 1)
    lo = np.maximum(0, i - w)
    return (c[i] - c[lo]) / (i - lo)


def roll_var(a, w):
    return np.maximum(roll_mean(a * a, w) - roll_mean(a, w) ** 2, 0.0)


def base_series(name, x_hist, x_on, cache):
    if name in cache:
        return cache[name]
    x = np.r_[x_hist, x_on]
    nh = len(x_hist)
    m = x_hist.mean()
    sd = x_hist.std() + 1e-12
    if name == "x":
        a = (x - m) / sd
    elif name == "absdev":
        a = np.abs(x - m) / sd
    elif name == "sign":
        a = np.sign(x - np.median(x_hist))
    elif name == "rank":
        a = np.searchsorted(np.sort(x_hist), x, side="right") / (nh + 1.0) - 0.5
    elif name == "sq":
        a = ((x - m) / sd) ** 2
    elif name == "diff":
        a = np.r_[0.0, np.diff(x)] / sd
    elif name == "innov":
        h = x_hist - m
        P = 5
        X = np.stack([h[P - k - 1 : nh - k - 1] for k in range(P)], 1)
        phi = np.linalg.lstsq(X, h[P:], rcond=None)[0]
        z = x - m
        e = z.copy()
        for k in range(P):
            e[P:] -= phi[k] * z[P - k - 1 : len(z) - k - 1]
        e[:P] = 0.0
        a = e / sd
    else:
        raise ValueError(name)
    cache[name] = a
    return a


def _stat_roll(a, stat, p, w):
    """trailing-window statistic at every position."""
    if stat == "vr":
        c = np.cumsum(np.r_[0.0, a])
        sq = np.r_[np.zeros(p - 1), c[p:] - c[:-p]]
        return roll_var(sq, w) / (p * roll_var(a, w) + 1e-12)
    if stat == "acf":
        b = np.r_[np.full(p, a[:p].mean()), a[:-p]]
        return (roll_mean(a * b, w) - roll_mean(a, w) * roll_mean(b, w)) / (
            roll_var(a, w) + 1e-12
        )
    if stat == "mean":
        return roll_mean(a, w)
    if stat == "var":
        return roll_var(a, w)
    if stat == "kurt":
        m1, m2, m3, m4 = (roll_mean(a**k, w) for k in (1, 2, 3, 4))
        v = np.maximum(m2 - m1**2, 1e-18)
        return np.clip(
            (m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / v**2 - 3, -50, 50
        )
    if stat == "runs":
        s = (a > np.median(a[:REF_N])).astype(float)
        ch = np.r_[0.0, (np.diff(s) != 0).astype(float)]
        return roll_mean(ch, w)
    if stat in ("qhi", "qlo"):
        # exceedance rate beyond the running series' 90th / below 10th percentile of the first quarter (history proxy)
        ref = a[:REF_N]
        th = np.percentile(ref, 90 if stat == "qhi" else 10)
        e = (a > th) if stat == "qhi" else (a < th)
        return roll_mean(e.astype(float), w)
    raise ValueError(stat)


def gene_column(gene, x_hist, x_on, cache):
    base, stat, p, w, contrast = gene
    a = base_series(base, x_hist, x_on, cache)
    nh = len(x_hist)
    on = slice(nh, nh + len(x_on))
    key = (base, stat, p, w)
    if key not in cache:
        cache[key] = _stat_roll(a, stat, p, w)
    r = cache[key]
    hist = r[w:nh]
    hist = hist[np.isfinite(hist)]
    if len(hist) == 0:
        hist = np.array([0.0])
    v = r[on]
    if contrast == "logratio":
        out = np.log(np.abs(v) + 1e-9) - np.log(np.abs(np.median(hist)) + 1e-9)
    elif contrast == "z":
        out = (v - hist.mean()) / (hist.std() + 1e-9)
    elif contrast == "diff":
        out = v - np.median(hist)
    else:
        z = np.nan_to_num((v - hist.mean()) / (hist.std() + 1e-9), nan=0.0, posinf=0.0, neginf=0.0)
        z = np.clip(z, -10, 10); out = np.empty_like(z); c = 0.0
        if contrast == "sr":
            for i, zi in enumerate(z):
                c = np.logaddexp(0.0, c) + (zi - 0.5); out[i] = c
        else:
            sgn = 1.0 if contrast == "cusum_up" else -1.0
            for i, zi in enumerate(z):
                c = max(0.0, c + sgn * zi - CUSUM_K); out[i] = c
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def genome_view(genome, x_hist, x_on):
    if not genome:
        return np.zeros((len(x_on), 0), np.float32)
    cache = {}
    return np.stack([gene_column(g, x_hist, x_on, cache) for g in genome], 1)


SCALE_FREE_BASES = ("sign", "rank"); SCALE_FREE_STATS = ("vr", "acf", "runs")
SCALE_FREE_P = 0.8   # 2026-09-24: scale / level genes failed the unseen-data proxy hard (-0.18); bias the search toward scale-free ones


def is_scale_free(g):
    return g[0] in SCALE_FREE_BASES or g[1] in SCALE_FREE_STATS


def _random_gene_any(rng):
    st = STATS[rng.integers(len(STATS))]
    return (
        BASES[rng.integers(len(BASES))],
        st,
        int(rng.choice(PARAMS[st])),
        int(rng.choice(WINDOWS)),
        CONTRASTS[rng.integers(len(CONTRASTS))],
    )


def random_gene(rng):
    g = _random_gene_any(rng)
    for _ in range(10):
        if is_scale_free(g) or rng.random() > SCALE_FREE_P: break
        g = _random_gene_any(rng)
    return g


def mutate(gene, rng):
    b, st, p, w, c = gene
    k = rng.integers(5)
    if k == 0:
        b = BASES[rng.integers(len(BASES))]
    elif k == 1:
        st = STATS[rng.integers(len(STATS))]
        p = int(rng.choice(PARAMS[st]))
    elif k == 2:
        p = int(rng.choice(PARAMS[st]))
    elif k == 3:
        w = int(rng.choice(WINDOWS))
    else:
        c = CONTRASTS[rng.integers(len(CONTRASTS))]
    return (b, st, p, w, c)
