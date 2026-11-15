"""CrunchDAO Structural Break Real-Time — consol101-fast deployment (submission #7 candidate).

consol94 = 43 v4 features on the raw series
         + 43 v4 features on the AR-whitened PIT innovation stream
           (RichFeaturizerV4 initialized with the historical innovations)
         + 8 winning calibrated innovation statistics (Deep8)
         = 94 columns, full-length training, tuned LightGBM, leaky-max rho=0.99.
Both v4 views use refresh=1/heavy_every=16. Degrade-don't-raise on budget overrun:
a raise scores ZERO and leaves the live OOS entry on whatever ran last.

THIS IS THE .6298 BUILD = the .6289 build (main.py md5 5c0624b21a00728cb47a65932424108b,
preserved in .6289/) plus exactly two changes: INFER_PARALLELISM 4 -> 12 and refresh 8 -> 1.
Restored 2026-07-28 after the LGB+ET blend scored .6251 (see the blend post-mortem below).
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import collections as _collections
from scipy.special import erf as _erf

from consol_stream import DEEP8_NAMES, Deep8Stream, InnovStream, _fit_ar, _ppf_interp
from vol_stream import VOL_NAMES, EWMAVolStream, VStats7Stream
from ev_stream import EV_NAMES, ExtremeValueStream, batch_ev
from glr_stream import GLR_NAMES, GLRVarStream, batch_glr
from garch_stream import GARCH_NAMES, GARCHVarStream, batch_garch
from gstd_stream import GStdStream, batch_gstd
from trimgap_stream import TRIMGAP_NAMES, TrimGapStream, batch_trimgap
from robust_streams import (
    ARBOCPD_NAMES,
    ArbocpdStream,
    DISTBATTERY_NAMES,
    DistBatteryStream,
    FLIGNER_NAMES,
    FlignerStream,
    SPECBASIS_NAMES,
    SpecBasisStream,
)
from compound_streams import (
    COMPOUND_EXTRA_NAMES,
    BinModeConvStreakStream,
    IDRStdRatioStream,
    MBDDwellStream,
    TPOrderDensityStream,
    batch_compound_extra,
)
from mac_xrate_stream import MACXRateStream
from vr_stream import VrStream, N_COLS as VR_N_COLS
from gene_stream import GenomeStream
from rich_features_v4 import (
    FEATURE_NAMES,
    RichFeaturizerV4,
    rich_series_features_v4,
)

# refresh=1 is BOARD-VERIFIED: .6298 vs refresh=8's .6289 (+0.09 bp) in 56m24s vs 16m05s.
# DO NOT re-tune this locally. A knee sweep (400 train / 200 held out) said refresh=3 was a
# free 2.4x speedup at equal score; it did not replicate and INVERTED with training size:
#   train  400: r8 0.6331 < r1 0.6365 < r3 0.6378   (r3 best)
#   train 3000: r3 0.6291 < r8 0.6304 < r1 0.6314   (r3 WORST, and r8 DOMINATES r3)
# A positive control settled why: the same harness could not reproduce the board's own
# r1 > r8 result (r1 - r8 = +0.0010, CI [-0.0009, +0.0029], straddles zero), i.e. it has no
# power at board effect size. Only the board can rank these arms. See
# sb_opt_local/sim/verify_refresh_power.py.
_IV_KW = dict(refresh=1, heavy_every=16)
_HO_KW = dict(refresh=1, heavy_every=16)

# ⛔ LGB + ExtraTrees BLEND: REFUTED ON THE BOARD 2026-07-28. Do not retry as specified.
# Offline it looked like the best lever available, and it FAILED at deployment scale with a
# clean monotone dose-response in the WRONG direction as training data grew:
#   68k rows   (400 series):   blend - lgb = +0.0062   (the original sweep)
#   507k rows  (3000 series):  blend - lgb = +0.0026   (pre-push screen, all 3 seeds positive)
#   1.68M rows (BOARD, 10000): blend - lgb = -0.0047   (.6251 vs .6298, 1h11m)
# MECHANISM: ET capacity is uncapped in n at fixed min_samples_leaf=200, so the partner model
# gets deeper and more overfit exactly as the training set grows (depth 41-43 / 1.10M nodes
# already at 507k rows). What was measured as a decorrelated partner at 68k rows is a
# different model at 1.68M. The blend arithmetic was sound -- the ET was not the same ET.
# 🚨 THE PROCESS LESSON: a pre-registered LARGE-NEGATIVE screen with margin -0.004 PASSED at
# +0.0026, and reality was -0.0047. So the harness is not merely unable to SELECT small
# effects -- it could not rule out a large negative either, at 3000 training series.
# ⇒ Any future local test must train at DEPLOYMENT SCALE (~9000 series, hold out 1000).
# Both of today's failures share this one root cause: measuring at 400-3000 training series
# for a build that trains on 10000. The refuted build is kept at
# sb_opt_local/main_blend_REFUTED.py with et_flat.py (the 320x ET predict path, still good
# code and reusable if a capacity-capped ET is ever tried).

# Leaky-max decay: s_t = max(p_t, RHO * s_{t-1}).
# Board swept 0.99/0.98/0.97 (+0.055/+0.020 board pts). Measured 2026-07-27 on 6
# honest blocks with a paired bootstrap: the SE of a rho difference is ~0.011-0.037
# board points, so those gaps are ~1.8-2.1 SE, and held-out data ranks the arms in
# the OPPOSITE order. Treat any value in [0.97, 0.995] as equivalent; 0.97 is kept
# only because it is what the 62.8641 run used. Do not spend pushes tuning this.
RHO = 0.97

# Training row subsample. stride=3 is what every board result to date was trained on.
# stride=1 (3x the rows) was previously ruled out as impossible -- it peaked at 14.44 GB on
# the 16 GB runner -- but that ceiling was the row ASSEMBLY, not the data: the matrix itself
# is only 2.44 GB. With the contiguous-block rewrite in _series_features_with_labels the
# assembly peaks at 4.28 GB (measured, 10000 series), so stride=1 now fits with room.
# MEASURED at STRIDE=1, all 10000 series, full train() including the LightGBM fit:
#   train 693s | PEAK RSS 6.30 GB (2.40 GB of which is the local parquet load) | model 3.8 MB
#   => 9.7 GB of headroom on the 16 GB runner, vs the 14.44 GB the old assembly needed.
# ⛔ stride=1 REFUTED ON THE BOARD 2026-07-28: .6286 vs .6298 (-0.12 bp) in 52m24s.
# The 9000-train harness PREDICTED this BEFORE the result was known (-0.0019 vs the board's
# -0.0012) -- its second correct call in a row, and the first made in advance.
# LIKELY MECHANISM, and it is the same one that killed the ET blend: min_child_samples=400 is
# an ABSOLUTE threshold, so tripling row density makes 400 samples span a third of the time
# range and every leaf effectively 3x finer. stride=1 does not just add data, it silently
# triples capacity. See harness/capacity_screen.py for the mcs-scaled re-test.
# ⚠️ Keep the row-assembly rewrite regardless -- it is byte-identical (booster md5
# 10152aa18ef2fdae) and this run came in at 52m24s vs the .6298 build's 56m24s DESPITE 3x the
# training rows, because contiguous blocks avoid pickling ~5M individual arrays through joblib.
STRIDE = 3

IV_NAMES = [f"iv_{n}" for n in FEATURE_NAMES]
ALL_NAMES = (
    list(FEATURE_NAMES)
    + IV_NAMES
    + list(DEEP8_NAMES)
    + list(VOL_NAMES)
    + list(EV_NAMES)
    + list(GLR_NAMES)  # variance-shift GLR-CUSUM (6), banked +0.002 local
    + list(
        GARCH_NAMES
    )  # GARCH-conditional-vol variance CUSUM (6), +0.0036 tax-corr over glr
    + list(TRIMGAP_NAMES)  # build K: mean-minus-trimmed-mean gaps (4), certified
    # build I's prefix_dep_ratio columns were REMOVED: they read +0.0043 locally but
    # cost -0.18 on the board across two independent realizations (#80 unbounded 63.60,
    # #81 bounded 63.59). The family anti-transfers on what it encodes, not its scaling.
    # +0.00162 [+0.00025,+0.00307] arm-128 AND +0.00271 [+0.00031,+0.00513] on the
    # independent sid%5 replication (2026-08-19); appended at END so the TCN's
    # 121-dim mu/sd slice and mlp_track indices stay valid.
)
LATE_TREE_NAMES = list(ALL_NAMES) + list(COMPOUND_EXTRA_NAMES[:12])
TREE_NAMES = list(ALL_NAMES) + list(COMPOUND_EXTRA_NAMES)
STAGE_EXPERT_SPLIT = 64
ROBUST_WEIGHT = 0.0
CHRONOS_WEIGHT = 0.0
# FAST (2026-09-13): when both compound weights are exactly 0 the streams, alpha boosters and the
# Chronos transformer contribute 0*(x) == 0.0 to every score; skip computing them. Output is
# bit-identical (verified: smoke sha equal to the unskipped build on real series).
_SKIP_COMPOUND = (ROBUST_WEIGHT == 0.0 and CHRONOS_WEIGHT == 0.0)
TTA_SIGN = True   # 2026-09-30 VRE1-TTA: sign-flip test-time averaging of the tree+MLP base only (mirror series -> second stream set; average RAW p before the CDF map). Legs (TCN/GRU/VR/evolved) untouched. TTA_SIGN=False must be byte-identical to VRE1.
TTA_LEGS = frozenset({"x", "vr", "e"})   # 2026-09-30 v15: sign-flip TTA extended to the legs. "x" = the EXTRA_LEGS sequence legs, "vr" = the VR
TTA_TREE = True    # 2026-09-30 v16 flags: False = mirror only the MLP (tree score reused). Ledger 09-16: MLP-only read +0.00088, tree-only +0.00023 (local, undecidable)
TTA_W = 0.5        # weight of the mirrored base score; 0.5 = #132 exactly. <0.5 keeps more of the sign convention the data carry (mirrored series score lower solo)
X8_ARCHS = ("tcn",)   # x-leg archs served on 8 phase-interleaved states (T8). ("tcn","gru") = ledger item #2 (GRU served fresh; local -0.0002 +- 0.0016)
TTA_LEG_W = 0.5      # weight of the MIRRORED leg score (x/VR/evolved); 0.5 = v15 exactly (literal 0.5*(a+b) branch keeps the bytes). Audit 09-30: test only 0.35 (= v18b), only if v18b wins.
#   LightGBM leg, "e" = the evolved legs. Each family gets a mirrored state set (fed -x), and each leg's RAW score is averaged with its mirror
#   BEFORE that leg's CDF map (the same rule as the base). frozenset() reproduces submit_v14 byte-for-byte. TCN/GRU legs ship at weight 0.0 -> untouched.
CHRONOS_CADENCE = 128
_CHR_WIN = 256
assert len(ALL_NAMES) == 125

# @crunch/keep:on
INFER_PARALLELISM = 15  # runner = 16 cores / 16 GB; worker peak RSS measured at 237 MB
# (15 x 237 MB = 3.6 GB, RAM is not binding -- cores are). Threads are pinned to 1 per
# worker in infer(), so N workers = N threads; 15/16 leaves ONE core for the parent.
# Predictions are unaffected: series are independent, so redistributing them across more
# workers cannot change any per-series score. BOARD-CONFIRMED at 12: .6289 exactly,
# 26m37s -> 16m05s.
# 🔧 RAISED 12 -> 15 on 2026-08-01. The previous note here said "do NOT raise further --
# 12 -> 16 saves ~1 min", and that WAS correct for the build it was written on: the
# refresh=8, 16m05s build, where only ~5 min of infer remained against ~10-11 min of
# fixed cost. That build is two generations gone. The shipped build is refresh=1 with
# the delta fix: ~2 min fixed + ~45 min train + ~32 min infer = 1h36m (#30). With ~32 min
# of infer to compress, 12 -> 15 is worth ~6 min, not ~1. Ho Yin observed the runner
# capping at 75.5% CPU after train = 12.1/16 cores, i.e. the phase is core-bound and
# scales.
# ⚠️ 15 not 16, deliberately: the parent drives the generator and aggregation, so it needs
# a core. The oversubscription stall in the notes came from UNPINNED BLAS at 64 threads,
# not from worker count -- every layer is hard-pinned to 1 above, so 15 workers = 15
# threads. Revert to 12 (or 4) if a run misbehaves; the score must be UNCHANGED, so any
# score movement means something other than scheduling changed.

# LGB + small-MLP blend at LOW weight. Adopted 2026-07-30 from Katlun's #39, which gained
# +0.14 board pts with 0.20 weight on a 64x32 / 8-epoch sklearn MLP.
# 🔑 THE SHAPE IS THE POINT, and it is the opposite of what we tried before:
#     LGB + ExtraTrees at 70% weight, same model class   -> BOARD -0.47  (ours)
#     LGB + ET at 50/50, same class                      -> BOARD -0.47
#     4 models at EQUAL weight (het_ensemble)            -> null
#     MLP at 0.20, DIFFERENT class, standalone WEAKER    -> BOARD +0.14
#     TabPFN, whole weight curve swept                   -> -0.0049 (a genuine kill)
#   ⇒ low weight on a structurally different component pays; equal/half weight on a
#     similar-class model does not. Our -0.47 confounded THREE errors (70% weight, same class,
#     ET capacity scaling with n) and never varied the weight -- a dose curve is FREE once both
#     models are fit, and we simply never ran one.
# The component is deliberately WEAKER than the tree (solo ~0.625 vs ~0.648) and only ~0.83
# correlated with it; the decorrelation is what pays, not the strength. Training it harder makes
# it worse on BOTH axes (solo 0.6246 -> 0.6121 and blend value falls, e8 > e25 > e60), because it
# overfits exactly as the tree does; 8 epochs works by being implicitly early-stopped.
# WEIGHT 0.20 (was 0.10 for submission #24, which scored .6306 = +0.08 bp over .6298). Our fresh
# folds ranked w=0.20 BELOW w=0.10 (-0.0006 vs +0.0005) -- but that ordering is well inside fold
# noise, and the board then UNDER-shot our w=0.10 prediction (+0.05 predicted, +0.08 actual), so
# the fold ranking between the two weights is not trustworthy. Katlun's board-verified value is
# 0.20, so this tests the one difference between our builds that we measured as ambiguous. Our build already has the three components his .6313 arc credits (garch,
# pmax=10, refresh=1), so his +0.14 should NOT be expected to transfer intact.
# ⚠️ HONEST STANDING: fresh-fold gain is only +0.0005/+0.0006 -- inside noise, UNDER our +0.003
# push bar. Pushed as a free experiment (rank is by BEST, so the banked .6298 cannot be lowered)
# because the shape is board-corroborated on BOTH sides. Do not read a null result as surprising.
# Interesting sub-pattern: the gain concentrates on the HARDEST fold (LGB 0.6263 -> +0.0025) and
# vanishes on easy ones, so a hard test set may benefit more than the fold average implies.
# 🔑 LOG-SCALED MLP INPUT (added 2026-08-01, harness/logscale_mlp.py).
# 20 of the 121 columns are non-negative with skew > 5 -- `kurt_cusum` spans median 1.1 to
# 5.6e9, `skew_cusum` to 3.3e7, `var_cusum` to 2.0e5 -- and only 4 columns were log-scaled.
# After StandardScaler such a column is ~all mass at zero plus a few enormous values, so the MLP
# effectively sees a near-constant column with outliers. Worse, ALL FOUR `*_cusum` columns sit in
# the tracked top-52, so the delta bank was DIFFERENCING nine-order-of-magnitude accumulators.
# A break is a MULTIPLICATIVE change, so the informative quantity is a RATIO -- which a small ReLU
# must learn as division, while log1p(a)-log1p(b) ~= log(a/b) is a DIFFERENCE one linear layer
# represents exactly. Same mechanism that bought +0.023 solo on the raw pyramid (raw_frontier.py).
# log1p, not log: 15.94% of these values are EXACT ZEROS, which log1p sends to 0 with no floor.
# MEASURED, 9 folds, 5 fresh (3000-8000), tree fit once and SHARED so it is untouched:
#   fresh  shipped solo 0.6291 / blend w0.4 +0.0051   ->  log20 solo 0.6320 / w0.4 +0.0070
#   log20 vs shipped best-vs-best +0.0019 on fresh folds, 5/5 folds POSITIVE
#   SEL folds only +0.0001, so it IS fold-set dependent; pooled over 9 it is ~+0.0011.
# Sub-bar (+0.003) but larger than #30 (+0.0006) and #31 (+0.0009), which both gained on the board.
# ⚠️ LightGBM is INVARIANT to monotone transforms, so this CANNOT change the tree -- and it must
# not: `_track` is chosen from the tree's gain and stays identical.
MLP_LOG_COLS = (
    "mean_cusum", "var_cusum", "skew_cusum", "kurt_cusum", "cumsum_rng",
    "iv_mean_cusum", "iv_var_cusum", "iv_skew_cusum", "iv_kurt_cusum",
    "e_cusum_var_k025", "e_cusum_kurt", "e_cusum_skew", "v_cusum_var",
    "v_cusum_var025", "ev_max_run", "ev_abs_run", "ev_std_w", "ev_absmax_w",
    "ev_range_w", "ev_tail_w",
)
MLP_LOG_IDX = np.array([ALL_NAMES.index(_n) for _n in MLP_LOG_COLS], dtype=np.int64)



_TCN_DIL = (1, 2, 4, 8, 16, 32)


class _TcnState:
    """Streaming FeatTCN (third leg, arm-135 lineage; he16-retrained 2026-08-16).
    One step per stride-8 online point; ring buffers == the convs' left zero-padding.
    Verified vs torch to 1.1e-06 on the shipped weights."""

    __slots__ = ("W", "bufs", "ptrs")

    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float32) for k, v in w.items()}
        self.bufs = [np.zeros((2 * d + 1, 64), dtype=np.float32) for d in _TCN_DIL]
        self.ptrs = [0 for _ in _TCN_DIL]

    def step(self, x121):
        W = self.W
        h = W["proj.weight"][:, :, 0] @ x121 + W["proj.bias"]
        for bi, d in enumerate(_TCN_DIL):
            buf, p = self.bufs[bi], self.ptrs[bi]
            buf[p] = h
            n = 2 * d + 1
            cw = W[f"blocks.{bi}.cconv.conv.weight"]
            z = (
                cw[:, :, 0] @ buf[(p - 2 * d) % n]
                + cw[:, :, 1] @ buf[(p - d) % n]
                + cw[:, :, 2] @ buf[p]
                + W[f"blocks.{bi}.cconv.conv.bias"]
            )
            mu = z.mean()
            sd = z.std(ddof=1) + 1e-5
            zn = (z - mu) / sd * W[f"blocks.{bi}.ln.g"][0, :, 0] + W[
                f"blocks.{bi}.ln.b"
            ][0, :, 0]
            zg = 0.5 * zn * (1.0 + _erf(zn / np.float32(1.4142135623730951)))
            h = h + W[f"blocks.{bi}.pw.weight"][:, :, 0] @ zg + W[f"blocks.{bi}.pw.bias"]
            self.ptrs[bi] = (p + 1) % n
        return float(W["head.weight"][0, :, 0] @ h + W["head.bias"][0])


def _mlp_view(feats: np.ndarray) -> np.ndarray:
    """The MLP's view of a feature block: log1p the heavy-tailed accumulator columns.

    🚨 THIS IS THE ONE FUNCTION THAT MUST BE CALLED FROM BOTH `train()` AND `infer()`.
    train() applies it to the full-resolution per-series block BEFORE `batch_delta_rows`, so the
    delta bank becomes a LOG-RATIO bank. infer() applies it to each single row BEFORE
    `dbank.push`, so the streaming bank differences the same logged values. Applying it on only
    one side would recreate exactly the train/infer mismatch that #30 was pushed to fix -- and
    that class of bug is silent: the model still trains, still scores, just worse.
    The TREE never sees this: it is fit on the raw matrix and `_track` comes from its gain.
    """
    out = feats.copy()
    out[..., MLP_LOG_IDX] = np.log1p(np.maximum(out[..., MLP_LOG_IDX], 0.0))
    return out


MLP_WEIGHT = 0.44
# 🔧 0.342 -> 0.44 (2026-08-01, harness/w_at_n14.py). RETUNED AT n=14, NOT INHERITED.
# The seed change MOVED the optimum: more/better-averaged nets make a stronger, more
# LGB-correlated MLP, so the blend carries more of it. Vertex fit, 9 folds, fine grid 0.25-0.60:
#   n=5  -> ALL9 vertex 0.363 [0.266,0.450] | FRESH 0.430 [0.326,0.525]
#   n=14 -> ALL9 vertex 0.444 [0.337,0.549] | FRESH 0.514 [0.388,0.639]
# Shipping the POOLED vertex (0.444 -> 0.44), NOT the fresh one. That is the #35 lesson:
# there I shipped the FRESH vertex 0.434 while the pooled data put that point at -0.00024,
# and the board agreed with pooled (.6347 vs .6348). Read the paired value AT THE POINT BEING
# SHIPPED, not the grid argmax of the more optimistic fold set.
# Expected: FRESH +0.0009 / POOLED +0.0003 over w=0.342 at n=14.
# ⚠️ MORE SEEDS FLATTENS THE w CURVE -- at n=14 fresh reads +0.0080/+0.0081/+0.0082/+0.0081/
# +0.0079 across w 0.43..0.60. The vertex INTERVAL WIDENED with seeds (0.184 -> 0.212 pooled)
# because there is less curvature to locate, not more noise. So 0.44 vs 0.46 vs 0.50 are
# effectively equivalent; what matters is that 0.342 is now clearly BELOW the optimum.
# 🔑 Neat consequence: #35's w=0.434 was PREMATURE at n=5 but is nearly correct at n=14 --
# the seed change moved the optimum to where that push had already put it.
# 🔧 0.30 -> 0.342 on 2026-08-01. NOT a guess and NOT expected to move the score: the 9-fold
# paired w-curve (harness/w_semantics.npy, deployed 5-seed ensemble, infer_matched semantics) is
# CONCAVE over w>=0.20, and a quadratic fit puts its vertex at w=0.342 (bootstrap over folds:
# median 0.338, 90% interval [0.259, 0.405]). The shipped 0.30 sat slightly below it.
# ⚠️ THE PEAK IS WORTH +0.000052 AUC = +0.0052 BOARD POINTS over w=0.30 -- roughly HALF the
# ~0.01 bp run-to-run variance on byte-identical code. So this is "no reason not to", not a gain:
# expect no measurable change, and do NOT read a null as a failure of the analysis.
# Board-confirmed neighbours: #31 shipped w=0.30 and scored .6340 vs #30's .6334 at w=0.20
# (+0.06 bp, ~1x transfer from the +0.0006 measured locally). The 0.20->0.30 step was real; this
# one is inside the noise floor by construction.
# 🔑 NOTE FOR PRETRAINED BUILDS: train() writes MLP_WEIGHT into the model file and infer() reads
# it from there (`model.get("mlp_weight")`). With a shipped model the constant below is used only
# at TRAIN time -- changing it later means REGENERATING the model, not just editing this line.
MLP_TRACK_N = 52  # columns whose backward differences feed the MLP

# --- TCN blend constants (CLOSURE) -------------------------------------------
# W_TCN is a blend HYPERPARAMETER, treated exactly like LGBM_PARAMS / MLP_WEIGHT: it lives in
# source, not in a fitted artifact. 🚨 Its value is taken from the SHIPPED BUNDLE (0.2), NOT from
# tcn16_build_2026-08-16.py, which still reads W_TCN = 0.10. The build script is stale; the
# artifact is the operator that was actually measured. Hardcoding 0.10 would have silently changed
# the served equation while every schema check still passed.
W_TCN = 0.0
W_GRU = 0.0   # G1: fourth leg, GRU over the same stride-8 feature stream; hand-set (local sweep), taken from the base share
# EXTRA_LEGS (discovery pipeline, 2026-09-13): generic sequence legs over the SAME standardized stride-8
# stream, built/served by seqleg.py from a spec. Each entry: {"key", "spec", "w"}; the weight comes out
# of the base share. Empty list == the G1w operator exactly (bit-identical, gated by smoke sha).
EXTRA_LEGS = [{'key': 'x1', 'spec': {'arch': 'tcn', 'hidden': 64, 'blocks': 8, 'layers': 1, 'dropout': 0.1, 'wd': 0.0001, 'epochs': 12, 'seed': 500, 'input': 'features', 'nseeds': 3, 'aug': 'histnull', 'name': 'tcn_h64_b8_d10_wd1e-4_x3_hn_s500'}, 'w': 0.15}, {'key': 'x2', 'spec': {'arch': 'gru', 'hidden': 64, 'blocks': 1, 'layers': 1, 'dropout': 0.2, 'wd': 0.0001, 'epochs': 16, 'seed': 700, 'input': 'features', 'nseeds': 6, 'aug': 'histnull', 'name': 'gru_h64_d20_wd1e-4_e16_x6_hn_s700'}, 'w': 0.2}]
TCN_DIM = 121   # the TCN eats the first 121 of the 125 feature columns (see infer(): arr_base[0][:121])
# VR leg (2026-09-23, selfimprove raw_b3_vratio / vr_leg_stream): a LightGBM on the 125 bank columns + 16 streaming
# Lo-MacKinlay variance-ratio columns (vr_stream.py, the same code that built its training features), CDF-mapped and
# ADDED to the served blend at W_VR: p = (1 - W_VR) * p_blend + W_VR * CDF_vr(p_vr). Local all-t evidence vs the T8
# incumbent: +0.00059 4/5 (seed 500), +0.00067 4/5 (seed 700); vs the same LightGBM without VR +0.00035 5/5 at w .10.
# As tree columns instead of a leg it LOST (-0.00017 2/5), hence a separate leg.
W_VR = 0.10
VR_LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7,
                     bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=500)
VR_ROUNDS, VR_STRIDE = 300, 6
# EVOLVED LEGS (generic, 2026-09-24): each entry adds p = (1 - w) * p + w * CDF_leg(booster(125 bank + 16 VR + genome)),
# applied in order after the VR leg. Built by selfimprove/add_leg.py; the train path rebuilds each leg from this spec.
ELEGS_SPEC = [{'key': 'evo_g05676fix', 'genome': [['innov', 'kurt', 0, 1024, 'sr'], ['innov', 'var', 0, 1024, 'diff'], ['sign', 'vr', 8, 512, 'logratio'], ['diff', 'vr', 8, 128, 'logratio'], ['sign', 'vr', 32, 1024, 'z'], ['innov', 'qhi', 0, 512, 'logratio'], ['sq', 'qhi', 0, 256, 'logratio'], ['innov', 'kurt', 0, 128, 'cusum_up']], 'w': 0.15}]   # [{"key": str, "genome": [[base, stat, param, window, contrast], ...], "w": float}]
Q_GRID = 4096   # CDF-map resolution; q_cb/q_tcn are quantiles on np.linspace(0, 1, Q_GRID)
# Order matters: infer() asserts [b.num_feature() for b in robust_boosters] == [6, 4, 14, 12].
_ALPHA_FILES = (
    "alpha_fligner_prefix_replay_v1.joblib",
    "alpha_arbocpd_causal_v2.joblib",
    "alpha_distbattery.joblib",
    "alpha_specbasis.joblib",
)
REQUIRE_TCN = True  # a bundle without a TCN is a broken bundle, not a quieter one
# 🔬 MLP_HISTNULL (2026-09-15, treeaug_20260914/mlp_histnull_sklearn.py): the MLP partner ALSO trains on the
# history-null pseudo-series (seqleg.histnull_pseudo: the last HISTNULL_PO points of every history with
# n_h >= HISTNULL_MIN_H as a break-free online segment, labels 0), featurised by the same _series_delta_rows
# path at the same stride. The TREE never sees them (tree + histnull read +0.0005 3/5; the MLP +0.00215 5/5
# in the blend, stacking to +0.00455 5/5 with the HB3T15G20 legs). The q_cb closure adds the same rows to each
# fold's MLP fit (a pseudo-series inherits its parent's fold) and is still measured on real rows only.
MLP_HISTNULL = True
# 🪞 MLP_MIRROR_ROWS (2026-09-30, v22 MLPMIR; PLAN_2026-09-30_model_change.md section 3B): the MLP nets ALSO train on
# the sign-mirrored copy of every REAL training series, (id, -h, -o): the _series_delta_rows recipe (same features,
# stride, track) on the mirrored series. That is the mirrored stream set infer() already feeds the MLP under TTA_SIGN
# (`_xh_m = -x_historical`, `_pt_m = -float(point)`, `dbank_m.push(_mlp_view(arr_base_m[0]))`). One precision detail:
# infer() widens the float32 feature row to float64 BEFORE _mlp_view, while _series_delta_rows takes the log1p on the
# float32 block (a pre-existing <=~1e-6 train/serve offset on the REAL rows, left unchanged here on purpose). The
# mirrored rows widen first (_mirror_series_delta_rows), so before float32 storage they equal infer()'s mirrored rows
# EXACTLY (pre-flight mlpmir/preflight_mirror_rows.py: max|d| 0.0, stored float32 rows identical in every cell).
# Labels are the real rows' labels (a sign flip moves no break). The history-null pseudo-series are NOT mirrored
# (histnull unchanged). The TREE never sees mirrored rows, and mlp_track still comes from the tree's gain. The q_cb
# closure adds the mirrored rows of each fold's TRAINING series only and still scores real rows only, so q_cb is
# quantiles of real OOF rows. False reproduces v16 exactly.
MLP_MIRROR_ROWS = True
# ⚡ MLP_FIT_PARALLEL (2026-09-30, v22 build time; train side only): the MLP_SEEDS nets are fit in parallel worker
# processes with ONE BLAS thread each, instead of one after another with every BLAS thread. The nets are tiny
# (489->64->32, batch 4096), so BLAS threading buys almost nothing (dev: 2.2 s with 16 threads vs 2.7 s with 1 per
# 200k-row fit; the 3090's 30-thread fits ran ~3x slower still) while 14 independent fits scale with cores. Same rows,
# scaler, seeds and hyperparameters, results in seed order; they differ from the sequential path only by BLAS summation
# order (float rounding). False = the v16 sequential loop exactly (bit-identical closure, verified).
MLP_FIT_PARALLEL = True
# #39's OWN hyperparameters, which differ from sklearn defaults in ways that matter: batch 4096
# (default 200) and lr 5e-4 (default 1e-3) give ~2.9k gradient steps instead of ~60k, i.e. far
# LESS training -- consistent with our own sweep where e8 > e25 > e60 on both solo and blend value.
MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    max_iter=8,
    alpha=2e-4,
    batch_size=4096,
    learning_rate_init=5e-4,
    random_state=42,
)
# Seed list for the averaged MLP ensemble -- harness SEED_LIST verbatim (mlp_seed_ens.py:60):
# the shipped 42 leads every prefix, so n_seed=1 reproduces the previous build exactly.
MLP_SEEDS = (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013)
# 🔧 5 -> 14 seeds (2026-08-01, harness/seed_curve_log.py). THE LARGEST LOCAL EFFECT MEASURED
# ON THIS PROJECT. Paired within-fold vs n=5, 9 folds, at w=0.434:
#   n=8 +0.0007 (9/9) | n=10 +0.0008 (9/9) | n=14 +0.0011 SE 0.0001 (9/9) | n=20 +0.0009 (9/9)
#   vertex fit n=14.57, worth +0.0012 AUC = +0.12 bp; bootstrap 90% CI [13.88, 15.61]
# Every n from 6..20 is positive; from n=8 it is 9-of-9 folds. t ~ 11.
# 🔑 THE PRE-LOG CURVE MISSED THIS ENTIRELY: it put n=10 at +0.00005 pooled (i.e. nothing).
# On the LOG build the same comparison is +0.0008, 9/9 -- a 16x difference. log-scaling and
# seed averaging INTERACT: the better-conditioned input makes each net's errors more
# independent, so averaging recovers more. This is why measurements must be retaken on the
# build that actually ships (the same lesson as the delta-bank fix).
# ⚠️ The top is a PLATEAU (n=12..20 all +0.0009..+0.0011, within SE). 14 is the paired argmax
# and sits inside the bootstrap CI; 13 or 16 would be indistinguishable. What is solid is
# that n>=8 clearly beats n=5.
# COST, accepted under the ship-every-optimum directive: ~16.4 us/pt per seed of ~1350 at
# infer, so 5->14 is ~+11% inference and grows the SHIPPED model ~4.74 -> ~13 MB.
# 🔑 THE MLP EATS A TEMPORAL DELTA BANK, THE TREE DOES NOT. Input = 121 current features +
# {1,2,4,8,16,32,64}-step backward deltas of the top-52-by-gain columns = 485 dims. Our first
# reproduction of #39 copied its ARCHITECTURE but not its INPUT (121 dims, no deltas) -- and since
# the tree also sees only the current vector, a delta-free MLP is decorrelated from it by model
# class alone, whereas a delta-fed one is decorrelated by INFORMATION too.
# Fresh folds, deployment scale, at w=0.20:  no deltas +0.0011 | field 16 +0.0018 |
#   field 64 +0.0027 | field 128 +0.0007 | field 256 +0.0016  -> clean interior optimum at 64.
# Weight also has an interior optimum: 0.2/0.3 +0.0027, 0.4 +0.0023, 0.5 +0.0015.

LGBM_PARAMS = dict(
    n_estimators=600,  # fewtrees: 1200 over-trains; 600 wins cross-block mean+min+stability
    learning_rate=0.015,
    num_leaves=63,
    max_depth=6,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=20.0,  # fewtrees regularization
    min_child_samples=400,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _init_streams(x_historical):
    """Pre-initialize all streaming objects for one series (expensive, parallelizable)."""
    fv = RichFeaturizerV4(x_historical, **_HO_KW)
    ins = InnovStream(x_historical)
    gs = GStdStream(ins.e_hist)
    iv = RichFeaturizerV4(gs.uh, **_IV_KW)
    d8 = Deep8Stream()
    ews = EWMAVolStream(ins.e_hist)
    vs = VStats7Stream(ews.v_hist, ews.lam)
    evs = ExtremeValueStream(x_historical)
    g = GLRVarStream(x_historical)
    ga = GARCHVarStream(x_historical)
    pds = TrimGapStream(x_historical)
    mac = MACXRateStream(x_historical)
    mbd = MBDDwellStream(x_historical)
    tp = TPOrderDensityStream(x_historical)
    idr = IDRStdRatioStream(x_historical)
    bin_mode = BinModeConvStreakStream(x_historical)
    return fv, ins, iv, d8, ews, vs, evs, g, ga, gs, pds, mac, mbd, tp, idr, bin_mode


def _innovation_stream_batch(h_arr: np.ndarray, o_arr: np.ndarray):
    """Vectorized (fast) PIT + AR whitening — identical math to InnovStream."""
    h = h_arr[np.isfinite(h_arr)]
    if h.size < 8:
        h = np.zeros(8)
    o = np.where(np.isfinite(o_arr), o_arr, float(np.median(h)))
    nh = h.size
    sh = np.sort(h)

    def pit(x):
        u = np.searchsorted(sh, x, side="right") / (nh + 1)
        return _ppf_interp(np.clip(u, 1.0 / (nh + 1), nh / (nh + 1)))

    zh, zo = pit(h), pit(o)
    p, phi, sigma, e_hist = _fit_ar(zh, pmax=10)
    L = o.size
    if p > 0:
        seed = zh[-p:] if zh.size >= p else np.zeros(p)
        zseq = np.concatenate([seed, zo])
        pred = np.zeros(L)
        for i in range(p):
            pred += phi[i] * zseq[p - 1 - i : p - 1 - i + L]
        resid = zo - pred
    else:
        resid = zo
    return e_hist, resid / sigma


def _series_features(x_hist, x_online) -> np.ndarray:
    """Batch 94-col feature matrix for one series (train-time fast path)."""
    h = np.asarray(x_hist, dtype=np.float64)
    o = np.asarray(x_online, dtype=np.float64)
    fv = np.asarray(rich_series_features_v4(h, o, **_HO_KW), dtype=np.float32)
    # Innovations come from InnovStream itself, stepped sequentially -- the SAME
    # object infer() uses. The old `_innovation_stream_batch` fast path computed
    # the AR reconstruction in a different summation order; innovations agreed only
    # to ~1e-14, and the rank statistics in the iv view (ad/ks/mwu/rank_shift/abs_ks)
    # are step functions, so that round-off flipped order comparisons and skewed
    # those columns on ~12.5% of training rows. Sequential stepping makes the batch
    # and stream paths identical by construction; the cost is negligible because we
    # already loop over `e` below for Deep8/vol.
    _ins = InnovStream(h)
    e_hist = np.asarray(_ins.e_hist, dtype=np.float64)
    e = np.array([_ins.step(float(p)) for p in o], dtype=np.float64)
    uh, u = batch_gstd(e_hist, e)
    iv = np.asarray(rich_series_features_v4(uh, u, **_IV_KW), dtype=np.float32)
    d8 = Deep8Stream()
    win = np.asarray([d8.step(float(et)) for et in e], dtype=np.float32)
    ews = EWMAVolStream(e_hist)
    vs = VStats7Stream(ews.v_hist, ews.lam)
    vol = np.asarray(
        [vs.step(ews.step(float(et)), ews.sigma) for et in e], dtype=np.float32
    )
    ev = batch_ev(
        h, o
    )  # extreme-value-8, raw-series AR whitening (matches stream in infer)
    glr = batch_glr(
        h, o
    )  # glr-var-6, raw-series AR whitening (bit-identical to GLRVarStream in infer)
    garch = batch_garch(
        h, o
    )  # garch-var-6, GARCH-conditional-vol standardized (bit-identical to GARCHVarStream)
    pdr = batch_trimgap(h, o)  # build K: trimgap-4 (bit-identical to TrimGapStream, 2e-17)
    assert (
        fv.shape[0]
        == iv.shape[0]
        == win.shape[0]
        == vol.shape[0]
        == ev.shape[0]
        == glr.shape[0]
        == garch.shape[0]
        == pdr.shape[0]
        == len(o)
    ), "row misalignment"
    out = np.hstack([fv, iv, win, vol, ev, glr, garch, pdr])
    assert out.shape[1] == len(ALL_NAMES), "column count mismatch"
    assert np.isfinite(out).all(), "non-finite features"
    return out


def _series_features_with_labels(args):
    """Compute features + labels for one series (picklable for joblib).

    Returns ONE CONTIGUOUS BLOCK, not a list of rows. The old version appended
    `feats[t]` per row, so joblib shipped ~5M separate 1-D arrays back to the parent --
    each carrying ~112 B of object header plus malloc padding on 484 B of payload, i.e.
    ~3.2 GB of overhead for 2.44 GB of numbers. Measured at stride=1 over 10000 series:
    peak RSS 10.82 GB -> 4.28 GB, with the assembled matrix md5-identical (9a25f58f6ff9).
    That is what makes stride=1 fit the 16 GB runner; see harness/mem_probe.py.
    """
    _dataset_id, x_hist, x_online, tau, stride = args
    feats = _series_features(x_hist, x_online)
    idx = np.arange(0, len(feats), stride)
    lab = np.zeros(len(idx), np.int8)
    if tau is not None:
        lab[idx >= tau] = 1
    return np.ascontiguousarray(feats[idx]), lab


def _series_alpha_rows(args):
    """CLOSURE: the four robust view matrices for one series (picklable for joblib).

    Uses the SAME certified stream objects, initialised the same way and stepped on the same
    quantities as infer() (main.py:938-958): Fligner on the raw point, the other three on the AR
    innovation. Streams are stateful, so every online point is stepped and the stride is applied
    only when collecting rows -- exactly as _series_features_with_labels does.
    """
    _sid, x_hist, x_online, tau, stride = args
    _fv, ins, _iv, _d8, _ews, _vs, _evs, _g, _ga, _gs, _pds, _mac, _mbd, _tp, _idr, _bm = (
        _init_streams(x_hist)
    )
    fligner_stream = FlignerStream(x_hist)
    arbocpd_stream = ArbocpdStream(ins.e_hist)
    distbattery_stream = DistBatteryStream(ins.e_hist)
    specbasis_stream = SpecBasisStream(ins.e_hist)

    rows = [[], [], [], []]
    for point in x_online:
        et = ins.step(point)
        rows[0].append(fligner_stream.step(point))
        rows[1].append(arbocpd_stream.step(et))
        rows[2].append(distbattery_stream.step(et))
        rows[3].append(specbasis_stream.step(et))

    n = len(rows[0])
    idx = np.arange(0, n, stride)
    lab = np.zeros(len(idx), np.int8)
    if tau is not None:
        lab[idx >= tau] = 1
    # np.asarray([]) is 1-D, which would break the parent's concatenate against (n, width)
    # blocks. A zero-online series is degenerate but must not take the build down with it.
    _widths = (len(FLIGNER_NAMES), len(ARBOCPD_NAMES),
               len(DISTBATTERY_NAMES), len(SPECBASIS_NAMES))
    mats = tuple(
        np.ascontiguousarray(np.asarray(r, dtype=np.float32)[idx])
        if n else np.zeros((0, w), dtype=np.float32)
        for r, w in zip(rows, _widths)
    )
    return mats, lab


def _series_delta_rows(args):
    """Pass-2 worker: the MLP's training input for one series, INFER-MATCHED.

    DELTA-BANK FIX (2026-07-31, Ho Yin's find, delta_fix_test.py). The old path ran
    batch_delta_rows over the CONCATENATED STRIDED matrix, so a lag-k delta (a) spanned
    3k original steps, not k, and (b) crossed series boundaries at each start (~11% of
    cells impossible at inference). infer()'s DeltaBank is per-series at full resolution.
    This worker computes deltas at FULL resolution per series and strides AFTERWARD --
    `infer_matched` in his harness, the only variant identical to what DeltaBank emits,
    and the convention every local measurement (incl. the seed-ens +0.0030) used.
    Cost: recomputes the series' features once more (pass 1 discarded them stride-first).
    """
    _dataset_id, x_hist, x_online, stride, track = args
    from mlp_flat import batch_delta_rows

    feats = _mlp_view(_series_features(x_hist, x_online))   # <- LOG-SCALED (see _mlp_view)
    rows = batch_delta_rows(feats, track)
    return rows[np.arange(0, len(rows), stride)].astype(np.float32)


def _histnull_delta_rows(datasets, stride, track):
    """MLP_HISTNULL: the delta rows of the history-null pseudo-series (labels all 0) through the SAME
    _series_delta_rows path, plus each row's PARENT series index (the q_cb closure keeps a pseudo-series
    in its parent's fold). Empty when no history is long enough (the fixture)."""
    import seqleg as _sl
    from mlp_flat import DELTA_LAGS as _DL

    jobs = []
    for i, (_sid, h, _o, _tau) in enumerate(datasets):
        pp = _sl.histnull_pseudo(h)
        if pp is not None:
            jobs.append((i, pp))
    width = len(ALL_NAMES) + len(track) * len(_DL)
    if not jobs:
        return np.empty((0, width), dtype=np.float32), np.empty(0, dtype=np.int64)
    blocks = joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(_series_delta_rows)((i, pp[0], pp[1], stride, track)) for i, pp in jobs)
    parent = np.concatenate([np.full(b.shape[0], i, dtype=np.int64) for (i, _), b in zip(jobs, blocks)])
    rows = np.concatenate(blocks, 0).astype(np.float32, copy=False)
    assert rows.shape == (len(parent), width), (rows.shape, len(parent), width)
    return rows, parent


def _mirror_series(x_hist, x_online):
    """MLP_MIRROR_ROWS: the sign-mirrored series (-h, -o), with infer()'s negation convention: a float64 negation
    (`np.negative(np.asarray(x_historical, dtype=np.float64))`, `-float(point)`), which is exact (-0.0 stays -0.0)."""
    return (np.negative(np.asarray(x_hist, dtype=np.float64)),
            np.negative(np.asarray(x_online, dtype=np.float64)))


def _mirror_series_delta_rows(args, dtype=np.float32):
    """MLP_MIRROR_ROWS worker: the MLP delta rows of ONE series' sign mirror (id, -h, -o). The _series_delta_rows
    recipe on the mirrored series, except that the float32 feature block is widened to float64 BEFORE _mlp_view --
    the order infer() uses (arr_base_m = float32 row -> float64 -> _mlp_view -> DeltaBank in float64). The rows then
    equal infer()'s mirrored rows exactly; the only remaining difference is the float32 storage (`dtype`) of the
    training matrix. `dtype=np.float64` returns the pre-storage rows (used by the pre-flight check)."""
    _dataset_id, x_hist, x_online, stride, track = args
    from mlp_flat import batch_delta_rows

    h_m, o_m = _mirror_series(x_hist, x_online)
    feats = _mlp_view(_series_features(h_m, o_m).astype(np.float64))   # widen first, as infer() does
    rows = batch_delta_rows(feats, track)
    return rows[np.arange(0, len(rows), stride)].astype(dtype)


def _mirror_delta_rows(datasets, stride, track, keep=None):
    """MLP_MIRROR_ROWS: the MLP delta rows of the mirrored series (id, -h, -o) via _mirror_series_delta_rows (same
    stride, same track), in dataset order, for the series with keep[i] true (every series when keep is None). A
    mirrored series has exactly its real series' stride rows, so its labels are the real rows' labels."""
    from mlp_flat import DELTA_LAGS as _DL

    idx = [i for i in range(len(datasets)) if keep is None or bool(keep[i])]
    counts = [len(range(0, len(datasets[i][2]), stride)) for i in idx]
    n = int(sum(counts))
    out = np.empty((n, len(ALL_NAMES) + len(track) * len(_DL)), dtype=np.float32)

    def _jobs():   # the worker mirrors the series itself (same inputs as the real pass)
        for i in idx:
            _id, h, o, _tau = datasets[i]
            yield joblib.delayed(_mirror_series_delta_rows)((_id, h, o, stride, track))

    try:
        blocks = joblib.Parallel(n_jobs=-1, prefer="processes", return_as="generator")(_jobs())
    except TypeError:
        blocks = joblib.Parallel(n_jobs=-1, prefer="processes")(_jobs())
    at = 0
    for blk, k_exp in zip(blocks, counts):
        k = blk.shape[0]
        assert k == k_exp, f"mirrored series rows {k} != real rows {k_exp}"
        out[at:at + k] = blk
        at += k
        del blk
    del blocks
    assert at == n, f"mirror row mismatch: {at} != {n}"
    return out


def _fit_one_mlp(Xs, y, seed):
    """MLP_FIT_PARALLEL worker: one MLP_SEEDS net (MLP_PARAMS, random_state=seed) pinned to one BLAS thread."""
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return MLPClassifier(**dict(MLP_PARAMS, random_state=seed), early_stopping=False).fit(Xs, y)


def _fit_mlp_seeds(Xs, y):
    """The fitted MLP_SEEDS nets on the same scaled rows, in seed order. MLP_FIT_PARALLEL: one worker process per
    seed (joblib memmaps Xs once, read-only, for all workers); False: the v16 sequential loop."""
    if not MLP_FIT_PARALLEL:
        from sklearn.neural_network import MLPClassifier

        return [MLPClassifier(**dict(MLP_PARAMS, random_state=seed), early_stopping=False).fit(Xs, y)
                for seed in MLP_SEEDS]
    n_jobs = max(1, min(len(MLP_SEEDS), os.cpu_count() or 1))
    return joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(_fit_one_mlp)(Xs, y, seed) for seed in MLP_SEEDS)


def _fit_mlp_ship(datasets, stride, track, ya, progress=None):
    """The SHIPPED MLP ensemble (list of flattened nets). train()'s MLP block, moved here verbatim so a build driver
    can refit ONLY the MLP against a shipped bundle's mlp_track (tree untouched). Training rows, in order: the real
    delta rows (labels ya), the history-null pseudo-series (MLP_HISTNULL, labels 0), then the mirrored real series
    (MLP_MIRROR_ROWS, labels ya again). With MLP_MIRROR_ROWS False this is the v16 ship fit exactly."""
    from sklearn.preprocessing import StandardScaler

    from mlp_flat import DELTA_LAGS, flatten_mlp

    _t0 = time.perf_counter()
    n_rows = int(len(ya))
    # DELTA-BANK FIX: per-series full-resolution deltas, strided after (see
    # _series_delta_rows). The streamed f32 assembly also removes
    # the old 6.5 GB float64 transient that batch_delta_rows(Xa) created here.
    _Xd = np.empty((n_rows, len(ALL_NAMES) + len(track) * len(DELTA_LAGS)),
                   dtype=np.float32)
    _d_args = [(_id, h, o, stride, track) for _id, h, o, _tau in datasets]
    try:
        _dblocks = joblib.Parallel(n_jobs=-1, prefer="processes",
                                   return_as="generator")(
            joblib.delayed(_series_delta_rows)(a) for a in _d_args)
    except TypeError:
        _dblocks = joblib.Parallel(n_jobs=-1, prefer="processes")(
            joblib.delayed(_series_delta_rows)(a) for a in _d_args)
    _at = 0
    for _blk in _dblocks:
        _k = _blk.shape[0]
        _Xd[_at:_at + _k] = _blk
        _at += _k
        del _blk
    del _dblocks
    assert _at == n_rows, f"delta row mismatch: {_at} != {n_rows}"
    _ya_mlp = ya
    if MLP_HISTNULL:
        # MLP_HISTNULL: the pseudo-series join the MLP's rows (labels 0). The tree above never sees them.
        _Xh, _ = _histnull_delta_rows(datasets, stride, track)
        _Xd = np.concatenate([_Xd, _Xh], 0)
        _ya_mlp = np.concatenate([ya, np.zeros(_Xh.shape[0], dtype=np.int8)])
        del _Xh
    if MLP_MIRROR_ROWS:
        # MLP_MIRROR_ROWS: every real series once more, sign-mirrored, same labels (histnull rows are not mirrored).
        _Xm = _mirror_delta_rows(datasets, stride, track)
        assert _Xm.shape[0] == n_rows, f"mirror row mismatch: {_Xm.shape[0]} != {n_rows}"
        _Xd = np.concatenate([_Xd, _Xm], 0)
        _ya_mlp = np.concatenate([_ya_mlp, ya])
        del _Xm
    if progress is not None:
        progress(f"ship MLP rows: real {n_rows} + histnull/mirror {_Xd.shape[0] - n_rows} = {_Xd.shape[0]} "
                 f"(delta passes {time.perf_counter() - _t0:.1f}s)")
    _sc = StandardScaler().fit(_Xd)
    _Xs = _sc.transform(_Xd)
    del _Xd
    # SEED-AVERAGED MLP -- the a-priori cell from harness/mlp_seed_ens.py (n_seed=5,
    # w=0.20 unchanged, SEED_LIST with the shipped 42 first). LightGBM untouched;
    # fresh-fold mean +0.0030. Same scaler and same rows for every net: the seed is
    # the ONLY thing that varies. (MLP_FIT_PARALLEL: the seeds are fit in parallel, results in seed order.)
    _flats = [flatten_mlp(_m, _sc) for _m in _fit_mlp_seeds(_Xs, _ya_mlp)]
    del _Xs, _sc, _ya_mlp
    return _flats


# --- CLOSURE: out-of-fold champion predictions, the input to the q_cb CDF map ------------
# q_cb is quantiles of the OUT-OF-FOLD tree+MLP blend (main.py applies it to exactly that
# quantity). Rebuilding it honestly needs a series-grouped K-fold loop: every row of a series
# lands in one fold, so no series contributes to the model that scores it.
#
# K = 5 matches tcn16_build_2026-08-16.py's own 5-fold OOF. Katlun 2026-09-10: "cost is not a
# blocker" -- the 2-fold variant was only ever a compute compromise.
#
# Deliberately self-contained: it duplicates the fit sequence rather than refactoring train(),
# whose body is certified bit-for-bit against v1 and is not worth disturbing for this.
Q_CB_FOLDS = 5
Q_CB_FOLD_SEED = 2000   # the arm-128 partition seed used by the original build


def _q_cb_fold_of_series(n_series):
    """Series -> closure fold: the arm-128 partition (seed Q_CB_FOLD_SEED), unchanged."""
    rng = np.random.default_rng(Q_CB_FOLD_SEED)
    order = rng.permutation(n_series)
    fold_of_series = np.empty(n_series, dtype=np.int64)
    for f, chunk in enumerate(np.array_split(order, Q_CB_FOLDS)):
        fold_of_series[chunk] = f
    return fold_of_series


def _fold_mlp_scores(X_tr, y_tr, X_te):
    """Closure: mean P(1) of the MLP_SEEDS nets fit on (X_tr, y_tr) behind one StandardScaler, scored on X_te."""
    from sklearn.preprocessing import StandardScaler as _SS

    sc = _SS().fit(X_tr)
    Xs_tr, Xs_te = sc.transform(X_tr), sc.transform(X_te)
    p_mlp = np.zeros(X_te.shape[0], dtype=np.float64)
    for m in _fit_mlp_seeds(Xs_tr, y_tr):   # seed order, as the v16 loop
        p_mlp += m.predict_proba(Xs_te)[:, 1]
    p_mlp /= len(MLP_SEEDS)
    del Xs_tr, Xs_te, sc
    return p_mlp


def _oof_champion_predictions(datasets, stride, feats=None, progress=None, with_ctrl=False):
    """Series-grouped K-fold OOF predictions of the w_lgb*tree + w_mlp*mean(MLP) blend.

    MLPMIR: returns (oof_ctrl, oof_new). oof_new is the served recipe: each fold's MLP ALSO trains on the mirrored
    rows of that fold's TRAINING series (MLP_MIRROR_ROWS; a held-out series contributes no row, real or mirrored, to
    the model that scores it). Scored rows are real rows only, so q_cb describes the served (real) distribution.
    oof_ctrl shares the fold tree, track, real and histnull rows but has NO mirrored rows -- the v16 closure arm,
    exactly paired with oof_new. with_ctrl: False -> oof_ctrl is None; True -> every fold; a collection of fold ids ->
    only those folds (NaN elsewhere; a pairing probe against a ctrl computed elsewhere). `feats=(Xa, ya)` reuses an
    already-assembled stride matrix of exactly these datasets (same rows, same order)."""
    import numpy as _np
    import pandas as _pd
    from lightgbm import LGBMClassifier as _LGBM

    from mlp_flat import DELTA_LAGS as _DL

    counts = [len(range(0, len(o), stride)) for _, _, o, _ in datasets]
    n_rows = int(sum(counts))
    row_series = _np.repeat(_np.arange(len(datasets)), counts)

    if feats is None:
        # features once -- they do not depend on any fit
        args = [(_id, h, o, tau, stride) for _id, h, o, tau in datasets]
        Xa = _np.empty((n_rows, len(ALL_NAMES)), dtype=_np.float32)
        ya = _np.empty(n_rows, dtype=_np.int8)
        at = 0
        for blk, lab in joblib.Parallel(n_jobs=-1, prefer="processes")(
                joblib.delayed(_series_features_with_labels)(a) for a in args):
            k = blk.shape[0]
            Xa[at:at + k] = blk
            ya[at:at + k] = lab
            at += k
        assert at == n_rows
    else:
        Xa, ya = feats
        assert Xa.shape == (n_rows, len(ALL_NAMES)) and ya.shape == (n_rows,), "feats do not match datasets"

    fold_of_series = _q_cb_fold_of_series(len(datasets))
    row_fold = fold_of_series[row_series]

    oof = _np.full(n_rows, _np.nan, dtype=_np.float64)
    ctrl_folds = set(range(Q_CB_FOLDS)) if with_ctrl is True else {int(f) for f in (with_ctrl or ())}
    oof_ctrl = _np.full(n_rows, _np.nan, dtype=_np.float64) if ctrl_folds else None
    for f in range(Q_CB_FOLDS):
        tr = row_fold != f
        te = ~tr
        if not te.any() or len(_np.unique(ya[tr])) < 2:
            continue
        _tf0 = time.perf_counter()
        # Same thread-oversubscription guard as alpha_build: LGBM_PARAMS carries n_jobs=-1, which
        # on a small fixture spends its time in OpenMP barriers rather than on trees. Scaled to the
        # work so the full-scale path keeps every core.
        _p = dict(LGBM_PARAMS, n_jobs=(-1 if int(tr.sum()) >= 100_000 else 4))
        clf = _LGBM(**_p).fit(
            _pd.DataFrame(Xa[tr], columns=ALL_NAMES, copy=False), ya[tr])
        p_tree = clf.predict_proba(
            _pd.DataFrame(Xa[te], columns=ALL_NAMES, copy=False))[:, 1]

        # track columns come from THIS fold's tree -- choosing them on all data would leak
        gain = clf.booster_.feature_importance(importance_type="gain")
        track = _np.sort(_np.argsort(-gain)[:MLP_TRACK_N]).astype(_np.int64)
        _tf1 = time.perf_counter()

        Xd = _np.empty((n_rows, len(ALL_NAMES) + len(track) * len(_DL)), dtype=_np.float32)
        at = 0
        for blk in joblib.Parallel(n_jobs=-1, prefer="processes")(
                joblib.delayed(_series_delta_rows)((i, h, o, stride, track))
                for i, h, o, _t in datasets):
            k = blk.shape[0]
            Xd[at:at + k] = blk
            at += k
        assert at == n_rows

        X_tr, y_tr = Xd[tr], ya[tr]
        if MLP_HISTNULL:
            # MLP_HISTNULL: this fold's MLP also trains on the pseudo-series whose PARENT series is in its
            # training folds (labels 0). Scored rows stay real: q_cb describes the served distribution.
            Xh, parent = _histnull_delta_rows(datasets, stride, track)
            keep = fold_of_series[parent] != f
            X_tr = _np.concatenate([X_tr, Xh[keep]], 0)
            y_tr = _np.concatenate([y_tr, _np.zeros(int(keep.sum()), dtype=_np.int8)])
            del Xh, parent, keep
        X_te = Xd[te]
        del Xd
        _tf2 = time.perf_counter()
        if f in ctrl_folds:
            oof_ctrl[te] = (1.0 - MLP_WEIGHT) * p_tree + MLP_WEIGHT * _fold_mlp_scores(X_tr, y_tr, X_te)
        _tf3 = time.perf_counter()
        n_mir = 0
        if MLP_MIRROR_ROWS:
            # MLP_MIRROR_ROWS: the mirrored rows of this fold's TRAINING series only (same labels as their real rows,
            # which are exactly ya[tr] in the same series order). Histnull rows are not mirrored.
            Xm = _mirror_delta_rows(datasets, stride, track, keep=(fold_of_series != f))
            n_mir = int(Xm.shape[0])
            assert n_mir == int(tr.sum()), f"fold {f}: mirror rows {n_mir} != training real rows {int(tr.sum())}"
            X_tr = _np.concatenate([X_tr, Xm], 0)
            y_tr = _np.concatenate([y_tr, ya[tr]])
            del Xm
        _tf4 = time.perf_counter()
        p_mlp = _fold_mlp_scores(X_tr, y_tr, X_te)
        n_fit = int(X_tr.shape[0])
        del X_tr, y_tr, X_te

        oof[te] = (1.0 - MLP_WEIGHT) * p_tree + MLP_WEIGHT * p_mlp
        if progress is not None:
            progress(f"closure fold {f}: tree {_tf1 - _tf0:.1f}s, delta+histnull {_tf2 - _tf1:.1f}s, "
                     f"ctrl MLP {_tf3 - _tf2:.1f}s, mirror delta {_tf4 - _tf3:.1f}s, "
                     f"MLP {time.perf_counter() - _tf4:.1f}s; fit rows {n_fit} (mirror {n_mir}), "
                     f"test rows {int(te.sum())}")

    assert _np.isfinite(oof).all(), "OOF has unscored rows -- fold partition is not a cover"
    if ctrl_folds == set(range(Q_CB_FOLDS)):
        assert _np.isfinite(oof_ctrl).all(), "OOF (ctrl) has unscored rows -- fold partition is not a cover"
    return oof_ctrl, oof


def _vr_series_rows(args):
    """VR-leg training rows for one series: 125 bank columns + 16 streaming VR columns at t % VR_STRIDE == 0."""
    _id, h, o, tau = args
    F = _series_features(h, o); V = VrStream(h); t = np.arange(len(o))
    Vr = np.stack([V.step(float(v)) for v in o]).astype(np.float32) if len(o) else np.empty((0, VR_N_COLS), np.float32)
    keep = t % VR_STRIDE == 0
    y = np.zeros(len(o), np.int8)                       # same convention as _series_features_with_labels
    if tau is not None:
        y[t >= tau] = 1
    X = np.nan_to_num(np.concatenate([F.astype(np.float32), Vr], 1)[keep], nan=0.0, posinf=0.0, neginf=0.0)
    return X, y[keep]


def _eleg_series_rows(args):
    """evolved-leg rows: VR rows (125 + 16 VR) + this leg's genome, at t % VR_STRIDE == 0."""
    (_id, h, o, tau), genome = args[:4], args[4]
    X, y = _vr_series_rows((_id, h, o, tau)); S = GenomeStream(genome, h)
    E = np.stack([S.step(float(v)) for v in o]).astype(np.float32) if len(o) else np.empty((0, len(genome)), np.float32)
    keep = np.arange(len(o)) % VR_STRIDE == 0
    return np.nan_to_num(np.concatenate([X, E[keep]], 1), nan=0.0, posinf=0.0, neginf=0.0), y


def _train_eleg(datasets, genome):
    import lightgbm as _lgb
    rows = joblib.Parallel(n_jobs=-1, prefer="processes")(joblib.delayed(_eleg_series_rows)((i, h, o, tau, genome)) for i, h, o, tau in datasets)
    fold = np.random.default_rng(Q_CB_FOLD_SEED).integers(0, 5, len(rows)); oof = []
    for f in range(5):
        tr = [r for r, k in zip(rows, fold) if k != f]; te = [r for r, k in zip(rows, fold) if k == f]
        bst = _lgb.train(VR_LGB_PARAMS, _lgb.Dataset(np.concatenate([r[0] for r in tr]), np.concatenate([r[1] for r in tr])), VR_ROUNDS)
        oof.append(bst.predict(np.concatenate([r[0] for r in te])))
    q = np.quantile(np.concatenate(oof), np.linspace(0.0, 1.0, Q_GRID))
    bst = _lgb.train(VR_LGB_PARAMS, _lgb.Dataset(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])), VR_ROUNDS)
    return bst, q


def _train_vr_leg(datasets):
    """Ship booster + CDF map for the VR leg. The q map comes from a 5-fold series-grouped OOF (seed Q_CB_FOLD_SEED),
    the booster from all series. Mirrors selfimprove/ship_vr_leg.py + vr_leg_stream (which used the mlparch folds)."""
    import lightgbm as _lgb
    rows = joblib.Parallel(n_jobs=-1, prefer="processes")(joblib.delayed(_vr_series_rows)((i, h, o, tau)) for i, h, o, tau in datasets)
    fold = np.random.default_rng(Q_CB_FOLD_SEED).integers(0, 5, len(rows))
    oof = []
    for f in range(5):
        tr = [r for r, k in zip(rows, fold) if k != f]; te = [r for r, k in zip(rows, fold) if k == f]
        bst = _lgb.train(VR_LGB_PARAMS, _lgb.Dataset(np.concatenate([r[0] for r in tr]), np.concatenate([r[1] for r in tr])), VR_ROUNDS)
        oof.append(bst.predict(np.concatenate([r[0] for r in te])))
    q_vr = np.quantile(np.concatenate(oof), np.linspace(0.0, 1.0, Q_GRID))
    bst = _lgb.train(VR_LGB_PARAMS, _lgb.Dataset(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])), VR_ROUNDS)
    return bst, q_vr


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    stride = STRIDE
    args = [(_id, h, o, tau, stride) for _id, h, o, tau in datasets]
    # Row counts are known from the online lengths alone -- no features needed -- so the
    # destination can be preallocated and each block copied in and dropped as it arrives.
    counts = [len(range(0, len(o), stride)) for _, _, o, _ in datasets]
    n_rows = int(sum(counts))
    Xa = np.empty((n_rows, len(ALL_NAMES)), dtype=np.float32)
    ya = np.empty(n_rows, dtype=np.int8)

    # Streaming results frees each block as it lands (~2.4 GB less peak). Guarded:
    # return_as needs joblib >= 1.3 and requirements.txt does not pin joblib -- an
    # unsupported kwarg here would be a BAD RUN, so fall back rather than raise.
    try:
        blocks = joblib.Parallel(n_jobs=-1, prefer="processes", return_as="generator")(
            joblib.delayed(_series_features_with_labels)(a) for a in args
        )
    except TypeError:
        blocks = joblib.Parallel(n_jobs=-1, prefer="processes")(
            joblib.delayed(_series_features_with_labels)(a) for a in args
        )

    at = 0
    for blk, lab in blocks:
        k = blk.shape[0]
        Xa[at : at + k] = blk
        ya[at : at + k] = lab
        at += k
        del blk, lab
    del blocks
    assert at == n_rows, f"row count mismatch: {at} != {n_rows}"

    model = None
    if n_rows and len(np.unique(ya)) > 1:
        import pandas as pd
        from lightgbm import LGBMClassifier

        # copy=False: the DataFrame is only here to carry column names into LightGBM.
        X = pd.DataFrame(Xa, columns=ALL_NAMES, copy=False)
        clf = LGBMClassifier(**LGBM_PARAMS).fit(X, ya)
        del X

        # --- low-weight MLP partner (see MLP_WEIGHT above) ---
        # Stored as flat arrays, not a pickled estimator: infer() then needs no sklearn at all,
        # and a single-row forward pass costs 7.6us vs 79us through sklearn's wrapper.

        # Tracked columns are chosen from the FITTED tree's own gain, so the bank adapts if the
        # feature set changes; the indices ship with the model so infer() uses the same ones.
        _gain = clf.booster_.feature_importance(importance_type="gain")
        _track = np.sort(np.argsort(-_gain)[:MLP_TRACK_N]).astype(np.int64)
        # Xa freed first: the MLP eats per-series full-resolution delta rows, not the stride matrix.
        del Xa
        # MLPMIR: the ship MLP block now lives in _fit_mlp_ship (v16 verbatim + MLP_MIRROR_ROWS), so a build driver can
        # refit ONLY the MLP against a shipped bundle's booster/mlp_track.
        _flats = _fit_mlp_ship(datasets, stride, _track, ya)

        # CLOSURE: the q_cb CDF map, from series-grouped K-fold OOF predictions of this same
        # blend. Runs AFTER the shipped fit so a failure here cannot corrupt the shipped model.
        # CLOSURE: the TCN leg, on ITS OWN stride-8 grid -- that is the grid infer() steps it on
        # ((_t_on & 7) == 0). Stride 3 and stride 8 do not align, so this is a separate feature
        # pass, not a subsample. mu/sd come from THIS matrix, matching tcn16_build:87-88 where they
        # are taken from the TCN's own feature matrix Fh (an earlier draft took them off the
        # stride-3 tree matrix, which is a different grid).
        import tcn_build as _tb

        _tcn_raw, _tcn_lab, _tcn_raw_keep = [], [], []
        for _sid, _h, _o, _tau in datasets:
            _blk, _lab = _series_features_with_labels((_sid, _h, _o, _tau, _tb.TCN_STRIDE))
            _tcn_raw_keep.append(bool(_blk.shape[0]))
            if _blk.shape[0]:
                _tcn_raw.append(np.ascontiguousarray(_blk[:, :TCN_DIM], dtype=np.float32))
                _tcn_lab.append(np.asarray(_lab, dtype=np.float32))
        _stack = np.concatenate(_tcn_raw, 0)
        _tcn_mu = _stack.mean(0).astype(np.float32)
        _tcn_sd = (_stack.std(0) + 1e-6).astype(np.float32)
        del _stack
        _tcn_seqs = [((r - _tcn_mu) / _tcn_sd).astype(np.float32) for r in _tcn_raw]
        del _tcn_raw
        _tcn_w, _tcn_scores = _tb.train_tcn(_tcn_seqs, _tcn_lab, TCN_DIM)
        # G1: the GRU leg, same sequences, same convention (q_gru from ship-fit in-sample scores).
        import gru_build as _gb
        _gru_w, _gru_scores = _gb.train_gru(_tcn_seqs, _tcn_lab, TCN_DIM)
        _q_gru = np.quantile(_gru_scores, np.linspace(0.0, 1.0, Q_GRID))
        assert np.all(np.diff(_q_gru) >= 0), "q_gru must be a monotone quantile map"
        # EXTRA_LEGS: same sequences, same convention (q from ship-fit in-sample scores).
        import seqleg as _sl
        _extra = {}
        _hc_raw = [np.asarray(_sl.histcal(_h), dtype=np.float32) for _sid, _h, _o, _tau in datasets]   # per series, history only
        _hc_raw = [_c for _c, _r in zip(_hc_raw, _tcn_raw_keep) if _r]
        _xcal_mean = np.mean(_hc_raw, axis=0).astype(np.float32); _xcal_std = (np.std(_hc_raw, axis=0) + 1e-6).astype(np.float32)
        _hc_std = [((_c - _xcal_mean) / _xcal_std).astype(np.float32) for _c in _hc_raw]
        _kept = [_d for _d, _r in zip(datasets, _tcn_raw_keep) if _r]
        assert len(_kept) == len(_tcn_seqs) == len(_hc_std), (len(_kept), len(_tcn_seqs), len(_hc_std))
        for _leg in EXTRA_LEGS:
            _hcf, _rbf = _sl.input_flags(_leg["spec"])      # canonical input order: features, histcal, rawblk8
            if _hcf or _rbf:
                assert int(_tb.TCN_STRIDE) == _sl.RAWBLK_N, "rawblk8 assumes the stride-8 grid"
                _seqs_in = [_sl.assemble_seq(_s, (_c if _hcf else None), (_sl.rawblk_seq(_h, _o, _s.shape[0], _tb.TCN_STRIDE) if _rbf else None))
                            for _s, _c, (_sid, _h, _o, _tau) in zip(_tcn_seqs, _hc_std, _kept)]
                _lw, _lscores = _sl.train_leg(_leg["spec"], _seqs_in, _tcn_lab, _sl.n_in_for(_leg["spec"], TCN_DIM)); del _seqs_in
            elif _leg["spec"].get("aug") == "histnull":
                # Lever 4: extra NULL training series carved from the unbroken histories (same rule as the box grid:
                # seqleg.histnull_pseudo), featurised by the same stride-8 producer, labels 0, subsampled by aug_frac.
                _rng = np.random.default_rng(int(_leg["spec"]["seed"])); _frac = float(_leg["spec"].get("aug_frac", 1.0))
                _aug_seqs, _aug_labs = [], []
                for _sid, _h, _o, _tau in datasets:
                    _pp = _sl.histnull_pseudo(_h)
                    if _pp is None or _rng.random() >= _frac: continue
                    _blk, _ = _series_features_with_labels((_sid, list(_pp[0]), list(_pp[1]), None, _tb.TCN_STRIDE))
                    if _blk.shape[0]:
                        _aug_seqs.append(((np.ascontiguousarray(_blk[:, :TCN_DIM], dtype=np.float32) - _tcn_mu) / _tcn_sd).astype(np.float32)); _aug_labs.append(np.zeros(_blk.shape[0], np.float32))
                _lw, _lscores = _sl.train_leg(_leg["spec"], _tcn_seqs + _aug_seqs, _tcn_lab + _aug_labs, TCN_DIM)
                if _aug_seqs:   # the q map must describe the REAL score distribution (serving), not the augmented one
                    _nets = _sl.load_nets(_lw, _leg["spec"], TCN_DIM); _lscores = np.concatenate([_sl.predict(_nets, _s) for _s in _tcn_seqs])
                del _aug_seqs, _aug_labs
            else:
                _lw, _lscores = _sl.train_leg(_leg["spec"], _tcn_seqs, _tcn_lab, TCN_DIM)
            _lq = np.quantile(_lscores, np.linspace(0.0, 1.0, Q_GRID))
            assert np.all(np.diff(_lq) >= 0), f"q_{_leg['key']} must be a monotone quantile map"
            _extra[_leg["key"]] = (_lw, _lq)
        del _tcn_seqs, _tcn_lab
        _q_tcn = np.quantile(_tcn_scores, np.linspace(0.0, 1.0, Q_GRID))
        assert np.all(np.diff(_q_tcn) >= 0), "q_tcn must be a monotone quantile map"

        if not _SKIP_COMPOUND:   # HM3S: the robust/chronos closures are dead weight at weight 0.0; their artifacts are not shipped
            # CLOSURE: the four robust alpha boosters + the robust_quantiles CDF map.
            import alpha_build as _ab

            _a_args = [(i, h, o, t, stride) for i, h, o, t in datasets]
            _a_blocks = joblib.Parallel(n_jobs=-1, prefer="processes")(
                joblib.delayed(_series_alpha_rows)(a) for a in _a_args)
            _a_views = [np.concatenate([b[0][k] for b in _a_blocks], 0) for k in range(4)]
            _a_lab = np.concatenate([b[1] for b in _a_blocks])
            _a_rowser = np.repeat(
                np.arange(len(_a_blocks)), [len(b[1]) for b in _a_blocks]).astype(np.int64)
            del _a_blocks
            _alpha_boosters, _robust_q = _ab.fit_alphas(_a_views, _a_lab, _a_rowser)
            del _a_views, _a_lab, _a_rowser
            for _nm, _bst in zip(_ALPHA_FILES, _alpha_boosters):
                joblib.dump(_bst, os.path.join(model_directory_path, _nm), compress=3)
            np.save(os.path.join(model_directory_path, "robust_quantiles.npy"), _robust_q)

            # CLOSURE: the competition-trained Chronos booster. The checkpoint under
            # resources/chronos/ is a PRETRAINED external foundation model and ships as-is; only the
            # 1024-column LightGBM on top of its embeddings is refitted here.
            import chronos_build as _cb

            import torch as _torch_tr
            from chronos import ChronosPipeline as _ChrPipe

            _chr_pipe = _ChrPipe.from_pretrained(
                os.path.join(model_directory_path, "chronos"), torch_dtype=_torch_tr.float32)
            _chr_X, _chr_y, _n_emb = _cb.build_chronos_matrix(
                datasets, stride, CHRONOS_CADENCE, _chr_pipe, _torch_tr)
            del _chr_pipe
            if _chr_X.shape[0] and len(np.unique(_chr_y)) > 1:
                _chr_bst = _cb.fit_chronos_booster(_chr_X, _chr_y, LGBM_PARAMS)
                joblib.dump(_chr_bst, os.path.join(model_directory_path, "chronos.joblib"),
                            compress=3)
                del _chr_bst
            del _chr_X, _chr_y

        _, _p_oof = _oof_champion_predictions(datasets, stride)   # MLPMIR: q_cb from the mirror-row blend, real OOF rows
        _q_cb = np.quantile(_p_oof, np.linspace(0.0, 1.0, Q_GRID))
        assert _q_cb.shape == (Q_GRID,) and np.all(np.diff(_q_cb) >= 0), (
            "q_cb must be a monotone quantile map"
        )

        model = {
            "booster": clf.booster_,
            "columns": ALL_NAMES,
            "mlp_flat": _flats,
            "mlp_weight": MLP_WEIGHT,
            "mlp_track": _track,
            # CLOSURE: rebuilt here rather than shipped prebuilt.
            "tcn_mu": _tcn_mu,
            "tcn_sd": _tcn_sd,
            "w_tcn": W_TCN,
            "q_cb": _q_cb,
            "tcn": _tcn_w,
            "q_tcn": _q_tcn,
            "gru": _gru_w,
            "q_gru": _q_gru,
            "w_gru": W_GRU,
            "gru_mu": _tcn_mu,   # trained on the same standardized sequences as the TCN in this build
            "gru_sd": _tcn_sd,
        }
        for _leg in EXTRA_LEGS:
            _lw, _lq = _extra[_leg["key"]]
            model[f"xleg_{_leg['key']}"] = _lw
            model[f"xq_{_leg['key']}"] = _lq
            model[f"xw_{_leg['key']}"] = float(_leg["w"])
            model[f"xspec_{_leg['key']}"] = dict(_leg["spec"])
        model["xlegs"] = [_leg["key"] for _leg in EXTRA_LEGS]
        model["xcal_mean"] = _xcal_mean
        model["xcal_std"] = _xcal_std
        _vr_bst, _vr_q = _train_vr_leg(datasets)
        model["vr_booster"] = _vr_bst
        model["q_vr"] = _vr_q
        model["w_vr"] = W_VR
        model["elegs"] = []
        for _spec in ELEGS_SPEC:
            _b, _q = _train_eleg(datasets, _spec["genome"])
            model["elegs"].append(dict(key=_spec["key"], booster=_b, q=_q, w=float(_spec["w"]), genome=[list(g) for g in _spec["genome"]]))

    joblib.dump(model, os.path.join(model_directory_path, "model.joblib"))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    # Pin every threading layer to 1 thread. With INFER_PARALLELISM the platform
    # forks N worker processes; if each also runs a multi-threaded single-row
    # predict, N x cores threads oversubscribe and predict slows 10-100x (the
    # prime suspect for the 4h cloud stall). Single-row predict never benefits
    # from threads anyway, so 1 is strictly better here.
    # Hard-override (not setdefault): the cloud base image may pre-set these high.
    for _v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_v] = "1"
    # Keep the limiter alive for the whole generator (discarding it can revert).
    _tpl = None
    try:
        from threadpoolctl import threadpool_limits

        _tpl = threadpool_limits(limits=1)
    except ImportError:
        pass

    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    assert model is not None, "model missing"
    _tree_columns = model["columns"]
    _use_compound = _tree_columns in (LATE_TREE_NAMES, TREE_NAMES)
    assert _tree_columns in (ALL_NAMES, LATE_TREE_NAMES, TREE_NAMES), (
        "model/column mismatch"
    )
    booster = model["booster"]

    if _SKIP_COMPOUND:   # HM3S: nothing to load; every use below sits behind the same gate (the booster loop skips None)
        robust_boosters = []; robust_quantiles = None; robust_grid = None; chronos_booster = None; chronos_pipe = None
    else:
        robust_boosters = [
            joblib.load(os.path.join(model_directory_path, "alpha_fligner_prefix_replay_v1.joblib")),
            joblib.load(os.path.join(model_directory_path, "alpha_arbocpd_causal_v2.joblib")),
            joblib.load(os.path.join(model_directory_path, "alpha_distbattery.joblib")),
            joblib.load(os.path.join(model_directory_path, "alpha_specbasis.joblib")),
        ]
        robust_quantiles = np.load(os.path.join(model_directory_path, "robust_quantiles.npy"))
        assert [b.num_feature() for b in robust_boosters] == [6, 4, 14, 12]
        assert robust_quantiles.shape == (4096,) and np.all(np.diff(robust_quantiles) >= 0)
        robust_grid = np.linspace(0.0, 1.0, 4096)
        chronos_booster = joblib.load(os.path.join(model_directory_path, "chronos.joblib"))
        assert chronos_booster.num_feature() == 1024

        import torch as _torch
        from chronos import ChronosPipeline as _ChronosPipeline
        _torch.set_num_threads(1)
        chronos_pipe = _ChronosPipeline.from_pretrained(
            os.path.join(model_directory_path, "chronos"),
            torch_dtype=_torch.float32,
        )

    def _chronos_pad(values):
        a = np.asarray(values, dtype=np.float32)
        if len(a) >= _CHR_WIN:
            return a[-_CHR_WIN:]
        return np.concatenate((np.full(_CHR_WIN-len(a), a[0], dtype=np.float32), a))

    def _chronos_embed(values):
        with _torch.no_grad():
            embedding, _ = chronos_pipe.embed([
                _torch.tensor(_chronos_pad(values), dtype=_torch.float32)
            ])
        return embedding.mean(dim=1).float().numpy()[0].astype(np.float64)
    booster_early = model.get("booster_early")
    _columns_early = model.get("columns_early")
    _use_stage_experts = booster_early is not None
    if _use_stage_experts:
        assert _tree_columns == LATE_TREE_NAMES, "late expert must be the 137-column v4 tree"
        assert _columns_early == TREE_NAMES, "early expert must be the 141-column v5 tree"
        assert booster.num_feature() == len(LATE_TREE_NAMES)
        assert booster_early.num_feature() == len(TREE_NAMES)
    # Low-weight MLP partner. Named `mlpm`, NOT `mlp` or `et` -- the point loop below rebinds
    # short names (`et = ins.step(point)`), and a previous version of this build was broken by
    # exactly that shadowing.
    from mlp_flat import DeltaBank, FlatMLP

    _mf = model.get("mlp_flat")
    _track = model.get("mlp_track")
    # Seed ensemble: `mlp_flat` is now a LIST of flattened nets (a single dict is still
    # accepted, so a previous single-net model file keeps working). `mlpm` becomes the
    # list; averaging the nets' probabilities in the point loop is the entire change.
    if _mf is None:
        _nets = []
    elif isinstance(_mf, (list, tuple)):
        _nets = [FlatMLP(d) for d in _mf]
    else:
        _nets = [FlatMLP(_mf)]
    mlpm = _nets if _nets else None
    w_mlp = float(model.get("mlp_weight", 0.0)) if mlpm is not None else 0.0
    w_lgb = 1.0 - w_mlp
    if mlpm is not None:
        assert _track is not None, "mlp_track missing from model"
        for _n in _nets:
            assert _n.n_features == len(ALL_NAMES) + len(_track) * 7, (
                "mlp/delta dim mismatch"
            )
    # --- build E: streaming TCN third leg + CDF mixing maps ---
    _tcnw = model.get("tcn")
    _tcn_mu = model.get("tcn_mu")
    _tcn_sd = model.get("tcn_sd")
    _q_cb = model.get("q_cb")
    _q_cb_early = model.get("q_cb_early")
    _q_tcn = model.get("q_tcn")
    _gruw = model.get("gru")            # G1: optional fourth leg
    _q_gru = model.get("q_gru")
    _w_gru = float(model["w_gru"]) if _gruw is not None else 0.0
    _gru_mu = model.get("gru_mu")        # the GRU ships its OWN standardization (like each TCN build does)
    _gru_sd = model.get("gru_sd")
    if _gruw is not None:
        assert _q_gru is not None and len(_q_gru) == len(_q_cb), "gru present but q_gru missing/mismatched"
        assert _gru_mu is not None and _gru_sd is not None and _gru_mu.shape == _tcn_mu.shape, "gru present but gru_mu/gru_sd missing"
        from gru_build import GruStream as _GruStream
    _xkeys = list(model.get("xlegs") or [])     # EXTRA_LEGS served generically
    _xw = {k: float(model[f"xw_{k}"]) for k in _xkeys}
    _xq = {k: model[f"xq_{k}"] for k in _xkeys}
    _xspec = {k: model[f"xspec_{k}"] for k in _xkeys}
    _xweights = {k: model[f"xleg_{k}"] for k in _xkeys}
    _xw_total = float(sum(_xw.values()))
    _xcal_mean = model.get("xcal_mean"); _xcal_std = model.get("xcal_std")
    _xhc = {k: ("histcal" in str(_xspec[k].get("input", "features"))) for k in _xkeys}
    _xrb = {k: ("rawblk8" in str(_xspec[k].get("input", "features"))) for k in _xkeys}
    _any_rb = any(_xrb.values())
    if any(_xhc.values()):
        assert _xcal_mean is not None and _xcal_std is not None, "histcal legs need xcal_mean/xcal_std"
    if _xkeys:
        assert all(len(_xq[k]) == len(_q_cb) for k in _xkeys), "extra leg q map mismatched"
        assert _gru_mu is not None, "extra legs need the shared gru_mu/gru_sd standardization"
        import seqleg as _sl
    # 🚨 FAIL CLOSED (2026-09-10). Two silent-degradation traps lived here:
    #   1. w_tcn defaulted to 0.10 while the shipped operator uses 0.2 -- a missing key silently
    #      halved the TCN weight.
    #   2. a bundle with no "tcn" key skipped the leg entirely with no assert, which is exactly
    #      what train() produced before closure: a SUCCESSFUL run serving a different operator.
    # Given the carry semantics (a failed run does not replace the entry, a successful one does),
    # crashing is strictly safer than degrading. Both now raise.
    if REQUIRE_TCN:
        assert _tcnw is not None, (
            "model.joblib has no 'tcn' key -- refusing to serve a silently TCN-less operator. "
            "If a no-TCN build is intended, set REQUIRE_TCN = False deliberately."
        )
    if _tcnw is not None:
        assert "w_tcn" in model, "tcn present but w_tcn missing -- refusing to guess the weight"
    _w_tcn = float(model["w_tcn"]) if "w_tcn" in model else 0.0
    _qg = np.linspace(0.0, 1.0, len(_q_cb)) if _q_cb is not None else None
    if _tcnw is not None:
        assert all(v is not None for v in (_tcn_mu, _tcn_sd, _q_cb, _q_tcn)), (
            "tcn maps missing"
        )
        if _use_stage_experts:
            assert _q_cb_early is not None and len(_q_cb_early) == len(_q_cb), (
                "early expert map missing"
            )

    # Force the booster itself to single-threaded prediction (independent of the
    # env vars / threadpoolctl, which may not be honored on the cloud image).
    # Booster.predict() reads self.params, so a plain dict write is enough --
    # and it avoids the reset_parameter C call (a segfault risk on mismatched envs).
    _vrb = model.get("vr_booster")                      # VR leg: fail closed on a partial bundle
    _q_vr = model.get("q_vr")
    if _vrb is not None:
        assert _q_vr is not None and len(_q_vr) == len(_q_cb) and "w_vr" in model, "vr_booster present but q_vr/w_vr missing"
        assert _vrb.num_feature() == len(ALL_NAMES) + VR_N_COLS, "vr booster column mismatch"
    _w_vr = float(model["w_vr"]) if _vrb is not None else 0.0
    _elegs = list(model.get("elegs") or [])              # evolved legs: fail closed on an incomplete entry
    for _e in _elegs:
        assert _vrb is not None, "evolved legs need the VR leg (their rows include the VR columns)"
        assert all(k in _e for k in ("booster", "q", "w", "genome")) and len(_e["q"]) == len(_q_cb), "incomplete evolved leg"
        assert _e["booster"].num_feature() == len(ALL_NAMES) + VR_N_COLS + len(_e["genome"]), "evolved leg column mismatch"
    for _booster in (booster, booster_early, _vrb, *[_e["booster"] for _e in _elegs],
                     chronos_booster, *robust_boosters):
        if _booster is None:
            continue
        try:
            if getattr(_booster, "params", None) is None:
                _booster.params = {}
            _booster.params["num_threads"] = 1
        except Exception:
            pass

    # --- instrumentation + graceful degradation (NOT fail-fast) ---
    # The old comment here read "a failed run costs zero quota". That premise is
    # false: a failed run costs a bad_run AND leaves the live OOS entry on whatever
    # ran last (fuzzy-katlun sat on 62.7047 with best=62.8641 for exactly this
    # reason). So the budget no longer raises -- on overrun we stop computing
    # features and keep emitting one cheap score per remaining point, which still
    # scores. Budget raised 5400 -> 9000s: #15 ran 2991s, and the wall clock is the
    # ONLY nondeterministic path in this file, i.e. the only code-level way that
    # byte-identical submissions (#15 ok, #21 bad) can differ.
    # buildH detfix (#60 re-applied): NO wall-clock control flow. The degrade
    # branch rewrote 96-98% of rows if it ever fired; the OOS determinism
    # re-check would then flip deterministic->false -> rewardRank null.
    hb_secs = float(os.environ.get("SB_INFER_HEARTBEAT_SEC", "30"))
    n_series = n_points = 0
    feat_time = pred_time = 0.0
    slow_series_t = 0.0
    slow_series_i = -1

    def _log(tag, t_now):
        pps = n_points / t_now if t_now > 0 else 0.0
        fu = 1e6 * feat_time / n_points if n_points else 0.0
        pu = 1e6 * pred_time / n_points if n_points else 0.0
        print(
            f"[infer {tag}] series={n_series} points={n_points} "
            f"elapsed={t_now:.1f}s pts/s={pps:.0f} "
            f"feat={fu:.1f}us/pt predict={pu:.1f}us/pt "
            f"slowest_series=#{slow_series_i}({slow_series_t:.2f}s)",
            file=sys.stderr,
            flush=True,
        )

    yield  # signal readiness

    t_start = time.perf_counter()
    next_hb = hb_secs
    try:
        for x_historical, x_online in datasets:
            s_t0 = time.perf_counter()

            fv, ins, iv, d8, ews, vs, evs, g, ga, gs, pds, mac, mbd, tp, idr, bin_mode = _init_streams(x_historical)
            if _SKIP_COMPOUND:
                fligner_stream = arbocpd_stream = distbattery_stream = specbasis_stream = None
                chronos_history = None
            else:
                fligner_stream = FlignerStream(x_historical)
                arbocpd_stream = ArbocpdStream(ins.e_hist)
                distbattery_stream = DistBatteryStream(ins.e_hist)
                specbasis_stream = SpecBasisStream(ins.e_hist)
                chronos_history = _chronos_embed(x_historical)
            chronos_online = []
            chronos_current = None
            dbank = DeltaBank(_track) if mlpm is not None else None
            if TTA_SIGN:
                assert not _use_compound, "TTA_SIGN needs the plain (non-compound) tree columns"
                _xh_m = np.negative(np.asarray(x_historical, dtype=np.float64)).tolist() if isinstance(x_historical, list) else np.negative(np.asarray(x_historical, dtype=np.float64))
                fv_m, ins_m, iv_m, d8_m, ews_m, vs_m, evs_m, g_m, ga_m, gs_m, pds_m, _mac_m, _mbd_m, _tp_m, _idr_m, _bin_m = _init_streams(_xh_m)
                dbank_m = DeltaBank(_track) if mlpm is not None else None
            _tcn = _TcnState(_tcnw) if _tcnw is not None else None
            _tcn_held = 0.0
            _gru = _GruStream(_gruw) if _gruw is not None else None
            _gru_held = 0.0
            _xs = {k: _sl.make_stream(_xweights[k], _xspec[k]) for k in _xkeys}
            # T8 (2026-09-16): the TCN-arch extra leg gets EIGHT phase-interleaved states, one per residue of t mod 8, so the value served at t
            # is the value computed at t instead of the one computed up to 7 points earlier. State 0 sees exactly the sequence the single state
            # saw, so every t % 8 == 0 output is bit-identical to the previous build. Measured on the all-t local screen (every online point of
            # all 9,477 series, `allt_20260916/read_allt.log`): TCN fresh +0.00027 5/5; the GRU does NOT generalise across phases (-0.00031),
            # so non-TCN legs keep the carry-forward. Serving cost x1.187 per point (measured), inference budget 43% -> 51%.
            _xs8 = {k: [_sl.make_stream(_xweights[k], _xspec[k]) for _ in range(8)] for k in _xkeys if str(_xspec[k].get("arch")) in X8_ARCHS}
            _hc_vec = ((np.asarray(_sl.histcal(x_historical), dtype=np.float32) - _xcal_mean) / _xcal_std).astype(np.float32) if any(_xhc.values()) else None
            _rb = None
            if _any_rb:                                   # ring buffer of the last 8 raw points (history tail first; zero-pad a short history)
                _tail = [float(_v) for _v in list(x_historical)[-(_sl.RAWBLK_N - 1):]]
                _rb = _collections.deque([0.0] * (_sl.RAWBLK_N - 1 - len(_tail)) + _tail, maxlen=_sl.RAWBLK_N)
            _xheld = {k: 0.0 for k in _xkeys}
            _vrs = VrStream(x_historical) if _vrb is not None else None
            _es = [GenomeStream(_e["genome"], x_historical) for _e in _elegs]
            _xs_m, _xs8_m, _xheld_m, _hc_vec_m, _rb_m, _vrs_m, _es_m = {}, {}, {}, None, None, None, []
            if TTA_SIGN and TTA_LEGS:   # v15: one mirrored state set per leg family, fed the mirrored history/points (causal by construction)
                if "x" in TTA_LEGS and _xkeys:
                    _xs_m = {k: _sl.make_stream(_xweights[k], _xspec[k]) for k in _xkeys}
                    _xs8_m = {k: [_sl.make_stream(_xweights[k], _xspec[k]) for _ in range(8)] for k in _xkeys if str(_xspec[k].get("arch")) in X8_ARCHS}
                    _xheld_m = {k: 0.0 for k in _xkeys}
                    _hc_vec_m = ((np.asarray(_sl.histcal(_xh_m), dtype=np.float32) - _xcal_mean) / _xcal_std).astype(np.float32) if any(_xhc.values()) else None
                    if _any_rb:
                        _tail_m = [-float(_v) for _v in list(x_historical)[-(_sl.RAWBLK_N - 1):]]
                        _rb_m = _collections.deque([0.0] * (_sl.RAWBLK_N - 1 - len(_tail_m)) + _tail_m, maxlen=_sl.RAWBLK_N)
                if _vrb is not None and ("vr" in TTA_LEGS or "e" in TTA_LEGS):
                    _vrs_m = VrStream(_xh_m)
                if "e" in TTA_LEGS:
                    _es_m = [GenomeStream(_e["genome"], _xh_m) for _e in _elegs]
            _t_on = 0
            rmax = 0.0
            for point in x_online:
                if _rb is not None:
                    _rb.append(float(point))
                if _rb_m is not None:
                    _rb_m.append(-float(point))

                tf0 = time.perf_counter()
                _vr_row = _vrs.step(point) if _vrs is not None else None   # every point, causal (history + online so far)
                _e_rows = [_s.step(point) for _s in _es]
                _vr_row_m = _vrs_m.step(-float(point)) if _vrs_m is not None else None
                _e_rows_m = [_s.step(-float(point)) for _s in _es_m]
                et = ins.step(point)
                robust_rows = None if _SKIP_COMPOUND else (
                    fligner_stream.step(point),
                    arbocpd_stream.step(et),
                    distbattery_stream.step(et),
                    specbasis_stream.step(et),
                )
                v = ews.step(et)
                row = (
                    fv.update(point)
                    + iv.update(gs.step(et))
                    + d8.step(et)
                    + vs.step(v, ews.sigma)
                    + list(evs.step(point))
                    + g.step(point)
                    + ga.step(point)
                    + pds.step(point)
                )
                arr_base = (
                    np.asarray(row, dtype=np.float32).astype(np.float64).reshape(1, -1)
                )
                if _use_compound:
                    extra = (
                        mac.step(point)
                        + mbd.step(point)
                        + tp.step(point)
                        + idr.step(point)
                        + bin_mode.step(point)
                    )
                    extra_arr = np.asarray(extra, dtype=np.float64).reshape(1, -1)
                    if _use_stage_experts:
                        if _t_on < STAGE_EXPERT_SPLIT:
                            arr_tree = np.concatenate([arr_base, extra_arr], axis=1)
                            active_booster = booster_early
                            active_q_cb = _q_cb_early
                        else:
                            arr_tree = np.concatenate([arr_base, extra_arr[:, :12]], axis=1)
                            active_booster = booster
                            active_q_cb = _q_cb
                    elif _tree_columns == TREE_NAMES:
                        arr_tree = np.concatenate([arr_base, extra_arr], axis=1)
                        active_booster = booster
                        active_q_cb = _q_cb
                    else:
                        arr_tree = np.concatenate([arr_base, extra_arr[:, :12]], axis=1)
                        active_booster = booster
                        active_q_cb = _q_cb
                else:
                    arr_tree = arr_base
                    active_booster = booster
                    active_q_cb = _q_cb
                if TTA_SIGN:   # mirrored pass: the same stream calls, on -point (causal by construction)
                    _pt_m = -float(point)
                    et_m = ins_m.step(_pt_m)
                    v_m = ews_m.step(et_m)
                    row_m = (
                        fv_m.update(_pt_m)
                        + iv_m.update(gs_m.step(et_m))
                        + d8_m.step(et_m)
                        + vs_m.step(v_m, ews_m.sigma)
                        + list(evs_m.step(_pt_m))
                        + g_m.step(_pt_m)
                        + ga_m.step(_pt_m)
                        + pds_m.step(_pt_m)
                    )
                    arr_base_m = np.asarray(row_m, dtype=np.float32).astype(np.float64).reshape(1, -1)
                tf1 = time.perf_counter()
                p = float(active_booster.predict(arr_tree, num_threads=1)[0])
                _p_tree = p   # v16: kept for the TTA_TREE=False path
                if _SKIP_COMPOUND:
                    p_robust = 0.0
                else:
                    robust_predictions = np.asarray([
                        float(b.predict(r.reshape(1, -1).astype(np.float64), num_threads=1)[0])
                        for b, r in zip(robust_boosters, robust_rows)
                    ])
                    robust_clipped = np.clip(robust_predictions, 1e-6, 1.0-1e-6)
                    robust_logit = np.log(robust_clipped/(1.0-robust_clipped)).mean()
                    robust_raw = 1.0/(1.0+math.exp(-float(robust_logit)))
                    p_robust = float(np.interp(
                        robust_raw, robust_quantiles, robust_grid
                    ))
                if mlpm is not None:
                    # Blend RAW probabilities, BEFORE the leaky-max -- matching #39 and
                    # het_ensemble.score. Blending after the max is a different operator.
                    # LOG-SCALED, and it must be the SAME transform train() applied before
                    # batch_delta_rows -- `arr` itself stays RAW because the booster above
                    # already consumed it and the tree must not be touched.
                    _row = dbank.push(_mlp_view(arr_base[0]))
                    _pm = 0.0
                    for _n in mlpm:
                        _pm += _n.predict_one(_row)
                    p = w_lgb * p + w_mlp * (_pm / len(mlpm))
                if TTA_SIGN:
                    p_m = float(active_booster.predict(arr_base_m, num_threads=1)[0]) if TTA_TREE else _p_tree
                    if mlpm is not None:
                        _row_m = dbank_m.push(_mlp_view(arr_base_m[0]))
                        _pmm = 0.0
                        for _n in mlpm:
                            _pmm += _n.predict_one(_row_m)
                        p_m = w_lgb * p_m + w_mlp * (_pmm / len(mlpm))
                    p = 0.5 * (p + p_m) if TTA_W == 0.5 else (1.0 - TTA_W) * p + TTA_W * p_m   # RAW base scores; 0.5 keeps #132's exact bytes
                tf2 = time.perf_counter()
                feat_time += tf1 - tf0
                pred_time += tf2 - tf1
                n_points += 1
                if _tcn is not None:
                    if (_t_on & 7) == 0:
                        _xn = (arr_base[0][: _tcn_mu.shape[0]].astype(np.float32) - _tcn_mu) / _tcn_sd
                        _tcn_held = _tcn.step(_xn)
                    if _gru is not None and (_t_on & 7) == 0:
                        _xg = (arr_base[0][: _gru_mu.shape[0]].astype(np.float32) - _gru_mu) / _gru_sd
                        _gru_held = _gru.step(_xg)
                    _p_gru = float(np.interp(_gru_held, _q_gru, _qg)) if _gru is not None else 0.0
                    _p_x = 0.0
                    if _xkeys:
                        if _xs8:                                                   # T8: TCN-arch legs step EVERY point, each state on its own residue
                            _xin8 = (arr_base[0][: _gru_mu.shape[0]].astype(np.float32) - _gru_mu) / _gru_sd
                            _rbv8 = _sl.rawblk_vec(_rb) if _rb is not None else None
                            for _k in _xs8:
                                _xheld[_k] = _xs8[_k][_t_on & 7].step(_sl.assemble(_xin8, (_hc_vec if _xhc[_k] else None), (_rbv8 if _xrb[_k] else None)))
                        if (_t_on & 7) == 0:
                            _xin = (arr_base[0][: _gru_mu.shape[0]].astype(np.float32) - _gru_mu) / _gru_sd
                            _rbv = _sl.rawblk_vec(_rb) if _rb is not None else None
                            for _k in _xkeys:
                                if _k in _xs8: continue                            # already stepped fresh above
                                _xheld[_k] = _xs[_k].step(_sl.assemble(_xin, (_hc_vec if _xhc[_k] else None), (_rbv if _xrb[_k] else None)))
                        if _xs_m:                                                  # v15: mirrored leg states, same cadence, on the mirrored bank
                            _xin8_m = (arr_base_m[0][: _gru_mu.shape[0]].astype(np.float32) - _gru_mu) / _gru_sd
                            _rbv8_m = _sl.rawblk_vec(_rb_m) if _rb_m is not None else None
                            for _k in _xs8_m:
                                _xheld_m[_k] = _xs8_m[_k][_t_on & 7].step(_sl.assemble(_xin8_m, (_hc_vec_m if _xhc[_k] else None), (_rbv8_m if _xrb[_k] else None)))
                            if (_t_on & 7) == 0:
                                for _k in _xkeys:
                                    if _k in _xs8_m: continue
                                    _xheld_m[_k] = _xs_m[_k].step(_sl.assemble(_xin8_m, (_hc_vec_m if _xhc[_k] else None), (_rbv8_m if _xrb[_k] else None)))
                        for _k in _xkeys:
                            _hx = (0.5 * (_xheld[_k] + _xheld_m[_k]) if TTA_LEG_W == 0.5 else (1.0 - TTA_LEG_W) * _xheld[_k] + TTA_LEG_W * _xheld_m[_k]) if _xs_m else _xheld[_k]
                            _p_x += _xw[_k] * float(np.interp(_hx, _xq[_k], _qg))
                    p = (1.0 - _w_tcn - _w_gru - _xw_total) * float(np.interp(p, active_q_cb, _qg)) + _w_tcn * float(
                        np.interp(_tcn_held, _q_tcn, _qg)
                    ) + _w_gru * _p_gru + _p_x
                if _vrb is not None:
                    _xv = np.concatenate([arr_base[0].astype(np.float32), _vr_row.astype(np.float32)]).astype(np.float64).reshape(1, -1)
                    _xv = np.nan_to_num(_xv, nan=0.0, posinf=0.0, neginf=0.0)
                    _p_vr = float(_vrb.predict(_xv, num_threads=1)[0])
                    if _vrs_m is not None:
                        _xv_m = np.nan_to_num(np.concatenate([arr_base_m[0].astype(np.float32), _vr_row_m.astype(np.float32)]).astype(np.float64).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
                        if "vr" in TTA_LEGS:
                            _pvr_m = float(_vrb.predict(_xv_m, num_threads=1)[0])
                            _p_vr = 0.5 * (_p_vr + _pvr_m) if TTA_LEG_W == 0.5 else (1.0 - TTA_LEG_W) * _p_vr + TTA_LEG_W * _pvr_m
                    p = (1.0 - _w_vr) * p + _w_vr * float(np.interp(_p_vr, _q_vr, _qg))
                    for _e, _er, _er_m in zip(_elegs, _e_rows, _e_rows_m if _es_m else [None] * len(_elegs)):
                        _xe = np.nan_to_num(np.concatenate([_xv[0], _er.astype(np.float32).astype(np.float64)]).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
                        _pe = float(_e["booster"].predict(_xe, num_threads=1)[0])
                        if _er_m is not None:
                            _xe_m = np.nan_to_num(np.concatenate([_xv_m[0], _er_m.astype(np.float32).astype(np.float64)]).reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
                            _pe_m = float(_e["booster"].predict(_xe_m, num_threads=1)[0])
                            _pe = 0.5 * (_pe + _pe_m) if TTA_LEG_W == 0.5 else (1.0 - TTA_LEG_W) * _pe + TTA_LEG_W * _pe_m
                        p = (1.0 - _e["w"]) * p + _e["w"] * float(np.interp(_pe, _e["q"], _qg))
                p_v7 = p
                if _SKIP_COMPOUND:
                    _t_on += 1
                    p = p_v7   # == p_v7 + 0.0*(...) + 0.0*(...) exactly
                else:
                    chronos_online.append(point)
                    if chronos_current is None or (_t_on % CHRONOS_CADENCE) == 0:
                        chronos_current = _chronos_embed(chronos_online)
                    chronos_row = np.concatenate(
                        (chronos_current, chronos_current-chronos_history)
                    ).reshape(1, -1)
                    p_chronos = float(chronos_booster.predict(chronos_row, num_threads=1)[0])
                    _t_on += 1
                    p = (
                        p_v7
                        + ROBUST_WEIGHT*(p_robust-p_v7)
                        + CHRONOS_WEIGHT*(p_chronos-p_v7)
                    )
                assert math.isfinite(p), "non-finite prediction"
                pc = min(max(p, 0.0), 1.0)
                rmax = pc if pc > RHO * rmax else RHO * rmax
                # Cheap in-series budget check (~every 1024 pts) to catch a runaway series.
                # NOTE: degrade, never raise. A RuntimeError here scores ZERO and leaves
                # the live OOS entry on whatever ran last; a degraded tail still scores.
                # The yield below is NOT skipped -- one score per online point, always.

                yield pc  # build C: emit RAW blend — the rho=0.97 ratchet costs +0.0061 AUC out-of-population (paired CI [+0.0004,+0.0141], reduced-test 2026-08-16) while reading -0.0002 in-population; never board-validated since it predates the ledger
            s_dt = time.perf_counter() - s_t0
            if s_dt > slow_series_t:
                slow_series_t, slow_series_i = s_dt, n_series
            n_series += 1
            now = time.perf_counter() - t_start
            if now >= next_hb:
                _log("hb", now)
                next_hb = now + hb_secs
    finally:
        _log("done", time.perf_counter() - t_start)
