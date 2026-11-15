"""Apply the MLPMIR train-side patch to submit_v22_MLPMIR/main.py (a copy of submit_v16_flags/main.py).

Exact string replacements; every anchor must occur exactly once. Nothing at or below `def infer(` is touched
(asserted after writing). Idempotence guard: refuses to run on an already-patched file.
"""
import sys

PATH = "/mnt/HC_Volume_106714988/sbrt/submit_v22_MLPMIR/main.py"
BASE = "/mnt/HC_Volume_106714988/sbrt/submit_v16_flags/main.py"

src = open(PATH).read()
assert src == open(BASE).read(), "target is not a pristine copy of submit_v16_flags/main.py"
assert "MLP_MIRROR_ROWS" not in src


def sub(old, new):
    global src
    n = src.count(old)
    assert n == 1, f"anchor occurs {n}x: {old[:80]!r}"
    src = src.replace(old, new)


# ---------------------------------------------------------------------------------------------------------------
# A. the constant
sub(
    "MLP_HISTNULL = True\n",
    """MLP_HISTNULL = True
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
""",
)

# ---------------------------------------------------------------------------------------------------------------
# B. mirror helpers + the ship MLP fit moved out of train(), inserted after _histnull_delta_rows
sub(
    """    assert rows.shape == (len(parent), width), (rows.shape, len(parent), width)
    return rows, parent
""",
    """    assert rows.shape == (len(parent), width), (rows.shape, len(parent), width)
    return rows, parent


def _mirror_series(x_hist, x_online):
    \"\"\"MLP_MIRROR_ROWS: the sign-mirrored series (-h, -o), with infer()'s negation convention: a float64 negation
    (`np.negative(np.asarray(x_historical, dtype=np.float64))`, `-float(point)`), which is exact (-0.0 stays -0.0).\"\"\"
    return (np.negative(np.asarray(x_hist, dtype=np.float64)),
            np.negative(np.asarray(x_online, dtype=np.float64)))


def _mirror_series_delta_rows(args, dtype=np.float32):
    \"\"\"MLP_MIRROR_ROWS worker: the MLP delta rows of ONE series' sign mirror (id, -h, -o). The _series_delta_rows
    recipe on the mirrored series, except that the float32 feature block is widened to float64 BEFORE _mlp_view --
    the order infer() uses (arr_base_m = float32 row -> float64 -> _mlp_view -> DeltaBank in float64). The rows then
    equal infer()'s mirrored rows exactly; the only remaining difference is the float32 storage (`dtype`) of the
    training matrix. `dtype=np.float64` returns the pre-storage rows (used by the pre-flight check).\"\"\"
    _dataset_id, x_hist, x_online, stride, track = args
    from mlp_flat import batch_delta_rows

    h_m, o_m = _mirror_series(x_hist, x_online)
    feats = _mlp_view(_series_features(h_m, o_m).astype(np.float64))   # widen first, as infer() does
    rows = batch_delta_rows(feats, track)
    return rows[np.arange(0, len(rows), stride)].astype(dtype)


def _mirror_delta_rows(datasets, stride, track, keep=None):
    \"\"\"MLP_MIRROR_ROWS: the MLP delta rows of the mirrored series (id, -h, -o) via _mirror_series_delta_rows (same
    stride, same track), in dataset order, for the series with keep[i] true (every series when keep is None). A
    mirrored series has exactly its real series' stride rows, so its labels are the real rows' labels.\"\"\"
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
    \"\"\"MLP_FIT_PARALLEL worker: one MLP_SEEDS net (MLP_PARAMS, random_state=seed) pinned to one BLAS thread.\"\"\"
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return MLPClassifier(**dict(MLP_PARAMS, random_state=seed), early_stopping=False).fit(Xs, y)


def _fit_mlp_seeds(Xs, y):
    \"\"\"The fitted MLP_SEEDS nets on the same scaled rows, in seed order. MLP_FIT_PARALLEL: one worker process per
    seed (joblib memmaps Xs once, read-only, for all workers); False: the v16 sequential loop.\"\"\"
    if not MLP_FIT_PARALLEL:
        from sklearn.neural_network import MLPClassifier

        return [MLPClassifier(**dict(MLP_PARAMS, random_state=seed), early_stopping=False).fit(Xs, y)
                for seed in MLP_SEEDS]
    n_jobs = max(1, min(len(MLP_SEEDS), os.cpu_count() or 1))
    return joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(_fit_one_mlp)(Xs, y, seed) for seed in MLP_SEEDS)


def _fit_mlp_ship(datasets, stride, track, ya, progress=None):
    \"\"\"The SHIPPED MLP ensemble (list of flattened nets). train()'s MLP block, moved here verbatim so a build driver
    can refit ONLY the MLP against a shipped bundle's mlp_track (tree untouched). Training rows, in order: the real
    delta rows (labels ya), the history-null pseudo-series (MLP_HISTNULL, labels 0), then the mirrored real series
    (MLP_MIRROR_ROWS, labels ya again). With MLP_MIRROR_ROWS False this is the v16 ship fit exactly.\"\"\"
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
""",
)

# ---------------------------------------------------------------------------------------------------------------
# C. the closure: mirrored rows for TRAINING-fold series only, real rows scored, optional exactly-paired ctrl arm
OLD_OOF_START = "def _oof_champion_predictions(datasets, stride):\n"
OLD_OOF_END = """    assert _np.isfinite(oof).all(), "OOF has unscored rows -- fold partition is not a cover"
    return oof
"""
a = src.index(OLD_OOF_START)
z = src.index(OLD_OOF_END) + len(OLD_OOF_END)
assert src.count(OLD_OOF_START) == 1 and src.count(OLD_OOF_END) == 1 and a < z
NEW_OOF = '''def _q_cb_fold_of_series(n_series):
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
'''
src = src[:a] + NEW_OOF + src[z:]

# ---------------------------------------------------------------------------------------------------------------
# D. train(): the ship MLP block -> _fit_mlp_ship (same rows + mirrored rows)
OLD_T_START = """        # --- low-weight MLP partner (see MLP_WEIGHT above) ---
"""
OLD_T_END = """            _flats.append(flatten_mlp(_m, _sc))
            del _m
        del _Xs, _sc, _ya_mlp
"""
a = src.index(OLD_T_START)
z = src.index(OLD_T_END) + len(OLD_T_END)
assert src.count(OLD_T_START) == 1 and src.count(OLD_T_END) == 1 and a < z
old_block = src[a:z]
assert "_gain = clf.booster_.feature_importance(importance_type=\"gain\")" in old_block
assert "del Xa\n" in old_block and "_histnull_delta_rows(datasets, stride, _track)" in old_block
NEW_T = """        # --- low-weight MLP partner (see MLP_WEIGHT above) ---
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
"""
src = src[:a] + NEW_T + src[z:]

# ---------------------------------------------------------------------------------------------------------------
# E. train(): q_cb from the new (mirror-row) blend
sub(
    "        _p_oof = _oof_champion_predictions(datasets, stride)\n",
    "        _, _p_oof = _oof_champion_predictions(datasets, stride)   # MLPMIR: q_cb from the mirror-row blend, real OOF rows\n",
)

# ---------------------------------------------------------------------------------------------------------------
base = open(BASE).read()
tail_new = src[src.index("\ndef infer(") :]
tail_old = base[base.index("\ndef infer(") :]
assert tail_new == tail_old, "infer() changed -- refusing to write"
open(PATH, "w").write(src)
print("patched", PATH, "lines", src.count("\n"))
