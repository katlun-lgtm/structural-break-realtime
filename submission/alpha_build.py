"""CLOSURE: fit the four robust alpha boosters and the robust_quantiles CDF map.

Pure array-in/array-out so it never imports main -- main owns the stream classes and hands the
feature matrices in, which keeps the import one-directional.

Hyperparameters are lifted verbatim from the verified full-data refit
(recovered_full_compound_20260910/refit_corrected_robust.py:68-78), which is the build that
produced the shipped alpha_*.joblib files. infer() loads a raw Booster (it calls
b.predict(...) and b.num_feature()), so a Booster is what gets dumped -- NOT the classifier.
"""
from __future__ import annotations

import numpy as np

ALPHA_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
    random_state=42,
)
ALPHA_FOLDS = 5
ALPHA_FOLD_SEED = 2000
ALPHA_WIDTHS = (6, 4, 14, 12)   # fligner, arbocpd, distbattery, specbasis -- asserted by infer()


def _logit_mean_to_raw(preds):
    """The exact readout infer() uses: clip -> logit -> mean over views -> sigmoid."""
    clipped = np.clip(preds, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).mean(axis=0)
    return 1.0 / (1.0 + np.exp(-logit))


# 🚨 THREAD OVERSUBSCRIPTION (measured 2026-09-10). The first version passed n_jobs=-1. On a
# 48-series fixture (3,552 rows x 4-14 features) that gave LightGBM 16 OpenMP threads for work
# measured at 0.84 s single-fit, and it burned **3h21m of CPU in 26 min of wall time without
# finishing** -- OpenMP barrier spinning, not computation. py-spy showed the main thread parked in
# lightgbm/basic.py:4157 across every sample.
# The reference build hit this too and solved it the blunt way: refit_corrected_robust.py:10-11
# pins OMP_NUM_THREADS=1 before importing anything.
# Threads are scaled to the work here instead, so the full-scale path keeps its parallelism.
SMALL_ROWS = 100_000


def _threads_for(n_rows, n_jobs):
    if n_jobs is not None:
        return n_jobs
    return -1 if n_rows >= SMALL_ROWS else 4


def fit_alphas(views, labels, row_series, n_jobs=None, q_grid=4096):
    """views: 4 float32 matrices (n_rows, width). Returns (boosters, robust_quantiles).

    robust_quantiles is built from OUT-OF-FOLD alpha predictions, series-grouped -- Codex's
    requirement 5 calls it an "OOF-derived robust logit-mean quantile map", and an in-sample map
    would be biased toward the extremes exactly as it would be for q_cb.
    """
    from lightgbm import LGBMClassifier

    assert len(views) == len(ALPHA_WIDTHS)
    for v, w in zip(views, ALPHA_WIDTHS):
        assert v.shape[1] == w, f"alpha view width {v.shape[1]} != expected {w}"

    params = dict(ALPHA_PARAMS, n_jobs=_threads_for(len(labels), n_jobs))

    # --- shipped models: fit on everything ---
    boosters = []
    for v in views:
        m = LGBMClassifier(**params).fit(np.nan_to_num(v), labels)
        boosters.append(m.booster_)

    # --- OOF pass for the quantile map ---
    n_series = int(row_series.max()) + 1
    rng = np.random.default_rng(ALPHA_FOLD_SEED)
    order = rng.permutation(n_series)
    fold_of_series = np.empty(n_series, dtype=np.int64)
    for f, chunk in enumerate(np.array_split(order, ALPHA_FOLDS)):
        fold_of_series[chunk] = f
    row_fold = fold_of_series[row_series]

    oof = np.full((len(views), len(labels)), np.nan, dtype=np.float64)
    for f in range(ALPHA_FOLDS):
        tr, te = row_fold != f, row_fold == f
        if not te.any() or len(np.unique(labels[tr])) < 2:
            continue
        for vi, v in enumerate(views):
            m = LGBMClassifier(**params).fit(np.nan_to_num(v[tr]), labels[tr])
            oof[vi, te] = m.booster_.predict(np.nan_to_num(v[te]))
            del m
    assert np.isfinite(oof).all(), "alpha OOF has unscored rows"

    raw = _logit_mean_to_raw(oof)
    quantiles = np.quantile(raw, np.linspace(0.0, 1.0, q_grid))
    assert quantiles.shape == (q_grid,) and np.all(np.diff(quantiles) >= 0), (
        "robust_quantiles must be a monotone quantile map"
    )
    return boosters, quantiles
