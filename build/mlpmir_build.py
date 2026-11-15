"""MLPMIR build driver: PLAN_2026-09-30_model_change.md section 3B (MLP_MIRROR, package submit_v22_MLPMIR).

Run on the 3090 FROM THE PACKAGE DIR (it imports that package's main.py):
    cd /data/mlpmir/pkg && /data/rwb5venv/bin/python /data/mlpmir/mlpmir_build.py \
        --reuse /data/rwb5/fixture --n-series 150 --out /data/mlpmir/fixture
    --n-series 0 = all 10,000 series (never the 9,477-series cache).

Stage order: load -> closure -> gates P2/L1 (so a KILL shows ~25 min before the ship fit ends) -> ship MLP -> diag -> bundle.
Keeps the shipped `booster` and `mlp_track` (read from the bundle, never refit) and rebuilds only:
  mlp_flat  main._fit_mlp_ship(datasets, STRIDE, bundle mlp_track, ya): the 14 nets on real + histnull + mirrored rows
  q_cb      quantiles of main._oof_champion_predictions(...)'s oof_new (5-fold closure; mirrored rows of each fold's
            TRAINING series only; real OOF rows scored)
Reuses Xa/ya/row_t (and oof_ctrl when valid) from the RWB5 build dir (--reuse): the stride-3 featurisation is the v16
code in both packages; a spot check re-featurises 3 series with THIS package and requires exact equality.

Control arm for P2/L1 (--ctrl):
  compute  (default) in-run ctrl on every fold: same fold tree, track, real + histnull rows, no mirrored rows, fit the
           same way (MLP_FIT_PARALLEL) -> exactly paired with oof_new. Cheap with parallel seed fits (~1-2 min/fold).
  auto     reuse <reuse>/oof_ctrl.npy when it covers exactly these series (same n_series -> same folds), plus a
           pairing probe: the ctrl arm is recomputed in-run on --probe-folds (default fold 0) and compared with the reused
           one; otherwise compute the ctrl arm in-run on every fold (exactly paired, +14 MLP fits per fold).
  reuse    require the reused oof_ctrl (fail if absent/invalid); probe as above.
  The RWB5 oof_ctrl was fit with the sequential v16 loop; with MLP_FIT_PARALLEL the reused arm differs from the in-run
  recipe by float rounding only (dev, 40 series: closure OOF max|d| 4.4e-6), which the probe reports.

Writes <out>/model.joblib (new = dict(shipped); ONLY mlp_flat and q_cb overwritten), <out>/REPORT.json (rewritten after
every stage, so a killed run leaves partial gates), <out>/build.log, oof_new.npy (+ oof_ctrl_inrun.npy).
Never writes into the package dir (train() is not called).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument(
    "--reuse", required=True, help="RWB5 build dir with Xa/ya/row_t/series_*.npy"
)
ap.add_argument("--out", required=True)
ap.add_argument("--n-series", type=int, default=0, help="0 = all series")
ap.add_argument(
    "--pkg", default=os.getcwd(), help="package dir holding main.py (default: cwd)"
)
ap.add_argument("--raw", default="/data/rwb5/raw")
ap.add_argument(
    "--base-model",
    default=None,
    help="shipped bundle (default: <pkg>/resources/model.joblib)",
)
ap.add_argument("--ctrl", choices=("auto", "reuse", "compute"), default="compute")
ap.add_argument(
    "--probe-folds",
    default="0",
    help="comma list of folds to recompute ctrl on in reuse mode ('' = none)",
)
ap.add_argument(
    "--diag-series",
    type=int,
    default=200,
    help="series for the MLP old-vs-new / evenness diagnostic",
)
args = ap.parse_args()

PKG = os.path.abspath(args.pkg)
OUT = os.path.abspath(args.out)
REUSE = os.path.abspath(args.reuse)
assert os.path.isfile(os.path.join(PKG, "main.py")), f"no main.py in {PKG}"
assert not OUT.startswith(PKG + os.sep) and OUT != PKG, (
    "refusing to write build outputs inside the package dir"
)
assert OUT != REUSE, "refusing to write into the reuse dir"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PKG)

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy.special import expit  # noqa: E402

import main as M  # noqa: E402

T_START = time.perf_counter()
LOG = open(os.path.join(OUT, "build.log"), "a", buffering=1)


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc()} +{time.perf_counter() - T_START:8.1f}s] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cgroup_peak_gb():
    """memory.peak of this process's cgroup (systemd unit or scope): includes the joblib workers. None if unavailable."""
    try:
        with open("/proc/self/cgroup") as fh:
            rel = fh.read().strip().split("::", 1)[1]
        with open(os.path.join("/sys/fs/cgroup" + rel, "memory.peak")) as fh:
            return round(int(fh.read().strip()) / 1024**3, 3)
    except Exception:
        return None


def peak_rss_gb() -> dict:
    return {
        "self_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3
        ),
        "children_max_gb": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2, 3
        ),
        "cgroup_peak_gb": _cgroup_peak_gb(),
    }


TIMINGS: dict = {}
REPORT: dict = {
    "schema": "mlpmir-build-v1",
    "started_utc": utc(),
    "argv": sys.argv,
    "pkg": PKG,
    "out": OUT,
    "reuse": REUSE,
    "versions": {
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    },
    "constants": {
        "MLP_MIRROR_ROWS": bool(M.MLP_MIRROR_ROWS),
        "MLP_HISTNULL": bool(M.MLP_HISTNULL),
        "MLP_FIT_PARALLEL": bool(M.MLP_FIT_PARALLEL),
        "MLP_SEEDS": list(M.MLP_SEEDS),
        "MLP_PARAMS": dict(M.MLP_PARAMS),
        "MLP_WEIGHT": M.MLP_WEIGHT,
        "MLP_TRACK_N": M.MLP_TRACK_N,
        "STRIDE": M.STRIDE,
        "Q_GRID": M.Q_GRID,
        "Q_CB_FOLDS": M.Q_CB_FOLDS,
        "Q_CB_FOLD_SEED": M.Q_CB_FOLD_SEED,
        "TTA_SIGN": M.TTA_SIGN,
        "TTA_TREE": M.TTA_TREE,
        "TTA_W": M.TTA_W,
        "TTA_LEG_W": M.TTA_LEG_W,
    },
    "main_py_md5": md5(os.path.join(PKG, "main.py")),
    "timings_s": TIMINGS,
    "gates": {},
    "status": "running",
}


def save_report() -> None:
    REPORT["peak_rss"] = peak_rss_gb()
    REPORT["wall_s"] = round(time.perf_counter() - T_START, 1)
    tmp = os.path.join(OUT, "REPORT.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(REPORT, fh, indent=2, default=float)
        fh.write("\n")
    os.replace(tmp, os.path.join(OUT, "REPORT.json"))


class Stage:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"stage {self.name}: start")
        return self

    def __exit__(self, *exc):
        TIMINGS[self.name] = round(time.perf_counter() - self.t0, 1)
        log(f"stage {self.name}: {TIMINGS[self.name]:.1f}s  peak {peak_rss_gb()}")
        save_report()
        return False


def auc(labels, scores, times):
    """hunt_20260912/screen_variants.py:23-30 verbatim (time-step-grouped pair AUC); same as rwb5_build.py."""
    num = den = 0.0
    for tid in np.unique(times):
        m = times == tid
        y = labels[m]
        p = int(y.sum())
        n = len(y) - p
        if not p or not n:
            continue
        o = y[np.argsort(scores[m], kind="mergesort")]
        num += float(np.cumsum(1 - o)[o == 1].sum())
        den += float(p * n)
    return num / den


def pair_count(labels, times):
    den = 0.0
    for tid in np.unique(times):
        m = times == tid
        y = labels[m]
        p = int(y.sum())
        n = len(y) - p
        if p and n:
            den += float(p * n)
    return den


def flat_ensemble(flats, X):
    """Vectorised FlatMLP forward (mean P(1) over the nets) for a block of RAW delta rows."""
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros(X.shape[0], dtype=np.float64)
    for d in flats:
        h = (X - d["mean"]) / d["scale"]
        for i in range(len(d["W"]) - 1):
            h = np.maximum(h @ d["W"][i] + d["b"][i], 0.0)
        out += expit((h @ d["W"][-1] + d["b"][-1])[:, 0])
    return out / len(flats)


def flats_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if set(x) != set(y) or x["n_features"] != y["n_features"]:
            return False
        for k in ("mean", "scale"):
            if not np.array_equal(x[k], y[k]):
                return False
        for k in ("W", "b"):
            if len(x[k]) != len(y[k]) or not all(
                np.array_equal(p, q) for p, q in zip(x[k], y[k])
            ):
                return False
    return True


def main() -> int:
    log(
        f"MLPMIR build: n_series={args.n_series} reuse={REUSE} out={OUT} pkg={PKG} "
        f"lightgbm={lgb.__version__} sklearn={sklearn.__version__}"
    )
    assert M.MLP_MIRROR_ROWS, "package main.py does not train on mirrored rows"
    base_path = args.base_model or os.path.join(PKG, "resources", "model.joblib")
    REPORT["base_model"] = {"path": base_path, "md5": md5(base_path)}
    old = joblib.load(base_path)
    track = np.asarray(old["mlp_track"])
    assert track.dtype == np.int64 and track.shape == (M.MLP_TRACK_N,), (
        track.dtype,
        track.shape,
    )
    assert np.all(np.diff(track) > 0), "mlp_track must be sorted and unique"
    assert old["booster"].num_feature() == len(M.ALL_NAMES) == 125
    assert list(old["columns"]) == list(M.ALL_NAMES), (
        "shipped bundle columns != ALL_NAMES"
    )
    from mlp_flat import DELTA_LAGS

    width = len(M.ALL_NAMES) + len(track) * len(DELTA_LAGS)
    assert len(old["mlp_flat"]) == len(M.MLP_SEEDS) and all(
        d["n_features"] == width for d in old["mlp_flat"]
    )
    assert float(old["mlp_weight"]) == M.MLP_WEIGHT, (
        "bundle mlp_weight != MLP_WEIGHT (closure blend weight)"
    )
    REPORT["base_model"]["mlp_track"] = track.tolist()
    save_report()

    # ---- data: closure_20260910/fullscale_measure.py make() (as rwb5_build.py), all series in parquet order ------------
    with Stage("load_data"):
        X = pd.read_parquet(os.path.join(args.raw, "X_train.parquet"))
        Y = pd.read_parquet(os.path.join(args.raw, "y_train_index.parquet"))
        ids = X.index.get_level_values(0).to_numpy()
        assert np.all(np.diff(ids) >= 0), (
            "X_train ids are not sorted/contiguous: make() bounds would be wrong"
        )
        val = X["value"].to_numpy(np.float64)
        per = X["period"].to_numpy()
        assert set(np.unique(per).tolist()) <= {1, 2}, "unexpected period codes"
        uniq, starts = np.unique(ids, return_index=True)
        bounds = list(starts) + [len(ids)]
        tau_ix = Y["tau_index"].to_dict()
        n = len(uniq) if args.n_series <= 0 else min(args.n_series, len(uniq))

        def make(n):
            ds = []
            for k in range(n):
                sid = int(uniq[k])
                lo, hi = bounds[k], bounds[k + 1]
                v, p = val[lo:hi], per[lo:hi]
                h = v[p == 1]
                o = v[p == 2]  # RT codes period 1=history, 2=online
                t = int(tau_ix.get(sid, -1))
                ds.append((sid, h.tolist(), o.tolist(), None if t < 0 else t))
            return ds

        datasets = make(n)
        del X, Y, ids, val, per
        stride = M.STRIDE
        counts = [len(range(0, len(o), stride)) for _, _, o, _ in datasets]
        n_rows = int(sum(counts))
        REPORT["data"] = {
            "n_series_available": int(len(uniq)),
            "n_series": len(datasets),
            "n_break_series": int(sum(t is not None for *_, t in datasets)),
            "n_rows_stride": n_rows,
            "n_histnull_series": int(sum(len(h) >= 2560 for _, h, _, _ in datasets)),
            "first_ids": [d[0] for d in datasets[:3]],
            "last_id": datasets[-1][0],
        }
        log(f"data: {REPORT['data']}")

    # ---- reuse the RWB5 featurisation (Xa, ya, row_t) ----------------------------------------------------------------
    with Stage("load_reuse"):
        r_ids = np.load(os.path.join(REUSE, "series_ids.npy"))
        r_counts = np.load(os.path.join(REUSE, "series_counts.npy"))
        assert len(r_ids) >= len(datasets), (
            f"reuse dir has {len(r_ids)} series < {len(datasets)}"
        )
        assert np.array_equal(r_ids[: len(datasets)], [d[0] for d in datasets]), (
            "reuse series ids differ"
        )
        assert np.array_equal(r_counts[: len(datasets)], counts), (
            "reuse row counts differ"
        )
        full = len(r_ids) == len(datasets)
        mm = None if full else "r"
        Xa = np.load(os.path.join(REUSE, "Xa.npy"), mmap_mode=mm)
        ya = np.load(os.path.join(REUSE, "ya.npy"), mmap_mode=mm)
        row_t = np.load(os.path.join(REUSE, "row_t.npy"), mmap_mode=mm)
        assert (
            Xa.shape == (int(r_counts.sum()), len(M.ALL_NAMES))
            and Xa.dtype == np.float32
        )
        Xa = np.ascontiguousarray(Xa[:n_rows])
        ya = np.ascontiguousarray(ya[:n_rows])
        row_t = np.ascontiguousarray(row_t[:n_rows])
        assert ya.dtype == np.int8 and ya.shape == (n_rows,), (ya.dtype, ya.shape)
        exp_t = np.concatenate(
            [np.arange(0, len(o), stride, dtype=np.int64) for _, _, o, _ in datasets]
        )
        assert np.array_equal(row_t, exp_t), "reused row_t != stride row times"
        # spot check: THIS package's featuriser reproduces the reused rows exactly (first, middle, last series)
        offs = np.concatenate([[0], np.cumsum(counts)])
        spot = []
        for k in sorted({0, len(datasets) // 2, len(datasets) - 1}):
            sid, h, o, tau = datasets[k]
            blk, lab = M._series_features_with_labels((sid, h, o, tau, stride))
            eq = bool(
                np.array_equal(blk.astype(np.float32), Xa[offs[k] : offs[k + 1]])
                and np.array_equal(lab, ya[offs[k] : offs[k + 1]])
            )
            spot.append({"id": sid, "rows": int(blk.shape[0]), "equal": eq})
        assert all(s["equal"] for s in spot), f"reused features differ: {spot}"
        oc_path = os.path.join(REUSE, "oof_ctrl.npy")
        oc_ok = bool(full and os.path.isfile(oc_path))
        oof_ctrl_reused = None
        if oc_ok:
            oof_ctrl_reused = np.load(oc_path)
            oc_ok = bool(
                oof_ctrl_reused.shape == (n_rows,)
                and np.isfinite(oof_ctrl_reused).all()
            )
        REPORT["reuse"] = {
            "dir": REUSE,
            "n_series_in_reuse": int(len(r_ids)),
            "full_match": full,
            "spot_check": spot,
            "oof_ctrl_path": oc_path,
            "oof_ctrl_valid": oc_ok,
            "positive_rows": int(ya.sum()),
        }
        log(f"reuse: {REPORT['reuse']}")

    if args.ctrl == "compute" or (args.ctrl == "auto" and not oc_ok):
        ctrl_mode, with_ctrl = "compute", True
    else:
        assert oc_ok, (
            "--ctrl reuse but <reuse>/oof_ctrl.npy is missing or does not cover these series"
        )
        ctrl_mode = "reuse"
        with_ctrl = tuple(int(f) for f in args.probe_folds.split(",") if f.strip())
    REPORT["ctrl_mode"] = {
        "mode": ctrl_mode,
        "inrun_folds": "all" if with_ctrl is True else list(with_ctrl),
    }
    log(f"ctrl: {REPORT['ctrl_mode']}")
    save_report()

    # ---- closure: 5-fold OOF; oof_new = mirrored rows of TRAINING-fold series only, real rows scored -----------------
    with Stage("closure"):
        oof_ctrl_inrun, oof_new = M._oof_champion_predictions(
            datasets, stride, feats=(Xa, ya), progress=log, with_ctrl=with_ctrl
        )
        np.save(os.path.join(OUT, "oof_new.npy"), oof_new)
        if oof_ctrl_inrun is not None:
            np.save(os.path.join(OUT, "oof_ctrl_inrun.npy"), oof_ctrl_inrun)

    grid = np.linspace(0.0, 1.0, M.Q_GRID)
    fold_of_series = M._q_cb_fold_of_series(len(datasets))
    row_fold = fold_of_series[np.repeat(np.arange(len(datasets)), counts)]
    y64 = ya.astype(np.int64)
    with Stage("gates_P2_L1"):
        if ctrl_mode == "compute":
            oof_ctrl = oof_ctrl_inrun
        else:
            oof_ctrl = oof_ctrl_reused
            probe = {"folds": list(with_ctrl)}
            if with_ctrl:
                pm = np.isin(row_fold, list(with_ctrl))
                d = np.abs(oof_ctrl_inrun[pm] - oof_ctrl_reused[pm])
                probe.update(
                    {
                        "rows": int(pm.sum()),
                        "max_abs_inrun_vs_reused": float(d.max()),
                        "corr_inrun_vs_reused": float(
                            np.corrcoef(oof_ctrl_inrun[pm], oof_ctrl_reused[pm])[0, 1]
                        ),
                        "per_fold_auc_new_vs_inrun_ctrl": {
                            str(f): auc(
                                y64[row_fold == f],
                                oof_new[row_fold == f],
                                row_t[row_fold == f],
                            )
                            - auc(
                                y64[row_fold == f],
                                oof_ctrl_inrun[row_fold == f],
                                row_t[row_fold == f],
                            )
                            for f in with_ctrl
                        },
                    }
                )
                probe["exact"] = bool(probe["max_abs_inrun_vs_reused"] <= 1e-12)
                probe["status"] = (
                    "paired exactly"
                    if probe["exact"]
                    else "reused ctrl differs from the in-run recipe: read L1 with the probe folds' in-run deltas"
                )
            REPORT["gates"]["ctrl_pairing_probe"] = probe
            log(f"ctrl pairing probe: {probe}")
        q_new = np.quantile(oof_new, grid)
        q_ctrl = np.quantile(oof_ctrl, grid)
        q_ship = np.asarray(old["q_cb"], dtype=np.float64)
        assert q_new.shape == (M.Q_GRID,) and np.all(np.diff(q_new) >= 0), (
            "q_cb must be a monotone quantile map"
        )
        dq = q_ctrl - q_ship
        p2 = {
            "ctrl_source": ctrl_mode,
            "max_abs_qctrl_vs_shipped": float(np.max(np.abs(dq))),
            "median_elementwise_shift": float(np.median(dq)),
            "median_shift_at_p50": float(
                np.quantile(oof_ctrl, 0.5) - np.interp(0.5, grid, q_ship)
            ),
            "max_abs_qnew_vs_qctrl": float(np.max(np.abs(q_new - q_ctrl))),
            "max_abs_qnew_vs_shipped": float(np.max(np.abs(q_new - q_ship))),
        }
        p2["status"] = (
            "clean"
            if p2["max_abs_qctrl_vs_shipped"] <= 0.005
            else "drift: note in push message (not a stop; affects ctrl and new equally)"
        )
        REPORT["gates"]["P2"] = p2
        log(f"P2: {p2}")

        folds = []
        for f in range(M.Q_CB_FOLDS):
            te = row_fold == f
            a_c = auc(y64[te], oof_ctrl[te], row_t[te])
            a_n = auc(y64[te], oof_new[te], row_t[te])
            folds.append(
                {
                    "fold": f,
                    "rows": int(te.sum()),
                    "pairs": pair_count(y64[te], row_t[te]),
                    "auc_ctrl": a_c,
                    "auc_new": a_n,
                    "delta": a_n - a_c,
                }
            )
        dl = np.array([r["delta"] for r in folds])
        wp = np.array([r["pairs"] for r in folds])
        l1 = {
            "ctrl_source": ctrl_mode,
            "folds": folds,
            "fold_deltas": dl.tolist(),
            "n_up": int((dl > 0).sum()),
            "mean_delta": float(dl.mean()),
            "pair_weighted_delta": float(np.dot(dl, wp) / wp.sum()),
            "pooled_auc_ctrl": auc(y64, oof_ctrl, row_t),
            "pooled_auc_new": auc(y64, oof_new, row_t),
        }
        l1["pooled_delta"] = l1["pooled_auc_new"] - l1["pooled_auc_ctrl"]
        kill = l1["pair_weighted_delta"] < -0.0010 or l1["n_up"] <= 1
        l1["verdict"] = "KILL" if kill else "SHIP"
        l1["rule"] = "KILL if pair-weighted delta < -0.0010 or n_up <= 1 of 5"
        REPORT["gates"]["L1"] = l1
        log(
            f"L1: deltas={dl.tolist()} n_up={l1['n_up']} wdelta={l1['pair_weighted_delta']:.6f} -> {l1['verdict']}"
        )

    # ---- ship: the 14 nets on real + histnull + mirrored rows, against the SHIPPED mlp_track ---------------------------
    with Stage("ship_mlp"):
        flats = M._fit_mlp_ship(datasets, stride, track, ya, progress=log)
        assert len(flats) == len(M.MLP_SEEDS) and all(
            d["n_features"] == width for d in flats
        )
        assert not flats_equal(flats, old["mlp_flat"]), (
            "new MLP identical to the shipped one?"
        )

    # ---- diagnostic: new vs shipped MLP on real rows, and each one's evenness f(x) vs f(-x) ---------------------------
    with Stage("mlp_diag"):
        pick = np.unique(
            np.linspace(
                0, len(datasets) - 1, min(args.diag_series, len(datasets))
            ).astype(int)
        )

        def _rows(fn):
            return np.concatenate(
                joblib.Parallel(n_jobs=-1, prefer="processes")(
                    joblib.delayed(fn)(
                        (datasets[i][0], datasets[i][1], datasets[i][2], stride, track)
                    )
                    for i in pick
                )
            )

        R = _rows(M._series_delta_rows)  # real rows, v16 path
        Mr = _rows(M._mirror_series_delta_rows)  # mirrored rows, training path
        from mlp_flat import FlatMLP

        chk = max(
            abs(
                FlatMLP(flats[0]).predict_one(R[j])
                - flat_ensemble([flats[0]], R[j : j + 1])[0]
            )
            for j in range(min(20, len(R)))
        )
        po_r, po_m = (
            flat_ensemble(old["mlp_flat"], R),
            flat_ensemble(old["mlp_flat"], Mr),
        )
        pn_r, pn_m = flat_ensemble(flats, R), flat_ensemble(flats, Mr)

        def c(a, b):
            return float(np.corrcoef(a, b)[0, 1])

        REPORT["mlp_diag"] = {
            "series": int(len(pick)),
            "rows": int(len(R)),
            "note": "in-sample for the new nets (these series are in the ship fit); descriptive only",
            "flat_forward_vs_predict_one_max_abs": float(chk),
            "corr_new_vs_old_real": c(pn_r, po_r),
            "mean_abs_new_minus_old_real": float(np.mean(np.abs(pn_r - po_r))),
            "corr_new_vs_old_tta_avg": c(0.5 * (pn_r + pn_m), 0.5 * (po_r + po_m)),
            "old_corr_real_vs_mirror": c(po_r, po_m),
            "new_corr_real_vs_mirror": c(pn_r, pn_m),
            "old_mean_abs_real_minus_mirror": float(np.mean(np.abs(po_r - po_m))),
            "new_mean_abs_real_minus_mirror": float(np.mean(np.abs(pn_r - pn_m))),
            "old_mean_p_real_mirror": [float(po_r.mean()), float(po_m.mean())],
            "new_mean_p_real_mirror": [float(pn_r.mean()), float(pn_m.mean())],
        }
        log(f"mlp_diag: {REPORT['mlp_diag']}")
        del R, Mr

    # ---- bundle: dict(shipped), overwrite ONLY mlp_flat + q_cb --------------------------------------------------------
    with Stage("bundle"):
        new = dict(old)
        new["mlp_flat"] = flats
        new["q_cb"] = q_new
        out_model = os.path.join(OUT, "model.joblib")
        joblib.dump(new, out_model)
        chk = joblib.load(out_model)
        assert set(chk) == set(old), "bundle key set changed"
        bundle = {
            "path": out_model,
            "md5": md5(out_model),
            "bytes": os.path.getsize(out_model),
            "booster_identical_to_base": chk["booster"].model_to_string()
            == old["booster"].model_to_string(),
            "mlp_track_identical_to_base": bool(
                np.array_equal(chk["mlp_track"], old["mlp_track"])
                and chk["mlp_track"].dtype == old["mlp_track"].dtype
            ),
            "mlp_flat_reload_equal": flats_equal(chk["mlp_flat"], flats),
            "mlp_flat_n": len(chk["mlp_flat"]),
            "mlp_flat_n_features": sorted({d["n_features"] for d in chk["mlp_flat"]}),
            "q_cb_reload_equal": bool(np.array_equal(chk["q_cb"], q_new)),
            "q_cb_len": int(len(chk["q_cb"])),
            "q_cb_monotone": bool(np.all(np.diff(chk["q_cb"]) >= 0)),
            "q_cb_dtype": str(chk["q_cb"].dtype),
        }
        assert (
            bundle["booster_identical_to_base"]
            and bundle["mlp_track_identical_to_base"]
            and bundle["mlp_flat_reload_equal"]
            and bundle["q_cb_reload_equal"]
            and bundle["q_cb_monotone"]
        ), bundle
        REPORT["bundle"] = bundle
        log(f"bundle: {bundle}")

    REPORT["status"] = "done"
    REPORT["finished_utc"] = utc()
    save_report()
    log(
        f"DONE wall {time.perf_counter() - T_START:.1f}s gates: P2={REPORT['gates']['P2']['status']} "
        f"L1={REPORT['gates']['L1']['verdict']} (wdelta {REPORT['gates']['L1']['pair_weighted_delta']:.6f}, "
        f"n_up {REPORT['gates']['L1']['n_up']}/5) ctrl={ctrl_mode}"
    )
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as e:  # record the failure in REPORT.json, then re-raise
        REPORT["status"] = f"FAILED: {type(e).__name__}: {e}"
        REPORT["failed_utc"] = utc()
        save_report()
        log(REPORT["status"])
        raise
    raise SystemExit(rc)
