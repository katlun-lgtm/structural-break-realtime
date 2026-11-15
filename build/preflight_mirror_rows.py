"""MLPMIR pre-flight (PLAN_2026-09-30_model_change.md section 3B, mandatory): train/serve parity of the MIRRORED MLP rows.

For each series, the TRAINING mirror path of the package (main._mirror_series_delta_rows / main._mirror_delta_rows on
(id, -h, -o), stride 1 so every online point is compared) must equal, row for row, the rows infer() itself pushes
through the mirrored delta bank (`dbank_m.push(_mlp_view(arr_base_m[0]))`). The streaming rows are captured from the
package's REAL infer() (not a re-implementation): mlp_flat.DeltaBank is swapped for a recording subclass while infer()
runs; infer() creates `dbank` (real) and then `dbank_m` (mirror) per series.

Levels reported per series:
  pre_storage  main._mirror_series_delta_rows(..., dtype=float64) vs the stream rows: max|d| must be <= --tol (1e-9).
  stored       main._mirror_delta_rows (the float32 matrix the MLP trains on) vs float32(stream rows): every cell equal;
               its float64 max|d| is float32 storage rounding only (checked against half an ulp).
  stride       the stride-3 training rows are exactly the stride-1 rows [::3].
  real_control the REAL rows' current path (_series_delta_rows, v16, float32 _mlp_view) vs infer()'s real rows -- the
               pre-existing offset, reported for information (not a pass criterion; unchanged in v22 on purpose).

    /root/ml-comp/venv/bin/python preflight_mirror_rows.py [--pkg DIR] [--ids 0 5] [--raw PARQUET_DIR] [--out JSON]
Exit 0 = PASS.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", default="/mnt/HC_Volume_106714988/sbrt/submit_v22_MLPMIR")
ap.add_argument("--ids", type=int, nargs="+", default=[0, 5])
ap.add_argument("--raw", default="/root/ml-comp/sb_workspace/data")
ap.add_argument("--tol", type=float, default=1e-9, help="pre-storage max|d| bound")
ap.add_argument("--out", default=None)
args = ap.parse_args()

PKG = os.path.abspath(args.pkg)
sys.path.insert(0, PKG)

import joblib  # noqa: E402

import main as M  # noqa: E402
import mlp_flat  # noqa: E402

T0 = time.perf_counter()


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- capture infer()'s own delta-bank rows -------------------------------------------------------------------------
_Orig = mlp_flat.DeltaBank
BANKS: list = []


class _RecordingDeltaBank(_Orig):
    __slots__ = ("rows",)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows = []
        BANKS.append(self)

    def push(self, x):
        out = super().push(x)
        self.rows.append(np.array(out, dtype=np.float64, copy=True))
        return out


def load_series(ids):
    X = pd.read_parquet(
        os.path.join(args.raw, "X_train.parquet"), filters=[("id", "in", list(ids))]
    )
    Y = pd.read_parquet(os.path.join(args.raw, "y_train_index.parquet"))
    tau_ix = Y["tau_index"].to_dict()
    out = []
    for sid, g in X.reset_index().groupby("id"):
        g = g.sort_values("time")
        h = g[g.period == 1].value.to_numpy(float).tolist()
        o = g[g.period == 2].value.to_numpy(float).tolist()
        t = int(tau_ix.get(int(sid), -1))
        out.append((int(sid), h, o, None if t < 0 else t))
    assert [s[0] for s in out] == sorted(ids), "series ids not found"
    return out


def diff64(stream: np.ndarray, batch: np.ndarray, names: list) -> dict:
    """max|d| between float64 stream rows and batch rows (any dtype, compared in float64)."""
    r = {
        "rows_stream": int(stream.shape[0]),
        "rows_batch": int(batch.shape[0]),
        "rows_match": bool(stream.shape == batch.shape),
    }
    if not r["rows_match"]:
        return r
    b64 = batch.astype(np.float64)
    fin_s, fin_b = np.isfinite(stream), np.isfinite(b64)
    r["nonfinite_mismatch"] = int((fin_s != fin_b).sum())
    d = np.where(fin_s & fin_b, np.abs(stream - b64), 0.0)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    r["max_abs_d"] = float(d.max())
    r["max_rel_d"] = float((d / np.maximum(1.0, np.abs(stream))).max())
    r["argmax"] = {
        "row": int(i),
        "col": names[j],
        "stream": float(stream[i, j]),
        "batch": float(b64[i, j]),
    }
    return r


def stored_check(stream: np.ndarray, stored32: np.ndarray) -> dict:
    """stored float32 training rows vs float32(stream rows): identical cells, and |d| <= half an ulp (rounding only)."""
    s32 = stream.astype(np.float32)
    same = (s32 == stored32) | (np.isnan(s32) & np.isnan(stored32))
    d = np.abs(stored32.astype(np.float64) - stream)
    half_ulp = 0.5 * np.spacing(np.abs(s32)).astype(np.float64)
    fin = np.isfinite(stream)
    return {
        "f32_cells_differing": int((~same).sum()),
        "f32_cells_total": int(same.size),
        "max_abs_d_f64_after_f32_storage": float(np.max(d[fin], initial=0.0)),
        "within_half_ulp_storage_rounding": bool((d[fin] <= half_ulp[fin]).all()),
    }


def main() -> int:
    res_dir = os.path.join(PKG, "resources")
    model = joblib.load(os.path.join(res_dir, "model.joblib"))
    track = np.asarray(model["mlp_track"], dtype=np.int64)
    from mlp_flat import DELTA_LAGS

    names = list(M.ALL_NAMES) + [
        f"d{k}_{M.ALL_NAMES[c]}" for k in DELTA_LAGS for c in track
    ]
    assert M.TTA_SIGN, "the mirrored stream set only exists with TTA_SIGN"
    assert M.MLP_MIRROR_ROWS, "package does not train on mirrored rows"
    series = load_series(args.ids)
    report = {
        "started_utc": utc(),
        "pkg": PKG,
        "tol_pre_storage": args.tol,
        "track_len": int(len(track)),
        "row_width": len(names),
        "series": [],
    }
    ok = True
    for sid, h, o, tau in series:
        BANKS.clear()
        mlp_flat.DeltaBank = _RecordingDeltaBank
        try:
            gen = M.infer([(h, iter(o))], res_dir)
            assert next(gen) is None
            scores = np.asarray(list(gen), dtype=np.float64)
        finally:
            mlp_flat.DeltaBank = _Orig
        assert scores.size == len(o) and np.isfinite(scores).all()
        assert len(BANKS) == 2, (
            f"expected 2 delta banks (dbank, dbank_m), got {len(BANKS)}"
        )
        s_real, s_mirr = np.stack(BANKS[0].rows), np.stack(BANKS[1].rows)
        ds = [(sid, h, o, tau)]
        pre = M._mirror_series_delta_rows(
            (sid, h, o, 1, track), dtype=np.float64
        )  # training path, before storage
        stored = M._mirror_delta_rows(
            ds, 1, track
        )  # training path as stored (float32), stride 1
        stored3 = M._mirror_delta_rows(ds, M.STRIDE, track)  # what train() uses
        real_cur = M._series_delta_rows(
            (sid, h, o, 1, track)
        )  # real rows, v16 path (unchanged)
        e = {
            "id": sid,
            "tau": tau,
            "n_hist": len(h),
            "n_online": len(o),
            "mirror_pre_storage": diff64(s_mirr, pre, names),
            "mirror_stored": diff64(s_mirr, stored, names),
            "stride3_rows": int(stored3.shape[0]),
            "stride3_equal_stride1_subset": bool(
                stored3.shape[0] == len(range(0, len(o), M.STRIDE))
                and np.array_equal(stored3, stored[:: M.STRIDE])
            ),
            "mirror_differs_from_real": bool(
                s_mirr.shape == s_real.shape and not np.array_equal(s_mirr, s_real)
            ),
            "real_control_v16_path": diff64(s_real, real_cur, names),
        }
        if e["mirror_stored"]["rows_match"]:
            e["mirror_stored"].update(stored_check(s_mirr, stored))
        mp, ms = e["mirror_pre_storage"], e["mirror_stored"]
        e["pass"] = bool(
            mp["rows_match"]
            and mp["nonfinite_mismatch"] == 0
            and mp["max_abs_d"] <= args.tol
            and ms["rows_match"]
            and ms["f32_cells_differing"] == 0
            and ms["within_half_ulp_storage_rounding"]
            and e["stride3_equal_stride1_subset"]
            and e["mirror_differs_from_real"]
        )
        ok = ok and e["pass"]
        report["series"].append(e)
        print(json.dumps(e, indent=1, default=float), flush=True)
    report["pass"] = bool(ok)
    report["wall_s"] = round(time.perf_counter() - T0, 1)
    report["finished_utc"] = utc()
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
    print(
        f"PREFLIGHT {'PASS' if ok else 'FAIL'} ({report['wall_s']}s, {report['finished_utc']}): "
        + "; ".join(
            f"id {e['id']}: rows stream/batch {e['mirror_pre_storage']['rows_stream']}/"
            f"{e['mirror_pre_storage']['rows_batch']}, pre-storage max|d| {e['mirror_pre_storage'].get('max_abs_d')}, "
            f"stored f32 cells differing {e['mirror_stored'].get('f32_cells_differing')}, "
            f"stored max|d| {e['mirror_stored'].get('max_abs_d')} (half-ulp ok "
            f"{e['mirror_stored'].get('within_half_ulp_storage_rounding')}), real-row v16 offset "
            f"{e['real_control_v16_path'].get('max_abs_d')}"
            for e in report["series"]
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
