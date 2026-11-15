"""A/A gate for a single-factor bundle change (PLAN_2026-09-30_model_change.md section 3A Step 3).

    python aa_gate.py BASE/model.joblib NEW/model.joblib --changed booster,q_cb

Both bundles are joblib.load-ed; their key sets must be identical. Every key NOT in --changed is compared
recursively and must be identical: Booster -> model_to_string() equal; ndarray -> dtype, shape and bytes
equal; list/tuple/dict -> recurse; scalar -> ==. For the changed keys: booster.num_feature()==125,
booster.num_trees()==--trees (3000), len(q_cb)==Q_GRID and q_cb monotone. Exit 0 = PASS, 1 = FAIL.

MLPMIR copy (mlpmir/aa_gate.py, 2026-09-30): identical to rwb5/aa_gate.py plus a structural check for a changed
`mlp_flat`: a list of --mlp-nets (14) flattened nets, each with n_features == --mlp-features (125 + 52*7 = 489), the
same keys and array shapes as the base nets, all finite, and not identical to the base (something was refit).
    python aa_gate.py BASE/model.joblib NEW/model.joblib --changed mlp_flat,q_cb
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import sys

import joblib
import numpy as np

Q_GRID = 4096
# Keys the runbook names as must-be-unchanged; each pattern must match at least one key not in --changed.
MUST_CHECK = [
    "mlp_flat",
    "mlp_track",
    "mlp_weight",
    "tcn*",
    "gru*",
    "q_tcn",
    "q_gru",
    "q_vr",
    "vr_booster",
    "w_*",
    "elegs",
    "xleg_*",
    "xq_*",
    "xspec_*",
    "xw_*",
    "xlegs",
    "columns",
]


def _is_booster(x) -> bool:
    return type(x).__name__ == "Booster" and hasattr(x, "model_to_string")


def compare(a, b, path: str, errs: list) -> None:
    if _is_booster(a) or _is_booster(b):
        if not (_is_booster(a) and _is_booster(b)):
            errs.append(f"{path}: Booster vs {type(a).__name__}/{type(b).__name__}")
        elif a.model_to_string() != b.model_to_string():
            errs.append(f"{path}: Booster model_to_string differs")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            errs.append(f"{path}: ndarray vs {type(a).__name__}/{type(b).__name__}")
        elif a.dtype != b.dtype or a.shape != b.shape:
            errs.append(f"{path}: ndarray {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
        elif a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                compare(x, y, f"{path}[{i}]", errs)
        elif a.tobytes() != b.tobytes():
            errs.append(f"{path}: ndarray bytes differ")
        return
    if type(a) is not type(b):
        errs.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        return
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            errs.append(
                f"{path}: dict keys differ ({sorted(map(str, set(a) ^ set(b)))} / order)"
            )
            return
        for k in a:
            compare(a[k], b[k], f"{path}[{k!r}]", errs)
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            errs.append(f"{path}: len {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", errs)
        return
    if isinstance(a, (frozenset, set)):
        if a != b:
            errs.append(f"{path}: set differs")
        return
    if isinstance(a, (str, bytes, bool, int, float, complex, type(None), np.generic)):
        same = a == b
        if (
            not same
            and isinstance(a, (float, np.floating))
            and math.isnan(a)
            and math.isnan(b)
        ):
            same = True
        if not bool(same):
            errs.append(f"{path}: {a!r} != {b!r}")
        return
    errs.append(
        f"{path}: unhandled type {type(a).__name__} (refusing to pass what was not compared)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument(
        "--changed", required=True, help="comma-separated keys allowed to differ"
    )
    ap.add_argument(
        "--trees",
        type=int,
        default=3000,
        help="expected num_trees() of a changed booster",
    )
    ap.add_argument("--features", type=int, default=125)
    ap.add_argument("--q-grid", type=int, default=Q_GRID)
    ap.add_argument("--mlp-nets", type=int, default=14)
    ap.add_argument("--mlp-features", type=int, default=489)
    a = ap.parse_args()
    changed = [k for k in a.changed.split(",") if k]
    base, new = joblib.load(a.base), joblib.load(a.new)
    errs: list = []
    if set(base) != set(new):
        errs.append(
            f"key sets differ: only base {sorted(set(base) - set(new))}, only new {sorted(set(new) - set(base))}"
        )
    for k in changed:
        if k not in base or k not in new:
            errs.append(f"--changed key {k!r} missing from a bundle")
    unchanged = [k for k in base if k in new and k not in changed]
    for k in unchanged:
        compare(base[k], new[k], k, errs)
    for pat in MUST_CHECK:
        if not any(fnmatch.fnmatchcase(k, pat) for k in unchanged) and not any(
            fnmatch.fnmatchcase(k, pat) for k in changed
        ):
            errs.append(f"expected key pattern {pat!r} matches no key")
    info = {}
    for k in changed:
        if k not in new:
            continue
        v = new[k]
        if _is_booster(v):
            info[k] = {
                "num_feature": v.num_feature(),
                "num_trees": v.num_trees(),
                "differs_from_base": (not _is_booster(base.get(k)))
                or v.model_to_string() != base[k].model_to_string(),
            }
            if v.num_feature() != a.features:
                errs.append(f"{k}: num_feature {v.num_feature()} != {a.features}")
            if v.num_trees() != a.trees:
                errs.append(f"{k}: num_trees {v.num_trees()} != {a.trees}")
        elif k == "mlp_flat":
            bl = base.get(k)
            struct = []
            if not (isinstance(v, list) and isinstance(bl, list)):
                struct.append(f"type {type(v).__name__} (base {type(bl).__name__}), want list")
            else:
                if len(v) != a.mlp_nets or len(bl) != a.mlp_nets:
                    struct.append(f"len {len(v)} (base {len(bl)}) != --mlp-nets {a.mlp_nets}")
                for i, (x, y) in enumerate(zip(v, bl)):
                    if set(x) != set(y):
                        struct.append(f"net {i}: keys {sorted(x)} vs base {sorted(y)}")
                        continue
                    if not (x["n_features"] == y["n_features"] == a.mlp_features):
                        struct.append(f"net {i}: n_features {x['n_features']} (base {y['n_features']})")
                    arrs = [x["mean"], x["scale"], *x["W"], *x["b"]]
                    shp = [np.shape(t) for t in arrs]
                    shp_b = [np.shape(t) for t in (y["mean"], y["scale"], *y["W"], *y["b"])]
                    if shp != shp_b:
                        struct.append(f"net {i}: shapes {shp} vs base {shp_b}")
                    if not all(np.all(np.isfinite(np.asarray(t, dtype=np.float64))) for t in arrs):
                        struct.append(f"net {i}: non-finite parameters")
            same = []
            if not struct:
                errs_k: list = []
                compare(v, bl, k, errs_k)
                same = not errs_k
            info[k] = {
                "n_nets": len(v) if isinstance(v, list) else None,
                "n_features": sorted({d["n_features"] for d in v}) if isinstance(v, list) else None,
                "structure_ok": not struct,
                "differs_from_base": (not same) if not struct else None,
            }
            errs.extend(f"{k}: {m}" for m in struct)
            if not struct and same:
                errs.append(f"{k}: identical to the base bundle (declared changed but nothing was refit)")
        elif k == "q_cb" or k.startswith("q_"):
            q = np.asarray(v)
            mono = bool(q.ndim == 1 and np.all(np.diff(q) >= 0))
            info[k] = {
                "len": int(q.size),
                "monotone": mono,
                "dtype": str(q.dtype),
                "max_abs_vs_base": float(np.max(np.abs(q - np.asarray(base[k]))))
                if k in base
                else None,
            }
            if q.size != a.q_grid:
                errs.append(f"{k}: len {q.size} != Q_GRID {a.q_grid}")
            if not mono:
                errs.append(f"{k}: not monotone")
        else:
            info[k] = {
                "type": type(v).__name__,
                "note": "changed key without a structural check",
            }
    print(
        f"keys: {len(base)} base / {len(new)} new; unchanged compared: {len(unchanged)}; changed: {changed}"
    )
    for k, v in info.items():
        print(f"  changed {k}: {v}")
    if errs:
        print(f"AA_GATE FAIL ({len(errs)} problem(s)):")
        for e in errs[:50]:
            print("  -", e)
        return 1
    print("AA_GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
