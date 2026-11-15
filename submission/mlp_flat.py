"""Flatten a fitted sklearn MLPClassifier (+ its StandardScaler) into plain arrays.

WHY NOT CALL sklearn AT INFERENCE. Same lesson as `et_flat.py`: sklearn's per-row
`predict_proba` pays input validation and wrapper overhead that dwarfs the arithmetic. An
MLP forward pass here is two small matmuls (121->64->32->1, ~10k multiply-adds); calling it
through sklearn one row at a time costs orders of magnitude more than that. Shipping arrays
also means infer() does not import sklearn at all, removing an unpickle-version risk on the
runner where a failure is a BAD RUN rather than a low score.

The scaler is folded in as (mean, scale) so the caller passes a RAW feature row.

Predictions match sklearn to ~1e-15 (float summation order differs; verified in
harness/verify_mlp_flat.py).
"""
from __future__ import annotations

import numpy as np


def flatten_mlp(clf, scaler) -> dict:
    """Flatten a 2-hidden-layer MLPClassifier + StandardScaler into arrays."""
    assert clf.n_outputs_ == 1, "binary classifier expected"
    assert clf.activation == "relu", f"unsupported activation {clf.activation}"
    assert clf.out_activation_ == "logistic", f"unsupported output {clf.out_activation_}"
    return dict(
        mean=np.asarray(scaler.mean_, np.float64),
        scale=np.asarray(scaler.scale_, np.float64),
        W=[np.asarray(w, np.float64) for w in clf.coefs_],
        b=[np.asarray(v, np.float64) for v in clf.intercepts_],
        n_features=int(clf.n_features_in_),
    )


class FlatMLP:
    """Single-row forward pass over a flattened MLP. Takes a RAW (unscaled) feature row."""

    __slots__ = ("mean", "scale", "W", "b", "n_features", "_nl")

    def __init__(self, d: dict):
        self.mean = np.asarray(d["mean"], np.float64)
        self.scale = np.asarray(d["scale"], np.float64)
        self.W = [np.asarray(w, np.float64) for w in d["W"]]
        self.b = [np.asarray(v, np.float64) for v in d["b"]]
        self.n_features = int(d["n_features"])
        self._nl = len(self.W)

    def predict_one(self, x: np.ndarray) -> float:
        """P(class 1) for one raw feature row."""
        h = (np.asarray(x, np.float64) - self.mean) / self.scale
        for i in range(self._nl - 1):
            h = h @ self.W[i] + self.b[i]
            np.maximum(h, 0.0, out=h)          # relu
        # W[-1] is (n_hidden, 1) so this is a length-1 vector, not a scalar; numpy 2 refuses
        # to coerce a 1-element array via float(), so index it explicitly.
        z = float((h @ self.W[-1] + self.b[-1])[0])
        # logistic, written to avoid overflow on large |z|
        if z >= 0.0:
            return 1.0 / (1.0 + np.exp(-z))
        e = np.exp(z)
        return float(e / (1.0 + e))


def verify_parity(clf, scaler, X: np.ndarray, n: int = 500) -> float:
    """Max |flat - sklearn| over n raw rows."""
    ff = FlatMLP(flatten_mlp(clf, scaler))
    ref = clf.predict_proba(scaler.transform(X[:n]))[:, 1]
    got = np.array([ff.predict_one(X[i]) for i in range(min(n, len(X)))])
    return float(np.abs(ref - got).max())


DELTA_LAGS = (1, 2, 4, 8, 16, 32, 64)
# Backward-difference bank feeding the MLP (NOT the tree). Measured at deployment scale on fresh
# folds: field 16 (#39's {1,4,16}) +0.0018, field 64 (this) +0.0027, field 128 +0.0007, field 256
# +0.0016 -- a clean interior optimum at 64. Motivated by Katlun's lag_decomp (206.1M pairs):
# the TS-AUC shortfall is DISCRIMINATION-bound with 63% of it at lag >= 64, so a 16-step field is
# structurally blind to where most of the loss lives. Independently corroborated by our own
# evidence curve (per-series asymptote ~0.73 even past 400 post-break points).
# Strictly causal: x_t - x_{t-k} only. Zero-filled for t < k, matching the batch path exactly.


class DeltaBank:
    """Ring buffer over the tracked columns; emits [current 121 | deltas] per step."""

    __slots__ = ("track", "lags", "cap", "buf", "n")

    def __init__(self, track_idx, lags=DELTA_LAGS):
        self.track = np.asarray(track_idx, np.int64)
        self.lags = tuple(int(k) for k in lags)
        self.cap = max(self.lags) + 1
        self.buf = np.zeros((self.cap, len(self.track)), np.float64)
        self.n = 0

    def push(self, x) -> np.ndarray:
        """Feed the current 121-vector; returns the full MLP input row."""
        x = np.asarray(x, np.float64)
        cur = x[self.track]
        out = np.empty(len(x) + len(self.track) * len(self.lags), np.float64)
        out[:len(x)] = x
        at = len(x)
        for k in self.lags:
            w = len(self.track)
            if self.n >= k:
                out[at:at + w] = cur - self.buf[(self.n - k) % self.cap]
            else:
                out[at:at + w] = 0.0          # t < k: same convention as the batch path
            at += w
        self.buf[self.n % self.cap] = cur
        self.n += 1
        return out


def batch_delta_rows(B, track_idx, lags=DELTA_LAGS) -> np.ndarray:
    """Batch equivalent of DeltaBank, driven column-wise. Must match push() exactly."""
    B = np.asarray(B, np.float64)
    S = B[:, np.asarray(track_idx, np.int64)]
    parts = [B]
    for k in lags:
        d = np.zeros_like(S)
        if len(S) > k:
            d[k:] = S[k:] - S[:-k]
        parts.append(d)
    return np.hstack(parts)
