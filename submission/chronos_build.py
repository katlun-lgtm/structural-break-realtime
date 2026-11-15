"""CLOSURE: rebuild the competition-trained Chronos booster (chronos.joblib) inside train().

The pretrained checkpoint under resources/chronos/ is an external foundation model and ships as
such; what must be refitted locally is the 1024-column LightGBM on top of its embeddings.

🪤 THE CADENCE TRAP (chr_train_emb.py:6-8): the runner recomputes an embedding every CADENCE steps
and CARRIES IT FORWARD, so training rows must use the embedding at floor(t/CADENCE)*CADENCE, never
a freshly computed one. A model trained at one cadence and served at another cost -0.00091 locally
and -0.0003 on the board on submission #62.

Refresh semantics read off infer() (main.py:1099-1105), not assumed:
  - at online index t (0-based), refresh iff t % CADENCE == 0
  - the embedded window is x_online[:t+1], trailing 256, left-padded with its first value
  - history is embedded once per series from x_historical under the same padding
  - the row is concat(current, current - history), width 1024
"""
from __future__ import annotations

import numpy as np

CHR_WIN = 256
CHR_EMB_BATCH = 64      # windows per pipeline.embed call


def chronos_pad(values, win=CHR_WIN):
    a = np.asarray(values, dtype=np.float32)
    if len(a) >= win:
        return a[-win:]
    return np.concatenate((np.full(win - len(a), a[0], dtype=np.float32), a))


def _embed_batch(pipe, windows, torch):
    """Mean-pooled embeddings for a list of 1-D float32 windows, EOS included."""
    out = []
    for i in range(0, len(windows), CHR_EMB_BATCH):
        chunk = [torch.tensor(w, dtype=torch.float32) for w in windows[i:i + CHR_EMB_BATCH]]
        with torch.no_grad():
            emb, _ = pipe.embed(chunk)
        out.append(emb.mean(dim=1).float().numpy().astype(np.float64))
    return np.concatenate(out, 0) if out else np.zeros((0, 0))


def build_chronos_matrix(datasets, stride, cadence, pipe, torch):
    """Returns (X float32 (n_rows, 1024), y int8, n_embeddings) for the chronos booster."""
    windows, spans = [], []
    for _sid, x_hist, x_online, _tau in datasets:
        n = len(x_online)
        refresh = list(range(0, n, cadence)) if n else []
        start = len(windows)
        windows.append(chronos_pad(x_hist))                       # history first
        for r in refresh:
            windows.append(chronos_pad(x_online[:r + 1]))
        spans.append((start, len(refresh), n))

    embs = _embed_batch(pipe, windows, torch)
    dim = embs.shape[1] if embs.size else 0

    rows, labels = [], []
    for (start, n_ref, n), (_sid, _h, _o, tau) in zip(spans, datasets):
        if not n:
            continue
        hist = embs[start]
        onl = embs[start + 1:start + 1 + n_ref]
        idx = np.arange(0, n, stride)
        blk = idx // cadence                     # carried-forward: the LAST refresh at or before t
        cur = onl[blk]
        rows.append(np.concatenate((cur, cur - hist[None, :]), axis=1).astype(np.float32))
        lab = np.zeros(len(idx), np.int8)
        if tau is not None:
            lab[idx >= tau] = 1
        labels.append(lab)

    X = np.concatenate(rows, 0) if rows else np.zeros((0, 2 * dim), np.float32)
    y = np.concatenate(labels) if labels else np.zeros(0, np.int8)
    return X, y, len(windows)


def fit_chronos_booster(X, y, lgbm_params):
    """Same LGBM_PARAMS as the tree leg -- verified against the shipped artifact, which reads
    600 trees / lr 0.015 / num_leaves 63 / binary."""
    from lightgbm import LGBMClassifier

    assert X.shape[1] == 1024, f"chronos features must be 1024 wide, got {X.shape[1]}"
    return LGBMClassifier(**lgbm_params).fit(X, y).booster_
