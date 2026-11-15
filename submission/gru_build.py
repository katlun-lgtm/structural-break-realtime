"""GRU leg (G1): a recurrent sequence model over the same standardized stride-8 feature stream the
TCN reads. Different inductive bias (gated recurrent state vs dilated convolution); enters the blend
as its own CDF-rank leg. Ships inside the package: train_gru() is the producer, GruStream the server.

Cell = torch.nn.GRU semantics (r, z gates; n candidate), one layer, hidden GRU_HIDDEN, linear head.
"""
from __future__ import annotations

import numpy as np

GRU_STRIDE = 8          # steps once per 8 online points, like the TCN
GRU_HIDDEN = 64
GRU_EPOCHS = 12
GRU_BATCH = 64
GRU_LR = 1e-3
GRU_SEED = 500


def _build_module():
    import torch
    import torch.nn as nn

    class FeatGRU(nn.Module):
        def __init__(self, n_in=121, h=GRU_HIDDEN):
            super().__init__()
            self.gru = nn.GRU(n_in, h, batch_first=True)
            self.head = nn.Linear(h, 1)

        def forward(self, x):                     # x: (B, L, n_in) -> (B, L)
            out, _ = self.gru(x)
            return self.head(out).squeeze(-1)

    return FeatGRU


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


class GruStream:
    """Streaming numpy twin of FeatGRU: one call per stride-8 point; returns the logit."""

    __slots__ = ("Wih", "Whh", "bih", "bhh", "Wo", "bo", "h", "H")

    def __init__(self, w):
        W = {k: np.asarray(v, dtype=np.float32) for k, v in w.items()}
        self.Wih, self.Whh = W["gru.weight_ih_l0"], W["gru.weight_hh_l0"]
        self.bih, self.bhh = W["gru.bias_ih_l0"], W["gru.bias_hh_l0"]
        self.Wo, self.bo = W["head.weight"][0], W["head.bias"][0]
        self.H = self.Whh.shape[1]
        self.h = np.zeros(self.H, dtype=np.float32)

    def step(self, x):
        H = self.H
        gi = self.Wih @ x + self.bih
        gh = self.Whh @ self.h + self.bhh
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = ((1.0 - z) * n + z * self.h).astype(np.float32)
        return float(self.Wo @ self.h + self.bo)


def _fit_one(seqs, labels, n_in, seed, device):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    net = _build_module()(n_in=n_in).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=GRU_LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(seqs))
    net.train()
    for _ep in range(GRU_EPOCHS):
        idx = idx_all[rng.permutation(len(idx_all))]
        for i in range(0, len(idx), GRU_BATCH):
            grp = idx[i:i + GRU_BATCH]
            L = max(seqs[j].shape[0] for j in grp)
            X = np.zeros((len(grp), L, n_in), dtype=np.float32)
            Y = np.zeros((len(grp), L), dtype=np.float32)
            Mm = np.zeros((len(grp), L), dtype=np.float32)
            for bb, j in enumerate(grp):
                n = seqs[j].shape[0]
                X[bb, :n] = seqs[j]
                Y[bb, :n] = labels[j]
                Mm[bb, :n] = 1.0
            xb, yb, mb = (torch.from_numpy(a).to(device) for a in (X, Y, Mm))
            loss = (lossf(net(xb), yb) * mb).sum() / mb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    scores = []
    with torch.no_grad():
        for seq in seqs:
            x = torch.from_numpy(seq[None].astype(np.float32)).to(device)
            scores.append(net(x)[0].float().cpu().numpy().astype(np.float64))
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return w, (np.concatenate(scores) if scores else np.zeros(0))


def train_gru(seqs, labels, n_in):
    """Ship fit on all sequences at GRU_SEED. Returns (state_dict_numpy, in-sample scores) --
    the scores feed the q_gru quantile map, same convention as q_tcn."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _fit_one(seqs, labels, n_in, GRU_SEED, device)


def parity_check(w, seq, atol=1e-5):
    """Streaming numpy cell vs torch on one sequence; returns max abs logit difference."""
    import torch

    net = _build_module()(n_in=seq.shape[1])
    net.load_state_dict({k: torch.tensor(v) for k, v in w.items()})
    net.eval()
    with torch.no_grad():
        ref = net(torch.from_numpy(seq[None].astype(np.float32)))[0].numpy()
    st = GruStream(w)
    got = np.array([st.step(seq[t].astype(np.float32)) for t in range(seq.shape[0])])
    return float(np.abs(ref - got).max())
