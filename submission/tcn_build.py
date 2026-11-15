"""CLOSURE: rebuild the causal FeatTCN third leg inside train().

Architecture ported from featseq2_2026-08-12.py:100-145 and verified tensor-for-tensor against the
shipped bundle (40 tensors: proj 2 + 6 blocks x 6 + head 2, c=64, n_in=121, k=3, dil 1..32).

Two deliberate deviations from the historical build, both recorded rather than hidden:
  1. CPU, not .cuda(). Every platform run of ours is runtimeType: CPU. The rebuilt net is therefore
     NOT bit-identical to the shipped one; parity for this key is score-level by decision
     (Katlun 2026-09-10).
  2. Feature sequences are built on the STRIDE-8 grid, because that is the grid infer() steps the
     net on ((_t_on & 7) == 0). The tree/MLP path uses stride 3; the two grids do not align, so the
     TCN needs its own feature pass rather than a subsample of the stride-3 rows.
"""
from __future__ import annotations

import numpy as np

TCN_STRIDE = 8          # matches infer(): the net steps once per 8 online points
TCN_CHANNELS = 64
TCN_DILATIONS = (1, 2, 4, 8, 16, 32)
TCN_KERNEL = 3
TCN_EPOCHS = 12         # featseq2_2026-08-12.py:30
TCN_BATCH = 64          # :31
TCN_LR = 1e-3           # :32
TCN_SEED = 0            # :29


def _build_module():
    """Defined inside a function so importing this module never costs a torch import."""
    import torch
    import torch.nn as nn

    class CausalConv1d(nn.Module):
        def __init__(self, c_in, c_out, k, dilation):
            super().__init__()
            self.left = (k - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation, padding=0)

        def forward(self, x):
            return self.conv(nn.functional.pad(x, (self.left, 0)))

    class ChannelLayerNorm(nn.Module):
        """Over CHANNELS ONLY. GroupNorm(1, c) would mix the time axis in and break causality
        (arm 119 was voided for exactly that: output[t] depended on t' > t)."""

        def __init__(self, c):
            super().__init__()
            self.g = nn.Parameter(torch.ones(1, c, 1))
            self.b = nn.Parameter(torch.zeros(1, c, 1))

        def forward(self, x):
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True) + 1e-5
            return (x - mu) / sd * self.g + self.b

    class Block(nn.Module):
        def __init__(self, c, d, k=TCN_KERNEL):
            super().__init__()
            self.cconv = CausalConv1d(c, c, k, d)
            self.ln = ChannelLayerNorm(c)
            self.pw = nn.Conv1d(c, c, 1)

        def forward(self, x):
            return x + self.pw(nn.functional.gelu(self.ln(self.cconv(x))))

    class FeatTCN(nn.Module):
        def __init__(self, n_in=121, c=TCN_CHANNELS, dilations=TCN_DILATIONS, k=TCN_KERNEL):
            super().__init__()
            self.proj = nn.Conv1d(n_in, c, 1)
            self.blocks = nn.ModuleList([Block(c, d, k) for d in dilations])
            self.head = nn.Conv1d(c, 1, 1)

        def forward(self, x):
            h = self.proj(x)
            for b in self.blocks:
                h = b(h)
            return self.head(h).squeeze(1)

    return FeatTCN


def train_tcn(seqs, labels, tcn_dim):
    """seqs: list of (T_s, tcn_dim) float32 arrays, ALREADY standardised. labels: list of (T_s,).

    Returns (state_dict_as_numpy, per_series_scores) -- scores are the ship net's own in-sample
    outputs, which is what the shipped q_tcn is built from (tcn16_build_2026-08-16.py:197-201).
    That is asymmetric with q_cb (out-of-fold); the asymmetry is in the original and is reproduced,
    not corrected.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(TCN_SEED)
    net = _build_module()(n_in=tcn_dim)
    opt = torch.optim.Adam(net.parameters(), lr=TCN_LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(TCN_SEED)
    idx_all = np.arange(len(seqs))

    net.train()
    for _ep in range(TCN_EPOCHS):
        idx = idx_all[rng.permutation(len(idx_all))]
        for i in range(0, len(idx), TCN_BATCH):
            grp = idx[i:i + TCN_BATCH]
            L = max(seqs[j].shape[0] for j in grp)
            X = np.zeros((len(grp), tcn_dim, L), dtype=np.float32)
            Y = np.zeros((len(grp), L), dtype=np.float32)
            Mm = np.zeros((len(grp), L), dtype=np.float32)
            for bb, j in enumerate(grp):
                n = seqs[j].shape[0]
                X[bb, :, :n] = seqs[j].T
                Y[bb, :n] = labels[j]
                Mm[bb, :n] = 1.0
            out = net(torch.from_numpy(X))
            loss = (lossf(out, torch.from_numpy(Y)) * torch.from_numpy(Mm)).sum() / Mm.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()

    scores = []
    with torch.no_grad():
        for s in seqs:
            x = torch.from_numpy(s.T[None].astype(np.float32))
            scores.append(net(x)[0].numpy().astype(np.float64))

    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return w, np.concatenate(scores) if scores else np.zeros(0)
