"""Generic sequence leg for the discovery pipeline: one module that builds, trains and SERVES
recurrent (GRU/LSTM, N layers) and attention (N blocks) nets over the standardized stride-8 feature
stream. A leg is described by a spec dict {arch, hidden, blocks, layers, dropout, wd, epochs, seed}.
Ships inside the package: train_leg() is the producer, make_stream() the server; every streamer
derives its sizes from the weights it is given, so one module serves every spec.
"""
from __future__ import annotations

import math

import numpy as np

LN_EPS = 1e-5
ATT_HEADS = 4
ATT_MAXLEN = 256
BATCH = 64
LR = 1e-3
SERIES_W = None   # optional per-sequence null-row weights (list aligned with seqs), set by the caller before fit
AGES = None       # optional per-sequence age targets (list aligned with seqs; NaN where undefined), set by the caller before fit
RAWBLK_N = 8          # "features+rawblk8": the 8 raw points since the previous stride-8 step (history tail before t=7)
RAWBLK_CLIP = 10.0    # identical clip in training (box grid) and serving (ring buffer)
HISTNULL_PO = 768     # "aug": "histnull" -> pseudo-series carved from each UNBROKEN history: online = last 768 points
HISTNULL_MIN_H = 2560 #   (history = the rest), labels all 0; only histories of at least 2,560 points qualify


def histnull_pseudo(x_hist):
    """(pseudo_history, pseudo_online) carved from a history, or None if it is too short. Same rule as the box grid."""
    h = np.asarray(x_hist, dtype=np.float64)
    if len(h) < HISTNULL_MIN_H: return None
    return h[:-HISTNULL_PO], h[-HISTNULL_PO:]


SELFCAL_WS = (16, 32, 64, 128, 256)   # "selfcal5": z of log var(last W online pts) vs the series' own history distribution of W-window log vars
SELFCAL_CLIP = 8.0


def input_flags(spec):
    """(histcal, rawblk8) flags of a spec's input string. Canonical input order: features, histcal, rawblk8, selfcal5."""
    inp = str(spec.get("input", "features"))
    return ("histcal" in inp, "rawblk8" in inp)


def selfcal_flag(spec):
    return "selfcal5" in str(spec.get("input", "features"))


def n_in_for(spec, base):
    hc, rb = input_flags(spec)
    return int(base) + (len(HISTCAL_NAMES) if hc else 0) + (RAWBLK_N if rb else 0) + (len(SELFCAL_WS) if selfcal_flag(spec) else 0) + (1 if "treescore" in str(spec.get("input", "features")) else 0)


def selfcal_consts(x_hist):
    """Per-series (mu_W, sd_W) of log var over non-overlapping W-windows of the history, for W in SELFCAL_WS."""
    h = np.asarray(x_hist, dtype=np.float64); out = []
    for W in SELFCAL_WS:
        n = len(h) // W
        if n >= 4:
            lv = np.log(np.array([h[i * W:(i + 1) * W].var() for i in range(n)]) + 1e-9); out += [float(lv.mean()), float(lv.std() + 1e-3)]
        else: out += [0.0, 1.0]
    return np.array(out, dtype=np.float64)


def selfcal_vec(ring, consts):
    """z_W for the last W points of `ring` (a sequence of the most recent >= 256 raw points, history tail first)."""
    a = np.asarray(ring, dtype=np.float64); out = np.empty(len(SELFCAL_WS), np.float32)
    for wi, W in enumerate(SELFCAL_WS):
        seg = a[-W:]; out[wi] = np.clip((np.log(seg.var() + 1e-9) - consts[2 * wi]) / consts[2 * wi + 1], -SELFCAL_CLIP, SELFCAL_CLIP)
    return out


def rawblk_vec(vals):
    a = np.asarray(list(vals), dtype=np.float64); assert a.shape == (RAWBLK_N,), a.shape
    return np.clip(a, -RAWBLK_CLIP, RAWBLK_CLIP).astype(np.float32)


def rawblk_seq(x_hist, x_on, n_rows, stride=8):
    """(n_rows, 8) raw blocks for grid rows t = 0, stride, 2*stride, ...: values x[t-7..t] of history+online
    (zero left-pad if the history is shorter than 7). Must match the box grid builder exactly."""
    assert stride == RAWBLK_N, stride
    h = np.asarray(x_hist, dtype=np.float64); o = np.asarray(x_on, dtype=np.float64)
    full = np.concatenate([np.zeros(max(0, RAWBLK_N - 1 - len(h))), h, o]); n_h = len(full) - len(o)
    out = np.empty((int(n_rows), RAWBLK_N), dtype=np.float32)
    for k in range(int(n_rows)):
        t = k * stride; assert t < len(o), (t, len(o), n_rows)
        out[k] = np.clip(full[n_h + t - (RAWBLK_N - 1): n_h + t + 1], -RAWBLK_CLIP, RAWBLK_CLIP)
    return out


def assemble(x, hc_vec=None, rb_vec=None, sc_vec=None):
    parts = [np.asarray(x, dtype=np.float32)]
    if hc_vec is not None: parts.append(np.asarray(hc_vec, dtype=np.float32))
    if rb_vec is not None: parts.append(np.asarray(rb_vec, dtype=np.float32))
    if sc_vec is not None: parts.append(np.asarray(sc_vec, dtype=np.float32))
    return np.concatenate(parts)


def assemble_seq(seq, hc_vec=None, rb_mat=None, sc_mat=None):
    parts = [np.asarray(seq, dtype=np.float32)]
    if hc_vec is not None: parts.append(np.repeat(np.asarray(hc_vec, dtype=np.float32)[None], seq.shape[0], 0))
    if rb_mat is not None: assert rb_mat.shape == (seq.shape[0], RAWBLK_N), (rb_mat.shape, seq.shape); parts.append(np.asarray(rb_mat, dtype=np.float32))
    if sc_mat is not None: assert sc_mat.shape == (seq.shape[0], len(SELFCAL_WS)), (sc_mat.shape, seq.shape); parts.append(np.asarray(sc_mat, dtype=np.float32))
    return np.concatenate(parts, axis=1)


def _build_module(spec, n_in=121):
    import torch
    import torch.nn as nn

    arch = spec["arch"]; h = int(spec["hidden"]); layers = int(spec.get("layers", 1)); blocks = int(spec.get("blocks", 1)); drop = float(spec.get("dropout", 0.0))
    aux = spec.get("aux") == "age"

    class Rnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
            cell = nn.GRU if arch == "gru" else nn.LSTM
            self.rnn = cell(n_in, h, num_layers=layers, batch_first=True, dropout=(drop if layers > 1 else 0.0))
            self.head = nn.Linear(h, 1)
            if aux: self.head_age = nn.Linear(h, 1)          # auxiliary time-since-break head (training only; streams ignore it)

        def forward(self, x, aux_out=False):
            out, _ = self.rnn(self.drop(x)); z = self.head(out).squeeze(-1)
            return (z, self.head_age(out).squeeze(-1)) if aux_out else z

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.d, self.h = h, ATT_HEADS
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
            self.proj = nn.Linear(n_in, h); self.pos = nn.Embedding(ATT_MAXLEN, h)
            self.blocks = nn.ModuleList()
            for _ in range(blocks):
                b = nn.Module(); b.ln1 = nn.LayerNorm(h, eps=LN_EPS)
                b.q = nn.Linear(h, h); b.k = nn.Linear(h, h); b.v = nn.Linear(h, h); b.o = nn.Linear(h, h)
                b.ln2 = nn.LayerNorm(h, eps=LN_EPS); b.f1 = nn.Linear(h, 2 * h); b.f2 = nn.Linear(2 * h, h)
                self.blocks.append(b)
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            B, L, _ = x.shape; dh = self.d // self.h
            idx = torch.arange(L, device=x.device).clamp(max=ATT_MAXLEN - 1)
            hh = self.proj(self.drop(x)) + self.pos(idx)[None]
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            for b in self.blocks:
                a = b.ln1(hh)
                q = b.q(a).view(B, L, self.h, dh).transpose(1, 2); k = b.k(a).view(B, L, self.h, dh).transpose(1, 2); v = b.v(a).view(B, L, self.h, dh).transpose(1, 2)
                s = (q @ k.transpose(-1, -2)) / math.sqrt(dh); s = s.masked_fill(mask, float("-inf"))
                att = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, L, self.d)
                hh = hh + b.o(att); f = b.ln2(hh); hh = hh + b.f2(torch.nn.functional.gelu(b.f1(f)))
            return self.head(hh).squeeze(-1)

    class Tcn(nn.Module):
        """FeatTCN family: 1x1 proj -> residual blocks [causal dilated conv k=3, channel-LayerNorm (channels
        only; never the time axis), GELU, 1x1 conv] -> 1x1 head. Dilations 1,2,4,... (blocks of them)."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
            self.proj = nn.Conv1d(n_in, h, 1)
            self.dils = [2 ** i for i in range(blocks)]
            self.cconv = nn.ModuleList([nn.Conv1d(h, h, 3, dilation=d, padding=0) for d in self.dils])
            self.lng = nn.ParameterList([nn.Parameter(torch.ones(1, h, 1)) for _ in self.dils])
            self.lnb = nn.ParameterList([nn.Parameter(torch.zeros(1, h, 1)) for _ in self.dils])
            self.pw = nn.ModuleList([nn.Conv1d(h, h, 1) for _ in self.dils])
            self.head = nn.Conv1d(h, 1, 1)
            if aux: self.head_age = nn.Conv1d(h, 1, 1)

        def forward(self, x, aux_out=False):                    # x: (B, L, n_in) -> (B, L)
            hh = self.proj(self.drop(x).transpose(1, 2))
            for d, cc, g, b, pw in zip(self.dils, self.cconv, self.lng, self.lnb, self.pw):
                z = cc(nn.functional.pad(hh, (2 * d, 0)))
                mu = z.mean(dim=1, keepdim=True); sd = z.std(dim=1, keepdim=True) + 1e-5
                hh = hh + pw(nn.functional.gelu((z - mu) / sd * g + b))
            zz = self.head(hh).squeeze(1)
            return (zz, self.head_age(hh).squeeze(1)) if aux_out else zz

    class Cusum(nn.Module):
        """Neural CUSUM: evidence e_t = MLP(x_t); S_t = softplus-relaxed max(0, S_{t-1} + e_t); logit = w*S_t + head(x_t).
        Absorbing-state prior: accumulated evidence cannot be forgotten except through negative evidence."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
            self.ev = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
            self.local = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, 1))
            self.wS = nn.Parameter(torch.tensor(1.0)); self.b = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, aux_out=False):                    # x: (B, L, n_in)
            xd = self.drop(x); e = self.ev(xd).squeeze(-1); loc = self.local(xd).squeeze(-1)
            S = torch.zeros(x.shape[0], device=x.device); outs = []
            for t in range(x.shape[1]):
                S = torch.clamp(S + e[:, t], min=0.0); outs.append(S)                        # exact CUSUM recursion max(0, S + e)
            St = torch.stack(outs, 1); z = self.wS * St + loc + self.b
            return (z, z) if aux_out else z

    class Lru(nn.Module):
        """Linear recurrent unit / diagonal SSM leg: per layer h_t = lam * h_{t-1} + gam * (W x_t) with complex-diagonal lam
        (learned decay spectrum), y = GELU(Re(C h) + b) + residual MLP. `blocks` layers, `hidden` complex channels."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity(); self.inp = nn.Linear(n_in, h); L_ = max(1, blocks)
            self.nu = nn.ParameterList([nn.Parameter(torch.log(-torch.log(torch.rand(h) * 0.5 + 0.4))) for _ in range(L_)])   # |lam| in ~[0.4, 0.9]
            self.th = nn.ParameterList([nn.Parameter(torch.rand(h) * 0.5) for _ in range(L_)])                                  # phase
            self.Bw = nn.ModuleList([nn.Linear(h, 2 * h) for _ in range(L_)]); self.Cw = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(L_)])
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h)) for _ in range(L_)])
            self.ln = nn.ModuleList([nn.LayerNorm(h, eps=LN_EPS) for _ in range(L_)]); self.head = nn.Linear(h, 1)
            if aux: self.head_age = nn.Linear(h, 1)

        def forward(self, x, aux_out=False):
            hh = self.inp(self.drop(x))                                            # (B, L, h)
            for l in range(len(self.nu)):
                mod = torch.exp(-torch.exp(self.nu[l])); lam_r, lam_i = mod * torch.cos(self.th[l]), mod * torch.sin(self.th[l])
                gam = torch.sqrt(1 - mod ** 2)
                u = self.Bw[l](self.ln[l](hh)); ur, ui = u[..., :hh.shape[-1]] * gam, u[..., hh.shape[-1]:] * gam
                sr = torch.zeros(x.shape[0], hh.shape[-1], device=x.device); si = torch.zeros_like(sr); outs = []
                for t in range(x.shape[1]):
                    sr, si = lam_r * sr - lam_i * si + ur[:, t], lam_r * si + lam_i * sr + ui[:, t]; outs.append(torch.cat([sr, si], -1))
                y = self.Cw[l](torch.stack(outs, 1)); hh = hh + y; hh = hh + self.ff[l](hh)
            z = self.head(hh).squeeze(-1)
            return (z, self.head_age(hh).squeeze(-1)) if aux_out else z

    class TcnGru(nn.Module):
        """Dilated-conv front end (blocks dilations) feeding a GRU: early-t conv evidence, late-t recurrent integration in one leg."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity(); self.proj = nn.Conv1d(n_in, h, 1); self.dils = [2 ** i for i in range(blocks)]
            self.cconv = nn.ModuleList([nn.Conv1d(h, h, 3, dilation=d, padding=0) for d in self.dils])
            self.lng = nn.ParameterList([nn.Parameter(torch.ones(1, h, 1)) for _ in self.dils]); self.lnb = nn.ParameterList([nn.Parameter(torch.zeros(1, h, 1)) for _ in self.dils])
            self.pw = nn.ModuleList([nn.Conv1d(h, h, 1) for _ in self.dils]); self.rnn = nn.GRU(h, h, num_layers=1, batch_first=True); self.head = nn.Linear(h, 1)
            if aux: self.head_age = nn.Linear(h, 1)

        def forward(self, x, aux_out=False):
            hh = self.proj(self.drop(x).transpose(1, 2))
            for d, cc, g, b, pw in zip(self.dils, self.cconv, self.lng, self.lnb, self.pw):
                z = cc(nn.functional.pad(hh, (2 * d, 0))); mu = z.mean(dim=1, keepdim=True); sd = z.std(dim=1, keepdim=True) + 1e-5
                hh = hh + pw(nn.functional.gelu((z - mu) / sd * g + b))
            out, _ = self.rnn(hh.transpose(1, 2)); z = self.head(out).squeeze(-1)
            return (z, self.head_age(out).squeeze(-1)) if aux_out else z

    class EmaBank(nn.Module):
        """Per-step embedding integrated by K exponential moving averages with learned rates; head over [emb, ema_1..ema_K]."""
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity(); self.K = max(2, blocks)
            self.emb = nn.Sequential(nn.Linear(n_in, h), nn.GELU(), nn.Linear(h, h)); self.alpha_logit = nn.Parameter(torch.linspace(-3.0, 1.5, self.K))   # rates ~0.05..0.8
            self.head = nn.Sequential(nn.Linear(h * (self.K + 1), h), nn.GELU(), nn.Linear(h, 1))
            if aux: self.head_age = nn.Linear(h * (self.K + 1), 1)

        def forward(self, x, aux_out=False):
            e = self.emb(self.drop(x)); a = torch.sigmoid(self.alpha_logit)                  # (B, L, h), (K,)
            states = [torch.zeros(x.shape[0], e.shape[-1], device=x.device) for _ in range(self.K)]; outs = []
            for t in range(x.shape[1]):
                states = [(1 - a[k]) * states[k] + a[k] * e[:, t] for k in range(self.K)]; outs.append(torch.cat([e[:, t]] + states, -1))
            feat = torch.stack(outs, 1); z = self.head(feat).squeeze(-1)
            return (z, self.head_age(feat).squeeze(-1)) if aux_out else z

    return {"attn": Attn, "tcn": Tcn, "cusum": Cusum, "lru": Lru, "tcngru": TcnGru, "ema": EmaBank}.get(arch, Rnn)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


_erf = np.frompyfunc(math.erf, 1, 1)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float64))


def _ln(x, g, b):
    m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / np.sqrt(v + LN_EPS) * g + b


class RnnStream:
    """Streaming numpy twin of Rnn (GRU or LSTM, N layers): one call per stride-8 point -> logit."""

    def __init__(self, w, arch):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
        self.arch = arch; self.L = 1 + max(int(k.split("_l")[-1]) for k in self.W if k.startswith("rnn.weight_ih_l"))
        self.H = self.W["rnn.weight_hh_l0"].shape[1]
        self.h = [np.zeros(self.H) for _ in range(self.L)]; self.c = [np.zeros(self.H) for _ in range(self.L)]

    def step(self, x):
        W = self.W; inp = np.asarray(x, dtype=np.float64); H = self.H
        for l in range(self.L):
            gi = W[f"rnn.weight_ih_l{l}"] @ inp + W[f"rnn.bias_ih_l{l}"]; gh = W[f"rnn.weight_hh_l{l}"] @ self.h[l] + W[f"rnn.bias_hh_l{l}"]
            if self.arch == "gru":
                r = _sigmoid(gi[:H] + gh[:H]); z = _sigmoid(gi[H:2 * H] + gh[H:2 * H]); n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
                self.h[l] = (1.0 - z) * n + z * self.h[l]
            else:
                g = gi + gh; i = _sigmoid(g[:H]); f = _sigmoid(g[H:2 * H]); gg = np.tanh(g[2 * H:3 * H]); o = _sigmoid(g[3 * H:])
                self.c[l] = f * self.c[l] + i * gg; self.h[l] = o * np.tanh(self.c[l])
            inp = self.h[l]
        return float(W["head.weight"][0] @ inp + W["head.bias"][0])


class AttnStream:
    """Streaming numpy twin of Attn (N blocks, K/V cache per block): one call per stride-8 point -> logit."""

    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
        self.d = self.W["proj.weight"].shape[0]; self.h = ATT_HEADS; self.dh = self.d // self.h; self.maxlen = self.W["pos.weight"].shape[0]
        self.nb = 1 + max(int(k.split(".")[1]) for k in self.W if k.startswith("blocks."))
        self.t = 0; self.K = [[] for _ in range(self.nb)]; self.V = [[] for _ in range(self.nb)]

    def step(self, x):
        W = self.W; x = np.asarray(x, dtype=np.float64)
        hh = W["proj.weight"] @ x + W["proj.bias"] + W["pos.weight"][min(self.t, self.maxlen - 1)]
        for b in range(self.nb):
            p = f"blocks.{b}."; a = _ln(hh, W[p + "ln1.weight"], W[p + "ln1.bias"])
            q = (W[p + "q.weight"] @ a + W[p + "q.bias"]).reshape(self.h, self.dh); k = (W[p + "k.weight"] @ a + W[p + "k.bias"]).reshape(self.h, self.dh); v = (W[p + "v.weight"] @ a + W[p + "v.bias"]).reshape(self.h, self.dh)
            self.K[b].append(k); self.V[b].append(v); K = np.stack(self.K[b], axis=1); V = np.stack(self.V[b], axis=1)
            s = np.einsum("hd,htd->ht", q, K) / math.sqrt(self.dh); s = s - s.max(axis=1, keepdims=True); e = np.exp(s); wgt = e / e.sum(axis=1, keepdims=True)
            att = np.einsum("ht,htd->hd", wgt, V).reshape(self.d)
            hh = hh + W[p + "o.weight"] @ att + W[p + "o.bias"]; f = _ln(hh, W[p + "ln2.weight"], W[p + "ln2.bias"])
            hh = hh + W[p + "f2.weight"] @ _gelu(W[p + "f1.weight"] @ f + W[p + "f1.bias"]) + W[p + "f2.bias"]
        self.t += 1
        return float(W["head.weight"][0] @ hh + W["head.bias"][0])


class TcnStream:
    """Streaming numpy twin of Tcn: ring buffers == the convs' left zero-padding (as the shipped _TcnState)."""

    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
        self.nb = 1 + max(int(k.split(".")[1]) for k in self.W if k.startswith("cconv."))
        self.c = self.W["proj.weight"].shape[0]; self.dils = [2 ** i for i in range(self.nb)]
        self.bufs = [np.zeros((2 * d + 1, self.c)) for d in self.dils]; self.ptrs = [0] * self.nb

    def step(self, x):
        W = self.W; h = W["proj.weight"][:, :, 0] @ np.asarray(x, dtype=np.float64) + W["proj.bias"]
        for bi, d in enumerate(self.dils):
            buf, p = self.bufs[bi], self.ptrs[bi]; buf[p] = h; n = 2 * d + 1; cw = W[f"cconv.{bi}.weight"]
            z = cw[:, :, 0] @ buf[(p - 2 * d) % n] + cw[:, :, 1] @ buf[(p - d) % n] + cw[:, :, 2] @ buf[p] + W[f"cconv.{bi}.bias"]
            mu = z.mean(); sd = z.std(ddof=1) + 1e-5
            zn = (z - mu) / sd * W[f"lng.{bi}"][0, :, 0] + W[f"lnb.{bi}"][0, :, 0]
            h = h + W[f"pw.{bi}.weight"][:, :, 0] @ _gelu(zn) + W[f"pw.{bi}.bias"]
            self.ptrs[bi] = (p + 1) % n
        return float(W["head.weight"][0, :, 0] @ h + W["head.bias"][0])


HISTCAL_NAMES = ["w64", "w128", "w256", "w512", "kurt", "ar1", "skew", "absmean", "lrough"]


def histcal(x_hist):
    """Nine per-series constants from the HISTORY only (causal, label-free): sd of log rolling variance over
    non-overlapping windows of 64/128/256/512, kurtosis, lag-1 autocorrelation, skew, mean |x|, log mean
    squared increment. Standardised with the training set's mean/std (shipped) before entering a leg."""
    h = np.asarray(x_hist, dtype=np.float64)
    def wander(W):
        n = len(h) // W
        if n < 2: return 0.0
        v = np.array([h[i * W:(i + 1) * W].var() for i in range(n)]); return float(np.log(v + 1e-9).std())
    c = h - h.mean(); v = max(float(h.var()), 1e-12); d = np.diff(h)
    return np.array([wander(64), wander(128), wander(256), wander(512), float((c ** 4).mean() / v ** 2),
                     float((c[1:] * c[:-1]).mean() / v), float((c ** 3).mean() / v ** 1.5), float(np.abs(h).mean()), float(np.log((d * d).mean() + 1e-9))])


class CusumStream:
    """Streaming numpy twin of Cusum: one call per stride-8 point -> logit."""

    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}; self.S = 0.0

    def _mlp(self, prefix, x, n):
        h = x
        for i in range(n):
            h = self.W[f"{prefix}.{2*i}.weight"] @ h + self.W[f"{prefix}.{2*i}.bias"]
            if i < n - 1: h = _gelu(h)
        return h

    def step(self, x):
        x = np.asarray(x, dtype=np.float64); e = float(self._mlp("ev", x, 3)[0]); loc = float(self._mlp("local", x, 2)[0])
        self.S = max(0.0, self.S + e)
        return float(self.W["wS"] * self.S + loc + self.W["b"])


def _lin(W, p, x):
    return W[p + ".weight"] @ x + W[p + ".bias"]


class LruStream:
    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}; self.L = 1 + max(int(k.split(".")[1]) for k in self.W if k.startswith("nu."))
        h = self.W["inp.weight"].shape[0]; self.sr = [np.zeros(h) for _ in range(self.L)]; self.si = [np.zeros(h) for _ in range(self.L)]

    def step(self, x):
        W = self.W; hh = _lin(W, "inp", np.asarray(x, dtype=np.float64))
        for l in range(self.L):
            mod = np.exp(-np.exp(W[f"nu.{l}"])); lr, li = mod * np.cos(W[f"th.{l}"]), mod * np.sin(W[f"th.{l}"]); gam = np.sqrt(1 - mod ** 2)
            u = _lin(W, f"Bw.{l}", _ln(hh, W[f"ln.{l}.weight"], W[f"ln.{l}.bias"])); n = len(hh); ur, ui = u[:n] * gam, u[n:] * gam
            sr, si = lr * self.sr[l] - li * self.si[l] + ur, lr * self.si[l] + li * self.sr[l] + ui; self.sr[l], self.si[l] = sr, si
            hh = hh + _lin(W, f"Cw.{l}", np.concatenate([sr, si])); hh = hh + _lin(W, f"ff.{l}.2", _gelu(_lin(W, f"ff.{l}.0", hh)))
        return float(W["head.weight"][0] @ hh + W["head.bias"][0])


class TcnGruStream:
    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}; self.nb = 1 + max(int(k.split(".")[1]) for k in self.W if k.startswith("cconv."))
        self.c = self.W["proj.weight"].shape[0]; self.dils = [2 ** i for i in range(self.nb)]; self.bufs = [np.zeros((2 * d + 1, self.c)) for d in self.dils]; self.ptrs = [0] * self.nb
        self.rnn = RnnStream({k: v for k, v in w.items() if k.startswith("rnn.") or k.startswith("head.")}, "gru")

    def step(self, x):
        W = self.W; h = W["proj.weight"][:, :, 0] @ np.asarray(x, dtype=np.float64) + W["proj.bias"]
        for bi, d in enumerate(self.dils):
            buf, p = self.bufs[bi], self.ptrs[bi]; buf[p] = h; n = 2 * d + 1; cw = W[f"cconv.{bi}.weight"]
            z = cw[:, :, 0] @ buf[(p - 2 * d) % n] + cw[:, :, 1] @ buf[(p - d) % n] + cw[:, :, 2] @ buf[p] + W[f"cconv.{bi}.bias"]
            mu = z.mean(); sd = z.std(ddof=1) + 1e-5; zn = (z - mu) / sd * W[f"lng.{bi}"][0, :, 0] + W[f"lnb.{bi}"][0, :, 0]
            h = h + W[f"pw.{bi}.weight"][:, :, 0] @ _gelu(zn) + W[f"pw.{bi}.bias"]; self.ptrs[bi] = (p + 1) % n
        return self.rnn.step(h)


class EmaStream:
    def __init__(self, w):
        self.W = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}; self.a = 1.0 / (1.0 + np.exp(-self.W["alpha_logit"])); self.K = len(self.a)
        h = self.W["emb.2.weight"].shape[0]; self.states = [np.zeros(h) for _ in range(self.K)]

    def step(self, x):
        W = self.W; e = _lin(W, "emb.2", _gelu(_lin(W, "emb.0", np.asarray(x, dtype=np.float64))))
        self.states = [(1 - self.a[k]) * self.states[k] + self.a[k] * e for k in range(self.K)]
        f = np.concatenate([e] + self.states); return float(_lin(W, "head.2", _gelu(_lin(W, "head.0", f)))[0])


class EnsStream:
    """Seed-averaged leg: the mean logit of member streams (weights carry an e{i}. prefix)."""

    def __init__(self, streams):
        self.streams = list(streams)

    def step(self, x):
        return float(np.mean([st.step(x) for st in self.streams]))


def n_members(spec):
    return int(spec.get("nseeds", 1))


def member_seed(seed, i):
    return int(seed) + 100 * int(i)


def split_members(w):
    """Flat weight dict -> list of per-member dicts (a single net is its own list of one)."""
    if not any(str(k).startswith("e0.") for k in w): return [dict(w)]
    members = {}
    for k, v in w.items():
        i, rest = str(k).split(".", 1); members.setdefault(int(i[1:]), {})[rest] = v
    return [members[i] for i in sorted(members)]


def _single_stream(w, spec):
    a = spec["arch"]
    return {"attn": AttnStream, "tcn": TcnStream, "cusum": CusumStream, "lru": LruStream, "tcngru": TcnGruStream, "ema": EmaStream}.get(a, lambda ww: RnnStream(ww, a))(w)


def make_stream(w, spec):
    streams = [_single_stream(m, spec) for m in split_members(w)]
    return streams[0] if len(streams) == 1 else EnsStream(streams)


def fit(spec, seqs, labels, n_in, seed, device):
    """Masked BCE over padded batches; input dropout + weight decay from the spec. Returns
    (state_dict as numpy float32, in-sample scores concatenated over seqs)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    net = _build_module(spec, n_in)().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=float(spec.get("wd", 0.0)))
    lossf = nn.BCEWithLogitsLoss(reduction="none"); rng = np.random.default_rng(seed); idx_all = np.arange(len(seqs))
    rank_alpha = float(spec.get("rank_alpha", 0.0)) if spec.get("loss") == "rank" else 0.0   # per-t pairwise ranking loss (the metric is a cross-sectional AUC at each t)
    bs = int(spec.get("batch", BATCH))
    # optional per-step weights ∝ 1/(series alive at t): the metric weights every t equally, plain BCE weights long series' late steps most
    focal_gamma = float(spec.get("focal", 0.0)); aux_age = spec.get("aux") == "age"; aux_beta = float(spec.get("aux_beta", 0.1))
    sw = SERIES_W if spec.get("tw") == "wander" else None                       # set by the caller (box_fit) for the current seqs order
    tw = None
    if spec.get("tw") == "inv_alive":
        Lmax = max(q.shape[0] for q in seqs); alive = np.zeros(Lmax, np.float64)
        for q in seqs: alive[:q.shape[0]] += 1.0
        tw = (1.0 / np.maximum(alive, 1.0)); tw = (tw / tw.mean()).astype(np.float32)
    net.train()
    for _ep in range(int(spec.get("epochs", 12))):
        idx = idx_all[rng.permutation(len(idx_all))]
        for i in range(0, len(idx), bs):
            grp = idx[i:i + bs]; L = max(seqs[j].shape[0] for j in grp)
            X = np.zeros((len(grp), L, n_in), dtype=np.float32); Y = np.zeros((len(grp), L), dtype=np.float32); M = np.zeros((len(grp), L), dtype=np.float32)
            for bb, j in enumerate(grp):
                n = seqs[j].shape[0]; X[bb, :n] = seqs[j]; lab = np.asarray(labels[j], dtype=np.float32)
                fin = np.isfinite(lab); Y[bb, :n] = np.where(fin, lab, 0.0); M[bb, :n] = fin.astype(np.float32)   # NaN label = warm-up row, excluded from the loss
            if tw is not None: M = M * tw[None, :L]
            if sw is not None:                                                     # per-series null-row weights (wander)
                for bb, j in enumerate(grp): M[bb] = M[bb] * np.where(Y[bb] < 0.5, sw[j], 1.0)
            xb, yb, mb = (torch.from_numpy(a).to(device) for a in (X, Y, M))
            if aux_age:
                A = np.full((len(grp), L), np.nan, dtype=np.float32)
                for bb, j in enumerate(grp): A[bb, :seqs[j].shape[0]] = AGES[j]
                ab = torch.from_numpy(A).to(device); out, age_out = net(xb, aux_out=True)
                am = torch.isfinite(ab) & (mb > 0); loss_age = ((age_out - torch.nan_to_num(ab)) ** 2 * am).sum() / am.sum().clamp(min=1)
            else:
                out = net(xb); loss_age = 0.0
            per = lossf(out, yb)
            if focal_gamma > 0:
                pt = torch.exp(-per); per = ((1 - pt) ** focal_gamma) * per
            loss = (per * mb).sum() / mb.sum() + aux_beta * loss_age
            if rank_alpha > 0:
                # for every t: all (positive, negative) pairs of series present at t; logistic pairwise loss on the logit gap
                pos = (yb > 0.5) & (mb > 0); neg = (yb < 0.5) & (mb > 0)                               # (B, L)
                gap = out[:, None, :] - out[None, :, :]                                                   # (B, B, L): s_i - s_j
                pair = (pos[:, None, :] & neg[None, :, :]).float()
                if pair.sum() > 0: loss = loss + rank_alpha * (torch.nn.functional.softplus(-gap) * pair).sum() / pair.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval(); scores = []
    with torch.no_grad():
        for s in seqs:
            scores.append(net(torch.from_numpy(s[None].astype(np.float32)).to(device))[0].float().cpu().numpy().astype(np.float64))
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return w, (np.concatenate(scores) if scores else np.zeros(0))


def fit_ens(spec, seqs, labels, n_in, seed, device):
    """spec['nseeds'] members at seeds seed, seed+100, ...; flat weights with e{i}. prefixes; mean in-sample logit."""
    k = n_members(spec)
    if k == 1: return fit(spec, seqs, labels, n_in, seed, device)
    ws, ss = [], []
    for i in range(k):
        w, sc = fit(spec, seqs, labels, n_in, member_seed(seed, i), device); ws.append(w); ss.append(sc)
    flat = {f"e{i}.{kk}": v for i, w in enumerate(ws) for kk, v in w.items()}
    return flat, np.mean(np.stack(ss, 0), axis=0)


def load_nets(w, spec, n_in, device="cpu"):
    import torch
    nets = []
    for m in split_members(w):
        net = _build_module(spec, n_in)().to(device); net.load_state_dict({k: torch.tensor(np.asarray(v)) for k, v in m.items()}); net.eval(); nets.append(net)
    return nets


def predict(nets, seq, device="cpu"):
    """Mean member logit over one sequence (L, n_in) -> (L,) float64."""
    import torch
    x = torch.from_numpy(np.asarray(seq, dtype=np.float32)[None]).to(device)
    with torch.no_grad():
        outs = [n(x)[0].float().cpu().numpy() for n in nets]
    return np.mean(np.stack(outs, 0), axis=0).astype(np.float64)


def train_leg(spec, seqs, labels, n_in):
    """Ship fit at spec['seed'] (the producer); seed-averaged when spec['nseeds'] > 1."""
    import torch
    return fit_ens(spec, seqs, labels, n_in, int(spec["seed"]), "cuda" if torch.cuda.is_available() else "cpu")


def parity_check(w, spec, seq):
    ref = predict(load_nets(w, spec, seq.shape[1]), seq)
    st = make_stream(w, spec); got = np.array([st.step(seq[t]) for t in range(seq.shape[0])])
    return float(np.abs(ref - got).max())
