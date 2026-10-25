"""
Batched torch routes for decoding: log partition, exact single-pass sampling and membership, all over class
log-weights lw of shape (B, L, A) (revealed positions one-hot: 0 / -inf).

    TorchChain  automaton: scaled forward pass, FFBS sampling          (regular, counting, motifs)
    TorchChart  CNF inside chart; top-down sampling with all spans of one length in parallel (structural)
    TorchCopy   pairs (i, j) with y_i = y_j; closed form               (relational)
    det_logZ    directed matrix-tree determinant                       (arborescences; logZ only)

accepts(y) is isfinite(logZ(onehot(y))): the same route on a fully revealed input. It is cross-checked
against independent CPU checkers in tests/test_torch_routes.py. Sampling uses the Gumbel-max trick
(Gumbel 1954; Maddison, Tarlow & Minka, NeurIPS 2014): argmax(logits + Gumbel noise) is an exact draw.
"""
import torch
from .inside import CNF, inside_log

NEG = float("-inf")


# Uniforms for Gumbel noise. float32 cannot represent Gumbel values above ~16.6, so classes with relative
# probability below ~1e-8 are never drawn; float64 extends this to ~1e-16. Tested equivalent on synthetic
# denoisers (experiments/controls/test_precision.py); kept switchable.
GUMBEL_DTYPE = torch.float32


def _gumbel_argmax(scores, gen=None):
    """Exact categorical draw along the last axis from unnormalised log-probabilities."""
    dt = GUMBEL_DTYPE
    tiny = 1e-300 if dt == torch.float64 else 1e-20
    u = torch.rand(scores.shape, device=scores.device, dtype=dt, generator=gen).clamp_(tiny, 1.0)
    return torch.argmax(scores.to(dt) - torch.log(-torch.log(u)), dim=-1)


def onehot_log(y, A):
    """(..., ) integer symbols -> (..., A) log one-hot (0 at y, -inf elsewhere)."""
    lw = torch.full((*y.shape, A), NEG, device=y.device)
    lw.scatter_(-1, y.unsqueeze(-1), 0.0)
    return lw


class TorchChain:
    """Automaton route over a compile.WDFA. logZ: scaled forward pass, O(L |E|). sample: FFBS
    (Carter & Kohn 1994; Dang & Ermon arXiv:2607.07026)."""

    def __init__(self, wdfa, device="cpu"):
        e = torch.tensor(wdfa.edges, dtype=torch.long)
        self.src, self.sym, self.dst = (e[:, k].to(device) for k in range(3))
        self.Q, self.q0, self.A = wdfa.n, wdfa.q0, wdfa.A
        self.Fmask = torch.zeros(self.Q, dtype=torch.bool, device=device)
        self.Fmask[sorted(wdfa.F)] = True
        self.device, self.name = device, wdfa.name

    def _forward(self, lw):
        B, L, A = lw.shape
        mx = lw.max(-1, keepdim=True).values.clamp_min(-1e30)
        W = torch.exp(lw - mx)
        logscale = mx.squeeze(-1).double().sum(-1)
        a = torch.zeros(B, self.Q, device=lw.device); a[:, self.q0] = 1.0
        alphas = [a]
        for t in range(L):
            v = a[:, self.src] * W[:, t, self.sym]
            na = torch.zeros(B, self.Q, device=lw.device).index_add_(1, self.dst, v)
            s = na.sum(-1, keepdim=True)
            a = na / s.clamp_min(1e-38)
            logscale = logscale + torch.log(s.squeeze(-1).double())
            alphas.append(a)
        return torch.stack(alphas, 1), logscale

    def logZ(self, lw):
        al, ls = self._forward(lw)
        return ls + torch.log((al[:, -1] * self.Fmask).sum(-1).double())

    def sample(self, lw, gen=None):
        B, L, A = lw.shape
        al, _ = self._forward(lw)
        la = torch.log(al)
        q = _gumbel_argmax(torch.where(self.Fmask, la[:, L], torch.full_like(la[:, L], NEG)), gen)
        y = torch.empty(B, L, dtype=torch.long, device=lw.device)
        for t in range(L - 1, -1, -1):
            sc = la[:, t, self.src] + lw[:, t, self.sym]                   # (B, E)
            sc = torch.where(self.dst[None] == q[:, None], sc, torch.full_like(sc, NEG))
            e = _gumbel_argmax(sc, gen)
            y[:, t] = self.sym[e]; q = self.src[e]
        return y

    def accepts(self, y):
        return torch.isfinite(self.logZ(onehot_log(y, self.A)))


class TorchChart:
    """Grammar route over a compile.Inside grammar. logZ: batched inside (sum semiring, d-weighted).
    sample: top-down from the inside chart, processing all spans of one length in parallel; the CNF tree
    covers each span at most once, so one nonterminal per (row, span) suffices."""

    def __init__(self, ins, device="cpu"):
        self.g = CNF(ins.start, ins.binary, ins.lexical, ins.A).to(device)
        self.A, self.device, self.name = ins.A, device, getattr(ins, "name", "cfg")
        self.lexlog = torch.log(self.g.lex.float()).to(device)              # (T, N)

    def logZ(self, lw, return_chart=False):
        z, ch = inside_log(self.g, lw.float())
        return (z.double(), ch) if return_chart else z.double()

    def sample(self, lw, gen=None, chart=None):
        lw = lw.float()
        B, L, A = lw.shape
        dev, g = lw.device, self.g
        if chart is None:
            _, chart = inside_log(g, lw)
        nt = torch.full((B, L, L + 1), -1, dtype=torch.long, device=dev)
        nt[:, 0, L] = g.start
        bidx = torch.arange(B, device=dev)
        for span in range(L, 1, -1):
            n = L - span + 1
            i = torch.arange(n, device=dev)[:, None]
            m = torch.arange(1, span, device=dev)[None, :]
            left = chart[:, i, i + m, :]                                     # (B, n, span-1, N)
            right = chart[:, i + m, i + span, :]
            vals = left[..., g.rB] + right[..., g.rC]                        # (B, n, span-1, R)
            a = nt[:, torch.arange(n, device=dev), torch.arange(n, device=dev) + span]   # (B, n)
            ok = (g.rA[None, None, :] == a[..., None])                       # (B, n, R)
            vals = torch.where(ok[:, :, None, :], vals, torch.full_like(vals, NEG))
            flat = vals.reshape(B, n, -1)
            act = a >= 0
            if not act.any():
                continue
            pick = _gumbel_argmax(torch.where(act[..., None], flat, torch.zeros_like(flat)), gen)  # (B, n)
            mm = pick // g.rA.numel() + 1
            rr = pick % g.rA.numel()
            bb, ii = torch.nonzero(act, as_tuple=True)
            mi, ri = mm[bb, ii], rr[bb, ii]
            nt[bb, ii, ii + mi] = g.rB[ri]
            nt[bb, ii + mi, ii + span] = g.rC[ri]
        leaf = nt[:, torch.arange(L, device=dev), torch.arange(L, device=dev) + 1]     # (B, L)
        sc = lw + self.lexlog.T[leaf]                                         # (B, L, T)
        return _gumbel_argmax(sc, gen)

    def accepts(self, y):
        return torch.isfinite(self.logZ(onehot_log(y, self.A)))


class TorchCopy:
    """Relational route: y_i = y_j for each (i, j) in pairs; other positions free. Closed form."""

    def __init__(self, pairs, L, A, device="cpu", name="copy"):
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)
        paired = set(i for p in pairs for i in p)
        self.free = torch.tensor([i for i in range(L) if i not in paired], dtype=torch.long, device=device)
        self.L, self.A, self.name = L, A, name

    def logZ(self, lw):
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        z = torch.logsumexp(lw[:, i] + lw[:, j], -1).sum(-1)
        if self.free.numel():
            z = z + torch.logsumexp(lw[:, self.free], -1).sum(-1)
        return z.double()

    def sample(self, lw, gen=None):
        y = _gumbel_argmax(lw, gen)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        yp = _gumbel_argmax(lw[:, i] + lw[:, j], gen)
        y[:, i] = yp; y[:, j] = yp
        return y

    def accepts(self, y):
        return (y[:, self.pairs[:, 0]] == y[:, self.pairs[:, 1]]).all(-1)


def det_logZ(logW):
    """logW (B, n, n+1): logW[b, m-1, h] = log weight of head h for word m; h=m forbidden."""
    B, n, _ = logW.shape
    W = torch.exp(logW).transpose(1, 2).clone()                # (B, heads 0..n, words 1..n)
    idx = torch.arange(1, n + 1, device=logW.device)
    W[:, idx, idx - 1] = 0.0
    Lap = -W[:, 1:, :] + torch.diag_embed(W.sum(1))
    sign, logdet = torch.linalg.slogdet(Lap)
    return torch.where(sign > 0, logdet, torch.full_like(logdet, float("-inf")))


def copy_logZ(logW, k):
    """Closed-form log partition of the same-order copy y_i = y_{k+i} on (B, 2k, A) weights (E4 benchmark)."""
    return torch.logsumexp(logW[:, :k] + logW[:, k:2 * k], dim=-1).sum(-1)
