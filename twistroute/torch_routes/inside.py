"""
Batched, log-space inside algorithm for CNF grammars with per-position (mean-field) leaf weights.

    logZ, chart = inside_log(g, logW)        # logW: (B, L, T) log leaf weights per terminal

Revealed positions are one-hot in weight space (0 / -inf in log space); masked positions carry the
denoiser's class log-probabilities. All spans of one length are computed in parallel, so the loop is over
span length only (depth O(L), work O(L^3 |R|)).

semiring="sum": log partition (sum over derivations, i.e. the d-weighted sum)
semiring="max": log of the best derivation (constrained MAP)
"""
import torch


NEG_INF = float("-inf")


class CNF:
    def __init__(self, start, binary, lexical, n_terminals):
        names = sorted({a for a, _, _ in binary} | {b for _, b, _ in binary} | {c for _, _, c in binary}
                       | {a for a, _ in lexical} | {start})
        self.names = names
        self.idx = {a: k for k, a in enumerate(names)}
        self.N, self.T = len(names), n_terminals
        self.start = self.idx[start]
        self.rA = torch.tensor([self.idx[a] for a, _, _ in binary], dtype=torch.long)
        self.rB = torch.tensor([self.idx[b] for _, b, _ in binary], dtype=torch.long)
        self.rC = torch.tensor([self.idx[c] for _, _, c in binary], dtype=torch.long)
        lex = torch.zeros(n_terminals, self.N)
        for a, t in lexical:
            lex[t, self.idx[a]] = 1.0
        self.lex = lex
        self.groups = [torch.nonzero(self.rA == a).flatten() for a in range(self.N)]

    def to(self, device):
        for k in ("rA", "rB", "rC", "lex"):
            setattr(self, k, getattr(self, k).to(device))
        self.groups = [g.to(device) for g in self.groups]
        return self


def _reduce(x, dim, semiring):
    return torch.logsumexp(x, dim=dim) if semiring == "sum" else x.max(dim=dim).values


def inside_log(g, logW, semiring="sum"):
    B, L, T = logW.shape
    dev, dt = logW.device, logW.dtype
    lexlog = torch.log(g.lex.to(dtype=dt))                                  # (T, N): 0 or -inf
    leaf = _reduce(logW[..., :, None] + lexlog[None, None], 2, semiring)    # (B, L, N)
    chart = torch.full((B, L, L + 1, g.N), NEG_INF, device=dev, dtype=dt)
    ar = torch.arange(L, device=dev)
    chart[:, ar, ar + 1, :] = leaf
    for span in range(2, L + 1):
        n = L - span + 1
        i = torch.arange(n, device=dev)[:, None]                            # (n, 1)
        m = torch.arange(1, span, device=dev)[None, :]                      # (1, span-1)
        left = chart[:, i, i + m, :]                                        # (B, n, span-1, N)
        right = chart[:, i + m, i + span, :]
        vals = left[..., g.rB] + right[..., g.rC]                           # (B, n, span-1, R)
        red = _reduce(vals, 2, semiring)                                    # (B, n, R)
        out = torch.full((B, n, g.N), NEG_INF, device=dev, dtype=dt)
        for a, rows in enumerate(g.groups):
            if rows.numel():
                out[..., a] = _reduce(red[..., rows], -1, semiring)
        i0 = i[:, 0]
        chart[:, i0, i0 + span, :] = out
    return chart[:, 0, L, g.start], chart


def cnf_from_inside(ins):
    """Convert a CPU compile.Inside grammar into a CNF for the batched kernels (same rules)."""
    return CNF(ins.start, ins.binary, ins.lexical, ins.A)
