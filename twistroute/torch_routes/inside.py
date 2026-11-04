"""
Batched, log-space inside algorithm for CNF grammars with per-position (mean-field) leaf weights.

    logZ, chart = inside_log(g, logW)        # logW: (B, L, T) log leaf weights per terminal

Revealed positions are one-hot in weight space (0 / -inf in log space); masked positions carry the
denoiser's class log-probabilities. All spans of one length are computed in parallel, so the loop is over
span length only (depth O(L), work O(L^3 |R|)).

semiring="sum": log partition (sum over derivations, i.e. the d-weighted sum)
semiring="max": log of the best derivation (constrained MAP)

Span limits (CNF(..., span_limits={name: W})): a subtree rooted at a listed nonterminal covers at most W
positions; its chart entries for longer spans stay -inf. At each span length only the rules whose left-hand
side may cover that span are evaluated, and only split points m with m <= W(B) and span - m <= W(C) for
some such rule A -> B C. When every nonterminal except a right-branching spine (S -> X S with X bounded)
is bounded by W, the work drops from O(L^3 |R|) to O(L^2 W |R|). The chart itself stays (B, L, L+1, N).
"""
import math
import torch


NEG_INF = float("-inf")


class CNF:
    def __init__(self, start, binary, lexical, n_terminals, span_limits=None):
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
        lim = dict(span_limits or {})
        unknown = set(lim) - set(names)
        if unknown:
            raise ValueError(f"span_limits for unknown nonterminals {sorted(unknown)}")
        self.limit = implied_span_limits(names, binary, lexical, lim)      # per nonterminal
        self.bounded = bool(lim)
        self._plans = {}

    def plan(self, span):
        """Rules usable at this span length and the split range they allow: (rule ids, groups, m_lo, m_hi),
        or None when no rule applies. Groups index into the returned rule ids, one entry per nonterminal."""
        if not self.bounded:
            return None if span < 2 else (None, self.groups, 1, span - 1)
        if span in self._plans:
            return self._plans[span]
        rA, rB, rC = self.rA.tolist(), self.rB.tolist(), self.rC.tolist()
        act = [k for k in range(len(rA)) if self.limit[rA[k]] >= span]
        res = None
        if act:
            lo = min(max(1, span - min(self.limit[rC[k]], span)) for k in act)
            hi = max(min(span - 1, self.limit[rB[k]]) for k in act)
            lo, hi = int(lo), int(hi)
            if lo <= hi:
                dev = self.rA.device
                ids = torch.tensor(act, dtype=torch.long, device=dev)
                pos = {k: n for n, k in enumerate(act)}
                groups = [torch.tensor([pos[k] for k in g.tolist() if k in pos], dtype=torch.long, device=dev)
                          for g in self.groups]
                res = (ids, groups, lo, hi)
        self._plans[span] = res
        return res

    def to(self, device):
        self._plans = {}
        for k in ("rA", "rB", "rC", "lex"):
            setattr(self, k, getattr(self, k).to(device))
        self.groups = [g.to(device) for g in self.groups]
        return self


def implied_span_limits(names, binary, lexical, declared):
    """Largest yield each nonterminal can have: min(declared limit, max over its rules A -> B C of
    U(B) + U(C)), with 1 for a nonterminal that only has lexical rules; inf where recursion is unbounded.
    Iterated downward from inf to the fixed point. Implied by the declared limits, so the language is
    unchanged; it only lets the chart skip split points that no derivation can use."""
    U = {a: float(declared.get(a, math.inf)) for a in names}
    has_lex = {a for a, _ in lexical}
    while True:
        changed = False
        for a in names:
            best = 1.0 if a in has_lex else 0.0
            for lhs, b, c in binary:
                if lhs == a:
                    best = max(best, U[b] + U[c])
            v = min(float(declared.get(a, math.inf)), best)
            if v < U[a]:
                U[a], changed = v, True
        if not changed:
            return [U[a] for a in names]


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
        pl = g.plan(span)
        if pl is None:
            continue                                                        # nothing can cover this span
        ids, groups, lo, hi = pl
        rB, rC = (g.rB, g.rC) if ids is None else (g.rB[ids], g.rC[ids])
        n = L - span + 1
        i = torch.arange(n, device=dev)[:, None]                            # (n, 1)
        m = torch.arange(lo, hi + 1, device=dev)[None, :]                   # (1, splits)
        left = chart[:, i, i + m, :]                                        # (B, n, splits, N)
        right = chart[:, i + m, i + span, :]
        vals = left[..., rB] + right[..., rC]                               # (B, n, splits, R')
        red = _reduce(vals, 2, semiring)                                    # (B, n, R')
        out = torch.full((B, n, g.N), NEG_INF, device=dev, dtype=dt)
        for a, rows in enumerate(groups):
            if rows.numel():
                out[..., a] = _reduce(red[..., rows], -1, semiring)
        i0 = i[:, 0]
        chart[:, i0, i0 + span, :] = out
    return chart[:, 0, L, g.start], chart


def cnf_from_inside(ins):
    """Convert a CPU compile.Inside grammar into a CNF for the batched kernels (same rules and span limits)."""
    return CNF(ins.start, ins.binary, ins.lexical, ins.A, getattr(ins, "span_limits", None))
