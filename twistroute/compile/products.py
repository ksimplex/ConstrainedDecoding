"""
Conjunctions of routes of different kinds (CPU reference). Each construction returns an ordinary route of
one of the existing kinds, so the exact sampler, logZ and accepts of that kind apply unchanged.

intersect_dfa_cfg(dfa, grammar)   automaton AND grammar -> grammar (Bar-Hillel, Perles & Shamir 1961).
    Nonterminals are triples (p, X, q): X derives the substring while the automaton moves from state p to
    state q. Rules are built only for productive triples and kept only if reachable from the new start, so
    the size is at most |R| Q^3 and usually far less. The automaton is deterministic, so each derivation of
    y in the grammar maps to exactly one derivation in the product: d(y) is unchanged on the automaton's
    language. Span limits carry over (a triple inherits its symbol's limit).

DFAFactorProduct(dfa, fg)         automaton AND factor graph -> factor graph over augmented variables
    z_i = (y_i, s_i), where s_i is the automaton state after reading y_0..y_i. The automaton becomes a chain
    of pairwise factors (z_{i-1}, z_i) plus unary start / accept factors; the graph's factors are lifted to
    act on the y-part. Domain size A * Q; elimination cost (A Q)^(width), with the width of the chain plus
    the graph's own edges. Refuses past a table budget instead of running out of memory.
"""
from collections import defaultdict, deque
import itertools
import numpy as np
from ._util import CompilerRefuse
from .grammars import Inside
from .factor_graphs import FactorGraph

START = "<start>"


def _tri(x, p, q):
    return f"{x}@{p}>{q}"


def intersect_dfa_cfg(dfa, g, name=None):
    """Grammar for L(g) ∩ L(dfa) with derivation counts of g. g: compile.Inside; dfa: compile.WDFA."""
    if dfa.A != g.A:
        raise ValueError("automaton and grammar use different alphabets")
    by_sym = defaultdict(list)                                 # terminal -> [(p, q)]
    for p, y, q in dfa.edges:
        by_sym[y].append((p, q))
    lexical = sorted({(_tri(x, p, q), t) for x, t in g.lexical for p, q in by_sym[t]})
    prod = defaultdict(set)                                    # (symbol, p) -> {q} for productive triples
    for x, t in g.lexical:
        for p, q in by_sym[t]:
            prod[(x, p)].add(q)
    binary, changed = set(), True
    while changed:                                             # productive triples, to a fixed point
        changed = False
        for a, b, c in g.binary:
            for (sb, p), rs in list(prod.items()):
                if sb != b:
                    continue
                for r in list(rs):
                    for q in list(prod.get((c, r), ())):
                        rule = (_tri(a, p, q), _tri(b, p, r), _tri(c, r, q))
                        if rule not in binary:
                            binary.add(rule)
                        if q not in prod[(a, p)]:
                            prod[(a, p)].add(q); changed = True
    q0, F = dfa.q0, dfa.F
    s0 = g.start
    roots = {_tri(s0, q0, f) for f in F}
    binary |= {(START, b, c) for a, b, c in binary if a in roots}
    lexical = sorted(set(lexical) | {(START, t) for x, t in lexical if x in roots})
    # keep what is reachable from the start
    by_lhs = defaultdict(list)
    for a, b, c in binary:
        by_lhs[a].append((b, c))
    seen, dq = {START}, deque([START])
    while dq:
        a = dq.popleft()
        for b, c in by_lhs[a]:
            for x in (b, c):
                if x not in seen:
                    seen.add(x); dq.append(x)
    binary = sorted(r for r in binary if r[0] in seen)
    lexical = [r for r in lexical if r[0] in seen]
    if not binary and not lexical:
        raise CompilerRefuse("intersection is empty")
    lim = {}
    base = getattr(g, "span_limits", {}) or {}
    for x in seen:
        sym = s0 if x == START else x.split("@", 1)[0]
        if sym in base:
            lim[x] = base[sym]
    return Inside(START, binary, lexical, g.A, name=name or f"{dfa.name}&{g.name}", span_limits=lim)


def induced_width(L, scopes, order):
    """Largest bucket scope met when eliminating variables in `order` (no tables built)."""
    active = [set(s) for s in scopes]
    width = 0
    for v in order:
        rel = [s for s in active if v in s]
        if not rel:
            continue
        scope = set().union(*rel)
        width = max(width, len(scope))
        active = [s for s in active if v not in s] + [scope - {v}]
    return width


class DFAFactorProduct:
    """Route for L(dfa) ∩ C(fg) over the alphabet of both, through an augmented factor graph."""

    def __init__(self, dfa, fg, order=None, max_cells=2 * 10 ** 7, name=None):
        if dfa.A != fg.A:
            raise ValueError("automaton and factor graph use different alphabets")
        L, A, Q = fg.L, fg.A, dfa.n
        self.L, self.A, self.Q, self.dfa, self.fg = L, A, Q, dfa, fg
        self.name = name or f"{dfa.name}&{fg.name}"
        D = A * Q
        ys, ss = np.arange(D) // Q, np.arange(D) % Q
        nxt = np.full((Q, A), -1)                                 # transition table, -1 = dead
        for p, y, q in dfa.edges:
            nxt[p, y] = q
        start = (nxt[dfa.q0, ys] == ss).astype(float)
        accept = np.isin(ss, sorted(dfa.F)).astype(float)
        trans = (nxt[ss[:, None], ys[None, :]] == ss[None, :]).astype(float)   # (z_{i-1}, z_i)
        factors = [((0,), start), ((L - 1,), accept)] + [((i - 1, i), trans) for i in range(1, L)]
        for scope, logt in fg.factors:
            t = np.exp(logt)
            factors.append((tuple(scope), t[np.ix_(*[ys] * len(scope))]))
        order = list(order) if order is not None else list(fg.order)
        w = induced_width(L, [(i,) for i in range(L)] + [s for s, _ in factors], order)
        if float(D) ** w > max_cells:
            raise CompilerRefuse(f"{self.name}: elimination tables of {D}^{w} cells exceed {max_cells}")
        self.width = w
        self._aug = FactorGraph(L, D, factors, order=order, name=self.name)
        self._ys = ys

    def _lift(self, logW):
        return np.asarray(logW, float)[:, self._ys]

    def logZ(self, logW):
        return self._aug.logZ(self._lift(logW))

    def sample(self, logW, rng):
        z = self._aug.sample(self._lift(logW), rng)
        return tuple(int(self._ys[v]) for v in z)

    def accepts(self, seq):
        return len(seq) == self.L and self.dfa.accepts(seq) and self.fg.accepts(seq)
