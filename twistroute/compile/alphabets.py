"""
Alphabet changes and per-position restrictions for routes (CPU reference). Nothing here is domain-specific.

lift_dfa(dfa, h)        inverse homomorphism. h[a] is the string of base symbols that new symbol a stands for;
                        the lifted automaton reads a and moves as the base automaton does on h[a]. It accepts
                        a1..an exactly when the base accepts h(a1)..h(an) (regular languages are closed under
                        inverse homomorphisms). States after a partial word never appear, and the result is
                        trimmed to states that are reachable and can still accept.
minimize_dfa(dfa)       Moore partition refinement of a trimmed automaton (same language, fewest states).
symbol_classes(dfa)     symbols that induce the same transition from every state are interchangeable for the
                        route; returns the partition and the quotient automaton over classes. `refine` keeps
                        apart symbols that must stay distinguishable for another reason (e.g. a per-position
                        restriction that depends on the symbol), so the quotient stays exact under it.
Masked(route, allowed)  conjunction with independent per-position allowed sets: weights outside allowed[i] are
                        set to -inf before the route's own pass. A restriction of this kind is a constraint
                        like any other; imposing it by renormalising the model's predictions step by step
                        would change the target instead of conditioning on it.
"""
from collections import deque
import numpy as np
from .automata import WDFA


def trim_dfa(dfa, name=None):
    """Keep states reachable from q0 that can still reach an accepting state; renumber with q0 = 0."""
    fwd = {dfa.q0}
    dq = deque([dfa.q0])
    out = {}
    for s, y, d in dfa.edges:
        out.setdefault(s, []).append(d)
    while dq:
        s = dq.popleft()
        for d in out.get(s, ()):
            if d not in fwd:
                fwd.add(d); dq.append(d)
    inc = {}
    for s, y, d in dfa.edges:
        inc.setdefault(d, []).append(s)
    co = set(q for q in dfa.F if q in fwd)
    dq = deque(co)
    while dq:
        d = dq.popleft()
        for s in inc.get(d, ()):
            if s not in co and s in fwd:
                co.add(s); dq.append(s)
    keep = sorted(co, key=lambda q: (q != dfa.q0, q))
    if dfa.q0 not in co:                                    # empty language: one non-accepting state
        return WDFA(1, [], 0, [], dfa.A, name or dfa.name)
    idx = {q: k for k, q in enumerate(keep)}
    edges = [(idx[s], y, idx[d]) for s, y, d in dfa.edges if s in idx and d in idx]
    return WDFA(len(keep), edges, 0, [idx[q] for q in dfa.F if q in idx], dfa.A, name or dfa.name)


def lift_dfa(dfa, h, name=None):
    """Automaton over len(h) new symbols; new symbol a moves like the base word h[a]."""
    edges = []
    for q in range(dfa.n):
        for a, word in enumerate(h):
            s = q
            for y in word:
                s = dfa.delta.get((s, y))
                if s is None:
                    break
            if s is not None:
                edges.append((q, a, s))
    return trim_dfa(WDFA(dfa.n, edges, dfa.q0, dfa.F, len(h), name or f"lift({dfa.name})"))


def minimize_dfa(dfa, name=None):
    """Moore's partition refinement on the trimmed automaton; missing transitions go to an implicit sink."""
    d = trim_dfa(dfa)
    n, A = d.n, d.A
    part = [int(q in d.F) for q in range(n)]
    while True:
        sig = [(part[q],) + tuple(part[d.delta[(q, y)]] if (q, y) in d.delta else -1 for y in range(A))
               for q in range(n)]
        ids = {}
        new = [ids.setdefault(s, len(ids)) for s in sig]
        if len(ids) == len(set(part)):
            break
        part = new
    bid = {}
    for p in [part[d.q0]] + part:                                           # q0's block gets id 0
        bid.setdefault(p, len(bid))
    edges = sorted({(bid[part[s]], y, bid[part[t]]) for s, y, t in d.edges})
    F = sorted({bid[part[q]] for q in d.F})
    return WDFA(len(bid), edges, bid[part[d.q0]], F, A, name or d.name)


def symbol_classes(dfa, refine=None):
    """(cls, quotient): cls[a] is the class of symbol a; the quotient automaton reads class ids. Two symbols
    share a class iff they have the same transition from every state (and the same refine label)."""
    sig, cls = {}, []
    for a in range(dfa.A):
        key = tuple(dfa.delta.get((q, a), -1) for q in range(dfa.n)) + ((refine[a],) if refine is not None else ())
        cls.append(sig.setdefault(key, len(sig)))
    rep = {}
    for a, c in enumerate(cls):
        rep.setdefault(c, a)
    edges = [(q, c, dfa.delta[(q, a)]) for c, a in sorted(rep.items()) for q in range(dfa.n) if (q, a) in dfa.delta]
    return cls, WDFA(dfa.n, edges, dfa.q0, dfa.F, len(rep), f"classes({dfa.name})")


class Masked:
    """route AND (y_i in allowed[i] for every i). allowed: (L, A) bool array."""

    def __init__(self, route, allowed, name=None):
        self.route, self.allowed = route, np.asarray(allowed, bool)
        self.L, self.A = self.allowed.shape
        self.name = name or f"masked({getattr(route, 'name', 'route')})"

    def _mask(self, logW):
        return np.where(self.allowed, np.asarray(logW, float), -np.inf)

    def logZ(self, logW):
        return self.route.logZ(self._mask(logW))

    def sample(self, logW, rng):
        return self.route.sample(self._mask(logW), rng)

    def accepts(self, seq):
        return len(seq) == self.L and all(self.allowed[i, y] for i, y in enumerate(seq)) and self.route.accepts(seq)
