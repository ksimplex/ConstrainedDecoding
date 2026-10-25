"""
Automaton route (CPU reference): weighted DFAs over a small symbol alphabet.

A constraint C is compiled to a deterministic finite automaton; for factorized log-weights logW[L, A]
(per-position log-categoricals, revealed positions one-hot) the route provides
    logZ(logW)   = log sum_{y in C} prod_i exp(logW[i, y_i])      forward algorithm, O(L * |E|)
    sample(logW) = exact draw from that constrained product law       forward-filter backward-sample
    marginals    = per-position constrained marginals               forward-backward
Because the automaton is deterministic, every accepted string has exactly one accepting path, so the
sum is over strings (no derivation weighting). This is the route used by TWISTER (Thimmaiah et al.,
arXiv:2609.35609, Prop. 3.4) and the step-exact sampler of Dang & Ermon (arXiv:2607.07026); FFBS is
Carter & Kohn (1994, Biometrika 81:541). Forbidden/required motif automata use Aho & Corasick (1975,
CACM 18:333).

Builders return a WDFA; build_dfa enumerates reachable states by BFS and raises CompilerRefuse past a
state budget (the tractability boundary measured in E4).
"""
from collections import deque
import numpy as np
from ._util import NEG, lse as _lse, CompilerRefuse


class WDFA:
    def __init__(self, n, edges, q0, F, A, name=""):
        self.n, self.q0, self.F, self.A, self.name = n, q0, set(F), A, name
        self.edges = edges                                  # list of (src, sym, dst)
        self.delta = {(s, y): d for s, y, d in edges}
    @property
    def n_states(self):
        return self.n
    def accepts(self, seq):
        q = self.q0
        for y in seq:
            q = self.delta.get((q, y))
            if q is None:
                return False
        return q in self.F
    def _forward(self, logW):
        L = logW.shape[0]
        alpha = np.full((L + 1, self.n), NEG)
        alpha[0, self.q0] = 0.0
        for t in range(L):
            acc = [[] for _ in range(self.n)]
            at = alpha[t]
            for s, y, d in self.edges:
                if at[s] > NEG:
                    acc[d].append(at[s] + logW[t, y])
            alpha[t + 1] = [_lse(a) if a else NEG for a in acc]
        return alpha
    def logZ(self, logW):
        a = self._forward(logW)
        return _lse([a[-1, q] for q in self.F]) if self.F else NEG
    def sample(self, logW, rng):
        """Exact draw from the constrained law by forward filtering, backward sampling (FFBS)."""
        L = logW.shape[0]
        alpha = self._forward(logW)
        fin = np.array([alpha[L, q] if q in self.F else NEG for q in range(self.n)])
        p = np.exp(fin - _lse(fin))
        q = int(rng.choice(self.n, p=p / p.sum()))
        if not hasattr(self, "_incoming"):
            self._incoming = [[] for _ in range(self.n)]
            for s, y, d in self.edges:
                self._incoming[d].append((s, y))
        out = [0] * L
        for t in range(L - 1, -1, -1):
            cand = self._incoming[q]
            sc = np.array([alpha[t, s] + logW[t, y] for s, y in cand])
            pp = np.exp(sc - _lse(sc))
            k = int(rng.choice(len(cand), p=pp / pp.sum()))
            q, out[t] = cand[k][0], cand[k][1]
        return tuple(int(v) for v in out)
    def marginals(self, logW):
        L = logW.shape[0]
        alpha = self._forward(logW)
        beta = np.full((L + 1, self.n), NEG)
        for q in self.F:
            beta[L, q] = 0.0
        for t in range(L - 1, -1, -1):
            for s, y, d in self.edges:
                beta[t, s] = np.logaddexp(beta[t, s], logW[t, y] + beta[t + 1, d])
        Z = _lse([alpha[L, q] for q in self.F]) if self.F else NEG
        M = np.full((L, self.A), NEG)
        for t in range(L):
            for s, y, d in self.edges:
                M[t, y] = np.logaddexp(M[t, y], alpha[t, s] + logW[t, y] + beta[t + 1, d])
        return np.exp(M - Z)


def build_dfa(start, step, is_accept, A, max_states=200000, name=""):
    """BFS the reachable state graph. step(state, sym) -> state' or None (reject). Omitted transitions = dead."""
    idx = {start: 0}
    order = [start]
    edges = []
    dq = deque([start])
    while dq:
        st = dq.popleft()
        si = idx[st]
        for y in range(A):
            ns = step(st, y)
            if ns is None:
                continue
            if ns not in idx:
                if len(idx) >= max_states:
                    raise CompilerRefuse(f"{name}: >{max_states} states")
                idx[ns] = len(order)
                order.append(ns)
                dq.append(ns)
            edges.append((si, y, idx[ns]))
    F = [idx[s] for s in order if is_accept(s)]
    return WDFA(len(order), edges, 0, F, A, name)


def dfa_forbidden(patterns, A, name="forbidden"):
    """Accept strings containing none of the given symbol-tuple patterns (Aho-Corasick)."""
    goto = [{}]
    out = [False]
    for pat in patterns:
        node = 0
        for y in pat:
            if y not in goto[node]:
                goto.append({}); out.append(False)
                goto[node][y] = len(goto) - 1
            node = goto[node][y]
        out[node] = True
    fail = [0] * len(goto)
    dq = deque()
    for y, nxt in goto[0].items():
        dq.append(nxt)
    while dq:
        u = dq.popleft()
        for y, v in goto[u].items():
            dq.append(v)
            f = fail[u]
            while f and y not in goto[f]:
                f = fail[f]
            fail[v] = goto[f].get(y, 0) if f != v else 0
            out[v] = out[v] or out[fail[v]]
    def delta_ac(node, y):
        while node and y not in goto[node]:
            node = fail[node]
        return goto[node].get(y, 0)
    def step(node, y):
        nn = delta_ac(node, y)
        return None if out[nn] else nn
    return build_dfa(0, step, lambda s: True, A, name=name)


def dfa_contains(patterns, A, name="contains"):
    """Accept strings containing at least one of the patterns (mandatory motif). State = (AC node, seen)."""
    goto = [{}]; out = [False]
    for pat in patterns:
        node = 0
        for y in pat:
            if y not in goto[node]:
                goto.append({}); out.append(False); goto[node][y] = len(goto) - 1
            node = goto[node][y]
        out[node] = True
    fail = [0] * len(goto); dq = deque(goto[0].values())
    while dq:
        u = dq.popleft()
        for y, v in goto[u].items():
            dq.append(v); f = fail[u]
            while f and y not in goto[f]:
                f = fail[f]
            fail[v] = goto[f].get(y, 0) if f != v else 0
            out[v] = out[v] or out[fail[v]]
    def dd(node, y):
        while node and y not in goto[node]:
            node = fail[node]
        return goto[node].get(y, 0)
    def step(st, y):
        node, seen = st
        nn = dd(node, y)
        return (nn, seen or out[nn])
    return build_dfa((0, False), step, lambda s: s[1], A, name=name)


def dfa_exactly_k(cls, K, A, name="exactly_k"):
    step = lambda c, y: (None if c + (y == cls) > K else c + (y == cls))
    return build_dfa(0, step, lambda c: c == K, A, name=name)


def dfa_fraction_band(cls, lo, hi, A, name="band"):
    # accept if count of `cls` in [lo, hi]; state = count, bounded by hi
    step = lambda c, y: (None if c + (y == cls) > hi else c + (y == cls))
    return build_dfa(0, step, lambda c: lo <= c <= hi, A, name=name)


def dfa_regex_a_then_no_bb(A, name="regex"):
    """must start with symbol 0, then never two 1's in a row (a (0|1)* with no '11')."""
    # state: ('start',) | ('ok', last_was_1)
    def step(st, y):
        if st == "S":
            return None if y != 0 else ("ok", False)
        _, last1 = st
        if y == 1 and last1:
            return None
        return ("ok", y == 1)
    return build_dfa("S", step, lambda s: s != "S", A, name=name)


def dfa_copy_positional(k, A, name="copy_dfa"):
    """y_i = x_i for i in [k]; positions 0..k-1 then k..2k-1. State remembers read symbols -> A^k states."""
    def step(st, y):
        if st[0] == "r":
            nt = st[1] + (y,)
            return ("r", nt) if len(nt) < k else ("c", nt, 0)
        if st[0] == "done":
            return None
        _, tup, j = st
        if y != tup[j]:
            return None
        return ("done",) if j + 1 == k else ("c", tup, j + 1)
    start = ("r", ())
    cap = (A ** (k + 1) - 1) // (A - 1) + A ** k * k + 10 if A > 1 else 10 ** 6
    return build_dfa(start, step, lambda s: s == ("done",), A, max_states=cap, name=name)


def product_dfa(d1, d2, A, name="product", max_states=200000):
    def step(st, y):
        a = d1.delta.get((st[0], y)); b = d2.delta.get((st[1], y))
        return None if a is None or b is None else (a, b)
    acc = lambda s: s[0] in d1.F and s[1] in d2.F
    return build_dfa((d1.q0, d2.q0), step, acc, A, max_states=max_states, name=name)


def dfa_window_alldiff(w, A, name="window_alldiff"):
    """All-different within every window of w+1 positions (no symbol repeats within distance w).
    Treewidth w; path decomposition = chain over the last w symbols."""
    def step(st, y):
        return None if y in st else (st + (y,))[-w:]
    return build_dfa((), step, lambda s: True, A, name=name)


def dfa_dyck2_depth(D, max_states=300_000):
    """Dyck-2 with stack depth <= D. Symbols 0='(' 1=')' 2='[' 3=']' (compiler DYCK2 convention).
    Exact for length-L strings when D >= L/2."""
    OPEN = {0: 0, 2: 1}; CLOSE = {1: 0, 3: 1}
    def step(st, y):
        if y in OPEN:
            return None if len(st) >= D else st + (OPEN[y],)
        if not st or st[-1] != CLOSE[y]:
            return None
        return st[:-1]
    return build_dfa((), step, lambda s: s == (), 4, max_states=max_states, name=f"dyck2_D{D}")


def dfa_prefix_class(cls, A, name="prefix"):
    """TWISTER's Figure 1 constraint "a.*": the first symbol is `cls`, the rest is unconstrained."""
    return build_dfa("S", lambda st, y: (None if y != cls else "ok") if st == "S" else "ok",
                     lambda st: st == "ok", A, name=name)
