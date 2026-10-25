"""
Factor-graph routes (CPU reference).

CopyFG       same-order copy y_i = x_i over k positions: closed form, O(k * A). The equivalent automaton
             needs (k+1) * A^k states (measured in E4).
FactorGraph  generic hard pairwise/higher-order constraints; logZ by bucket (variable) elimination
             (Dechter, Artificial Intelligence 113:41, 1999), cost exponential only in the induced width
             of the elimination order. sample() runs the same elimination, keeps every bucket table, and
             samples variables in reverse elimination order (each bucket table is the exact conditional).
"""
import numpy as np
from scipy.special import logsumexp
from ._util import NEG, lse as _lse, categorical


class CopyFG:
    """y_i = x_i. logW layout: positions 0..k-1 = x, k..2k-1 = y. Z = prod_i sum_s w_x[i,s] w_y[i,s]."""
    def __init__(self, k, A):
        self.k, self.A, self.L = k, A, 2 * k
        self.n_states = 2 * k                                # factor count, not automaton states
    def accepts(self, seq):
        return len(seq) == self.L and all(seq[i] == seq[self.k + i] for i in range(self.k))
    def logZ(self, logW):
        return float(sum(_lse(logW[i] + logW[self.k + i]) for i in range(self.k)))
    def marginals(self, logW):
        M = np.zeros((self.L, self.A))
        for i in range(self.k):
            p = np.exp(logW[i] + logW[self.k + i] - _lse(logW[i] + logW[self.k + i]))
            M[i] = p; M[self.k + i] = p
        return M

    def sample(self, logW, rng):
        """Exact draw: each pair (i, k+i) jointly from logW[i] + logW[k+i]."""
        out = [0] * self.L
        for i in range(self.k):
            s = categorical(rng, logW[i] + logW[self.k + i])
            out[i] = out[self.k + i] = s
        return tuple(out)


def ve_logZ(L, A, unary_logW, factors, order):
    """Exact log-partition by bucket elimination. factors: list of (scope_tuple, logtable ndarray).
    Returns (logZ, max_bucket_width). Width is the governing cost parameter (FactorDLM)."""
    active = [((i,), np.asarray(unary_logW[i], float)) for i in range(L)]
    active += [(tuple(s), np.asarray(t, float)) for s, t in factors]
    width = 0
    def expand(full, s, t):
        order_local = list(np.argsort(s)); t2 = np.transpose(t, order_local); ss = sorted(s)
        fidx = {x: k for k, x in enumerate(full)}; shape = [1] * len(full)
        for x in ss: shape[fidx[x]] = A
        return t2.reshape(shape)
    for v in order:
        rel = [f for f in active if v in f[0]]
        rest = [f for f in active if v not in f[0]]
        if not rel:
            active = rest; continue
        scope = sorted(set().union(*[set(s) for s, _ in rel]))
        width = max(width, len(scope))
        acc = sum(expand(scope, s, t) for s, t in rel)
        ax = scope.index(v)
        from scipy.special import logsumexp
        acc = logsumexp(acc, axis=ax)
        new_scope = tuple(x for x in scope if x != v)
        active = rest + [(new_scope, acc)]
    total = 0.0
    for s, t in active:
        total += float(t) if t.ndim == 0 else float(np.sum(t))   # all scalars after full elimination
    return total, width


class FactorGraph:
    """Generic hard-constraint factor graph. edges: list of (i, j) with pairwise table (A,A) in {0,1}."""
    def __init__(self, L, A, factors, order=None, name="fg"):
        self.L, self.A, self.name = L, A, name
        with np.errstate(divide="ignore"):                        # hard constraints: log 0 = -inf
            self.factors = [(tuple(s), np.log(np.asarray(t, float))) for s, t in factors]
        self.order = order if order is not None else list(range(L))
    def accepts(self, seq):
        for s, logt in self.factors:
            if not np.isfinite(logt[tuple(seq[x] for x in s)]):
                return False
        return len(seq) == self.L
    def logZ(self, logW):
        return ve_logZ(self.L, self.A, logW, self.factors, self.order)[0]
    def width(self, logW=None):
        lw = np.zeros((self.L, self.A)) if logW is None else logW
        return ve_logZ(self.L, self.A, lw, self.factors, self.order)[1]


def _fg_sample(self, logW, rng):
    """Exact draw by variable elimination + backward sampling.

    Eliminate in self.order, storing each bucket's joint log-table over its scope. Sampling the buckets in
    reverse order, every other variable in a bucket's scope has already been sampled, so the bucket table
    restricted to those values is the exact conditional of the bucket variable.
    """
    L, A = self.L, self.A
    active = [((i,), np.asarray(logW[i], float)) for i in range(L)]
    active += [(tuple(s), np.asarray(t, float)) for s, t in self.factors]

    def expand(full, s, t):
        perm = list(np.argsort(s)); t2 = np.transpose(t, perm); ss = sorted(s)
        fidx = {x: k for k, x in enumerate(full)}; shape = [1] * len(full)
        for x in ss:
            shape[fidx[x]] = A
        return np.broadcast_to(t2.reshape(shape), [A] * len(full))

    buckets = []
    for v in self.order:
        rel = [f for f in active if v in f[0]]
        rest = [f for f in active if v not in f[0]]
        if not rel:
            active = rest
            continue
        scope = sorted(set().union(*[set(s) for s, _ in rel]))
        acc = sum(expand(scope, s, t) for s, t in rel)
        buckets.append((v, scope, acc))
        active = rest + [(tuple(x for x in scope if x != v), logsumexp(acc, axis=scope.index(v)))]
    y = {}
    for v, scope, acc in reversed(buckets):
        y[v] = categorical(rng, acc[tuple(slice(None) if x == v else y[x] for x in scope)])
    for i in range(L):
        if i not in y:
            y[i] = categorical(rng, logW[i])
    return tuple(int(y[i]) for i in range(L))


FactorGraph.sample = _fg_sample
