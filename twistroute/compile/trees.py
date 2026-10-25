"""
Spanning-tree route (CPU reference): head-pointer sequences constrained to form an arborescence.

logZ is the weighted directed matrix-tree theorem (Tutte 1948): log det of the reduced Laplacian.
marginals follow Koo, Globerson, Carreras & Collins (EMNLP 2007). This family has polynomial-size
arithmetic circuits but no small monotone circuit (Jerrum & Snir 1982), so automaton/chart/width-bounded
routes would refuse it; the determinant decodes it exactly.

sample() is Wilson's loop-erased random-walk algorithm (Wilson, STOC 1996) on the in-tree to root 0,
with a step cap and an exact sequential matrix-tree fallback for weight matrices with very long walks.
"""
import numpy as np
from ._util import NEG, categorical


class Arborescence:
    """Parent-pointer trees rooted at 0. L variables = words 1..L; symbol h in 0..L is the head of word m
    (h=0 means root, h=m forbidden). Constraint = heads form a spanning arborescence rooted at 0.
    Tractable by the weighted directed matrix-tree theorem (Tutte) — a DETERMINANT, not a circuit route.
    This family has small arithmetic circuits but every monotone/sd-DNNF circuit is exp-size (Jerrum-Snir),
    so the DFA/chart/bounded-width compiler would REFUSE it; the determinant decodes it exactly."""
    def __init__(self, n, name="arborescence"):
        self.n, self.L, self.A, self.name = n, n, n + 1, name
    def accepts(self, seq):
        if len(seq) != self.n:
            return False
        for m in range(1, self.n + 1):
            h = seq[m - 1]
            if h == m or h < 0 or h > self.n:
                return False
        for m in range(1, self.n + 1):          # each word reaches root 0 without cycling
            v, seen = m, set()
            while v != 0:
                if v in seen:
                    return False
                seen.add(v); v = seq[v - 1]
        return True
    def _laplacian(self, W):                     # W[h, m] = weight of head h for word m, h,m in 0..n (m>=1)
        n = self.n
        Lap = -W[1:, 1:].copy()
        np.fill_diagonal(Lap, 0.0)
        Lap[np.diag_indices(n)] = W[:, 1:].sum(0) - np.diag(W[1:, 1:])
        return Lap
    def logZ(self, logW):
        # logW is (L, A) = (word index 0..n-1, head 0..n); build W[head, word 1..n]
        W = np.zeros((self.n + 1, self.n + 1))
        W[:, 1:] = np.exp(logW).T
        for m in range(1, self.n + 1):
            W[m, m] = 0.0
        sign, logdet = np.linalg.slogdet(self._laplacian(W))
        return logdet if sign > 0 else NEG
    def marginals(self, logW):
        n = self.n
        W = np.zeros((n + 1, n + 1)); W[:, 1:] = np.exp(logW).T
        for m in range(1, n + 1):
            W[m, m] = 0.0
        Linv = np.linalg.inv(self._laplacian(W))
        M = np.zeros((self.L, self.A))
        for m in range(1, n + 1):
            for h in range(n + 1):
                if h == m:
                    continue
                d = Linv[m - 1, m - 1] - (Linv[m - 1, h - 1] if h > 0 else 0.0)
                M[m - 1, h] = W[h, m] * d
        return M


    def sample(self, logW, rng, max_steps=200_000):
        """Wilson's algorithm. Word m steps to head h with probability proportional to W[h, m] (h != m), so a
        completed tree T has probability proportional to prod_m W[p(m), m], the matrix-tree law. Falls
        back to exact sequential sampling (n matrix-tree marginal evaluations) after max_steps."""
        n = self.n
        W = np.exp(np.asarray(logW, float))
        for m in range(1, n + 1):
            W[m - 1, m] = 0.0
        cum = np.cumsum(W / W.sum(1, keepdims=True), 1)
        in_tree = np.zeros(n + 1, bool); in_tree[0] = True
        nxt = np.full(n + 1, -1)
        steps = 0
        for start in rng.permutation(np.arange(1, n + 1)):
            u = start
            while not in_tree[u]:                       # random walk; overwriting nxt erases loops
                h = int(np.searchsorted(cum[u - 1], rng.random() * cum[u - 1, -1], side="right"))
                nxt[u] = min(h, n); u = nxt[u]; steps += 1
                if steps > max_steps:
                    return self._sample_sequential(logW, rng)
            u = start
            while not in_tree[u]:
                in_tree[u] = True; u = nxt[u]
        return tuple(int(nxt[m]) for m in range(1, n + 1))

    def _sample_sequential(self, logW, rng):
        lw = np.array(logW, float, copy=True)
        out = []
        for m in range(self.n):
            h = categorical(rng, np.log(np.clip(self.marginals(lw)[m], 0, None)))
            row = np.full(lw.shape[1], NEG); row[h] = lw[m, h]; lw[m] = row
            out.append(h)
        return tuple(out)
