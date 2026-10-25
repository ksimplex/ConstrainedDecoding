"""
Grammar-chart route (CPU reference): weighted inside algorithm over a CNF grammar.

    logZ(logW, "sum") = log sum over derivations d of prod_i exp(logW[i, y_i(d)])   O(L^3 * |R|)
    logZ(logW, "max") = log of the best derivation (constrained MAP; Viterbi semiring)
    sample(logW)      = exact draw of a derivation proportional to its weight, by top-down sampling
                        from the inside chart (Johnson, Griffiths & Goldwater, NAACL 2007, Sec. 3)

For an ambiguous grammar the sum counts each string y with multiplicity d(y) = number of parses, so the
route's law is the d-weighted one. Theorem 3 (theory notes) shows twisted SMC is still exact for the
native conditional p(x | x in C) provided the terminal twist is the indicator 1_C (TWISTER Def. 4.1).

count_parses is an independent recursive parse counter used only by tests.
"""
import numpy as np
from ._util import NEG, categorical


class Inside:
    def __init__(self, start, binary, lexical, A, name="cfg"):
        self.start, self.binary, self.lexical, self.A, self.name = start, binary, lexical, A, name
        nts = sorted({a for a, _, _ in binary} | {b for _, b, _ in binary} | {c for _, _, c in binary}
                     | {a for a, _ in lexical} | {start})
        self.idx = {a: i for i, a in enumerate(nts)}
        self.N = len(nts)
    def accepts(self, seq):
        n = len(seq)
        if n == 0: return False
        ch = [[set() for _ in range(n + 1)] for _ in range(n)]
        lex = {}
        for a, t in self.lexical:
            lex.setdefault(t, set()).add(a)
        for i, y in enumerate(seq):
            ch[i][i + 1] = set(lex.get(y, set()))
        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span
                for k in range(i + 1, j):
                    for a, b, cc in self.binary:
                        if b in ch[i][k] and cc in ch[k][j]:
                            ch[i][j].add(a)
        return self.start in ch[0][n]
    def _chart(self, logW, semiring):
        L = logW.shape[0]
        ch = np.full((L, L + 1, self.N), NEG)
        red = np.logaddexp if semiring == "sum" else np.maximum
        for a, t in self.lexical:
            ai = self.idx[a]
            for i in range(L):
                ch[i, i + 1, ai] = red(ch[i, i + 1, ai], logW[i, t])
        for span in range(2, L + 1):
            for i in range(L - span + 1):
                j = i + span
                for a, b, cc in self.binary:
                    ai, bi, ci = self.idx[a], self.idx[b], self.idx[cc]
                    for k in range(i + 1, j):
                        v = ch[i, k, bi] + ch[k, j, ci]
                        ch[i, j, ai] = red(ch[i, j, ai], v)
        return ch
    def logZ(self, logW, semiring="sum"):
        return float(self._chart(logW, semiring)[0, logW.shape[0], self.idx[self.start]])

    def sample(self, logW, rng):
        """Exact draw y ~ prod_i exp(logW[i, y_i]) * d(y) / Z by top-down sampling from the sum chart.

        At span (i, j) with nonterminal a, choose rule a -> b c and split k with probability
        proportional to chart[i, k, b] * chart[k, j, c]; at a leaf choose the terminal by its weight.
        Each derivation is drawn with probability weight(d) / Z, so strings are d-weighted.
        """
        L = logW.shape[0]
        ch = self._chart(logW, "sum")
        s0 = self.idx[self.start]
        if not np.isfinite(ch[0, L, s0]):
            raise ValueError("constraint unsatisfiable under logW")
        if not hasattr(self, "_bylhs"):
            self._bylhs, self._lexby = {}, {}
            for a, b, c in self.binary:
                self._bylhs.setdefault(self.idx[a], []).append((self.idx[b], self.idx[c]))
            for a, t in self.lexical:
                self._lexby.setdefault(self.idx[a], []).append(t)
        out = [None] * L
        stack = [(0, L, s0)]
        while stack:
            i, j, a = stack.pop()
            if j - i == 1:
                ts = self._lexby.get(a, [])
                out[i] = int(ts[categorical(rng, [logW[i, t] for t in ts])])
                continue
            opts, sc = [], []
            for b, c in self._bylhs.get(a, []):
                for k in range(i + 1, j):
                    v = ch[i, k, b] + ch[k, j, c]
                    if np.isfinite(v):
                        opts.append((k, b, c)); sc.append(v)
            k, b, c = opts[categorical(rng, sc)]
            stack.append((i, k, b)); stack.append((k, j, c))
        return tuple(out)


def count_parses(start, binary, lexical, seq):
    """Independent recursive parse counter (cross-checks the numpy chart's sum semiring)."""
    from functools import lru_cache
    lex = {}
    for a, t in lexical:
        lex.setdefault(t, set()).add(a)
    bylhs = {}
    for a, b, c in binary:
        bylhs.setdefault(a, []).append((b, c))
    @lru_cache(None)
    def cnt(sym, i, j):
        if j - i == 1:
            return 1 if sym in lex.get(seq[i], set()) else 0
        tot = 0
        for b, c in bylhs.get(sym, []):
            for k in range(i + 1, j):
                tot += cnt(b, i, k) * cnt(c, k, j)
        return tot
    return cnt(start, 0, len(seq))


def random_cnf(nN, nBin, nLex, A, seed=0):
    g = np.random.default_rng(seed)
    nts = [f"N{i}" for i in range(nN)]
    binary = sorted({(nts[g.integers(nN)], nts[g.integers(nN)], nts[g.integers(nN)]) for _ in range(nBin)})
    lexical = sorted({(nts[g.integers(nN)], int(g.integers(A))) for _ in range(nLex)})
    return nts[0], binary, lexical


DYCK2 = ("S",
         [("S", "LP", "P1"), ("P1", "S", "P2"), ("P2", "RP", "S"), ("S", "LP", "P3"), ("P3", "RP", "S"),
          ("S", "LP", "P4"), ("P4", "S", "RP"), ("S", "LP", "RP"),
          ("S", "LS", "Q1"), ("Q1", "S", "Q2"), ("Q2", "RS", "S"), ("S", "LS", "Q3"), ("Q3", "RS", "S"),
          ("S", "LS", "Q4"), ("Q4", "S", "RS"), ("S", "LS", "RS")],
         [("LP", 0), ("RP", 1), ("LS", 2), ("RS", 3)], 4)


# Dyck-1 over {0='(', 1=')'}; the same language under an ambiguous and an unambiguous grammar.
DYCK1_AMBIGUOUS = ("S", [("S", "S", "S"), ("S", "A", "B"), ("S", "A", "C"), ("C", "S", "B")],
                   [("A", 0), ("B", 1)], 2)
DYCK1_UNAMBIGUOUS = ("S", [("S", "A", "P1"), ("P1", "S", "P2"), ("P2", "B", "S"), ("S", "A", "P3"), ("P3", "B", "S"),
                           ("S", "A", "P4"), ("P4", "S", "B"), ("S", "A", "B")],
                     [("A", 0), ("B", 1)], 2)

# RNA "some nested pairing exists" grammar over {A, C, G, U} = {0, 1, 2, 3}.
RNA_PAIRS = [(0, 3), (3, 0), (2, 1), (1, 2), (2, 3), (3, 2)]   # AU UA GC CG GU UG (Watson-Crick + wobble)


def rna_pair_grammar():
    """Every base paired, pairs non-crossing and complementary (Nussinov & Jacobson, PNAS 1980, style).

        S -> P S | <P-productions>       a structure is a nonempty sequence of top-level helices
        P -> T_x Q_xy | T_x T_y          pair (x, y) enclosing a nonempty / empty interior
        Q_xy -> S T_y
    Derivations are in bijection with secondary structures, so d(y) = number of structures compatible
    with y, which can grow exponentially in L (e.g. (GC)^n admits Catalan(n) structures).
    """
    nuc = "ACGU"
    T = {x: f"T{nuc[x]}" for x in range(4)}
    binary = [("S", "P", "S")]
    for x, y in RNA_PAIRS:
        q = f"Q{nuc[x]}{nuc[y]}"
        binary += [("P", T[x], q), ("P", T[x], T[y]), ("S", T[x], q), ("S", T[x], T[y]), (q, "S", T[y])]
    return "S", binary, [(T[x], x) for x in range(4)], 4
