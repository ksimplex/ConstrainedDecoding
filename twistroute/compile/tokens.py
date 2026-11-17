"""
A regular constraint over characters, read through a tokenizer: token classes and the automaton over classes.

lift_pattern(pattern, strings, end_ids)   compiles a regular expression (interegular) to an automaton over characters,
        maps every vocabulary token to its transition function on the automaton's states (the state reached from each
        state after reading the token's characters, or dead), and groups tokens with equal transition functions into
        classes: the constraint cannot tell them apart, so the route and the twist work on classes and stay exact
        (see models/classes.py). Returns (token_class, wdfa, info).

        End of sequence: a generation region of fixed length is filled with `end_ids` tokens after the text ends.
        The automaton gets one extra state, entered by an end token from any accepting state and closed under end
        tokens; a sequence is accepted if it ends in an accepting state or in that state. All end tokens form one
        class. Tokens with an empty decoded string that are not end tokens (other special tokens) are dead.

        Only states reachable at token boundaries are kept. A token that is a fragment of a multi-byte character
        decodes to U+FFFD and is treated as that one character.

pair_ratio_table(samples, token_class, A, lam)   the dependence table of the chain law from pilot token sequences:
        for classes a, b at neighbouring positions, n_ab pairs were seen and e_ab = n_a. n_.b / n are expected under
        independence. With a Gamma(lam, lam) prior on the ratio r_ab (mean 1: independence, worth lam observations)
        and n_ab ~ Poisson(r_ab e_ab), the posterior mean is (n_ab + lam) / (e_ab + lam): unchanged where nothing was
        expected, well below 1 where many were expected and none seen, never zero. lam = None picks the value that
        maximises the marginal likelihood of the table (negative binomial), on a grid. The table is collapsed from
        token sequences, so one pilot corpus serves every constraint. Rows are normalised (the matched chain law
        does not depend on row or column scalings of the table).
"""
import math
import numpy as np
from .automata import WDFA


def token_codes(strings):
    L = max(1, max(len(s) for s in strings)); codes = np.full((len(strings), L), -1, dtype=np.int64)
    for i, s in enumerate(strings):
        codes[i, :len(s)] = [ord(ch) for ch in s]
    return codes, np.array([len(s) for s in strings])


def lift_pattern(pattern, strings, end_ids=(), codes=None):
    import interegular
    f = interegular.parse_pattern(pattern).to_fsm().reduce()
    states = sorted(f.states); idx = {s: k for k, s in enumerate(states)}; Q = len(states); C = len(f.alphabet.by_transition)
    delta = np.full((Q + 1, C + 1), Q, dtype=np.int32)                   # state Q is dead; symbol C = no such character class
    for s, row in f.map.items():
        for c, t in row.items():
            delta[idx[s], c] = idx[t]
    sym = f.alphabet; other = sym[interegular.fsm.anything_else] if interegular.fsm.anything_else in sym else C
    codes, lens = token_codes(strings) if codes is None else codes
    uniq = np.unique(codes[codes >= 0]); lut = np.array([sym[chr(u)] if chr(u) in sym else other for u in uniq], dtype=np.int64)
    V, L = codes.shape; T = np.tile(np.arange(Q, dtype=np.int32), (V, 1)); T[lens == 0] = Q
    for p in range(L):
        rows = np.nonzero(lens > p)[0]
        if len(rows) == 0:
            break
        T[rows] = delta[T[rows], lut[np.searchsorted(uniq, codes[rows, p])][:, None]]
    fin = np.zeros(Q + 2, dtype=bool); fin[[idx[s] for s in f.finals]] = True
    END = Q + 1                                                          # columns: Q char states, dead (Q), end (Q + 1)
    Tx = np.full((V, Q + 2), Q, dtype=np.int32); Tx[:, :Q] = T
    end_ids = np.asarray(sorted(set(int(e) for e in end_ids)), dtype=np.int64)
    if len(end_ids):
        Tx[end_ids] = Q; Tx[np.ix_(end_ids, np.nonzero(fin)[0])] = END; Tx[end_ids, END] = END; fin[END] = True
    q0 = idx[f.initial]; reach = {q0}; frontier = [q0]
    while frontier:
        q = frontier.pop(); nxt = np.unique(Tx[:, q]); nxt = nxt[nxt != Q]
        for t in nxt.tolist():
            if t not in reach:
                reach.add(t); frontier.append(t)
    R = sorted(reach, key=lambda q: (q != q0, q)); pos = np.full(Q + 2, -1, dtype=np.int64); pos[R] = np.arange(len(R))
    Tr = Tx[:, R]; u, inv = np.unique(Tr, axis=0, return_inverse=True); inv = inv.reshape(-1)
    edges = [(k, a, int(pos[t])) for a in range(len(u)) for k, t in enumerate(u[a].tolist()) if t != Q]
    wdfa = WDFA(len(R), edges, 0, [int(pos[q]) for q in R if fin[q]], len(u), name="lifted pattern")
    info = dict(char_states=Q, states=len(R), classes=int(len(u)), edges=len(edges), end_class=int(inv[end_ids[0]]) if len(end_ids) else None)
    return inv.astype(np.int64), wdfa, info


def _counts(samples, token_class, A):
    n = np.zeros((A, A))
    for x in samples:
        y = token_class[np.asarray(x)]
        np.add.at(n, (y[:-1], y[1:]), 1.0)
    return n


def pair_ratio_table(samples, token_class, A, lam=None, grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)):
    n = _counts(samples, token_class, A); tot = n.sum()
    e = np.outer(n.sum(1), n.sum(0)) / max(tot, 1.0)
    def loglik(l):                                                       # negative binomial marginal of n_ab given e_ab, up to terms free of l
        from math import lgamma
        m = e > 0; nn, ee = n[m], e[m]
        return float(sum(lgamma(k + l) - lgamma(l) + l * math.log(l / (l + x)) + k * math.log(x / (l + x)) for k, x in zip(nn.tolist(), ee.tolist())))
    chosen = lam if lam is not None else max(grid, key=loglik)
    r = (n + chosen) / (e + chosen)
    return r / r.sum(1, keepdims=True), dict(lam=float(chosen), pairs=float(tot), seen_entries=int((n > 0).sum()), loglik={float(l): loglik(l) for l in grid} if lam is None else None)
