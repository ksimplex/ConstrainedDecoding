"""Cells of the comparison: named oracle models and constraints, and the random stream of each (cell, arm).

A cell is (model, constraint, N). Tokens are classes: 3 symbols, 2 for Dyck-1, 4 for Dyck-2, 5 with atoms.
    models       chain_0.6, chain_0.85 (Markov: a revealed position separates its two sides exactly),
                 hidden_0.85 and hidden_0.85_n<noise> (hidden chain: dependence decays, nothing separates),
                 segments_0.85 (independent segments), repel_0.8 (anticorrelated neighbours),
                 chain_0.85_atoms, hidden_0.85_atoms (5 symbols, 60% atoms: for dyck2_atoms)
    constraints  forbidden, counting, copy, dyck (whole sequence); window, two_window, copy_spread, copy_block
                 (about a tenth of the positions, for long sequences); dyck_chart, dyck2, dyck2_atoms (grammars
                 through the chart route, exact answers from models.GrammarTruth)
"""
import math, zlib

MODELS = ("chain_0.6", "chain_0.85", "hidden_0.85", "segments_0.85")
CONSTRAINTS = ("forbidden", "counting", "copy", "dyck")
GRAMMAR_CONSTRAINTS = {"dyck_chart": "dyck1_unamb",      # Dyck-1 as a grammar: the control for "dyck" (an automaton)
                       "dyck2": "dyck2",                 # two bracket types, every position constrained
                       "dyck2_atoms": "dyck2_atoms"}     # two bracket types and a free atom
ATOM_PI = (0.1, 0.1, 0.1, 0.1, 0.6)
SYMBOLS = {"dyck": 2, "dyck_chart": 2, "dyck2": 4, "dyck2_atoms": 5}       # alphabet size per constraint; 3 otherwise


def n_symbols(constraint):
    return SYMBOLS.get(constraint, 3)


def stream_seed(base, *parts):
    """A generator seed for one (model, constraint, N, arm, purpose) stream. Stable across processes and
    platforms (crc32, not hash()), so no two cells or arms share random numbers and a run is reproducible."""
    return zlib.crc32("|".join(str(p) for p in (base, *parts)).encode()) & 0x7FFFFFFF


def make_model(name, N, S):
    from twistroute.models import lazy_chain, noisy_hmm, repel_chain
    if name == "repel_0.8":                               # neighbours anticorrelated, at 80% of the admissible strength
        pi = lazy_chain(2, S, 0.0, seed=1, concentration=2.0).pi.numpy()
        return repel_chain(N, S, 0.8 * pi.min() / (1 - pi.min()), seed=1, concentration=2.0)
    if name.endswith("_atoms"):                           # content-heavy law for the grammars with a free atom:
        if S != 5:                                        # 60% atoms, 10% each bracket (symbols as in compile.DYCK2_ATOMS)
            raise ValueError("the _atoms models have 5 symbols")
        if name == "chain_0.85_atoms":
            return lazy_chain(N, S, 0.85, pi=ATOM_PI)
        if name == "hidden_0.85_atoms":
            return noisy_hmm(N, S, 0.85, 0.3, pi=ATOM_PI)
        raise ValueError(name)
    if name == "chain_0.6":
        return lazy_chain(N, S, 0.6, seed=1, concentration=2.0)
    if name == "chain_0.85":
        return lazy_chain(N, S, 0.85, seed=1, concentration=2.0)
    if name == "hidden_0.85":
        return noisy_hmm(N, S, 0.85, 0.3, seed=1, concentration=2.0)
    if name.startswith("hidden_0.85_n"):                  # the hidden chain with another noise level: less noise, more
        return noisy_hmm(N, S, 0.85, float(name.split("_n")[1]), seed=1, concentration=2.0)     # screening by one token
    if name == "segments_0.85":
        return lazy_chain(N, S, 0.85, seed=1, concentration=2.0, breaks=tuple(range(7, N - 1, 8)))
    raise ValueError(name)


def make_constraint(name, den, N):
    from twistroute import compile as C
    from twistroute.models import ConstrainedTruth, CopyTruth
    from twistroute.torch_routes import torch_route, build_route
    from twistroute.torch_routes import TorchChain, TorchCopy, TorchSubset
    cl = den.classes()
    if name == "copy":
        return cl, build_route(dict(kind="copy", A=den.S, pairs=N // 2, N=N)), CopyTruth(den, cl)
    if name in ("copy_spread", "copy_block"):             # copy on about 10% of the positions: N / 20 of the N / 2 pairs
        n, m = N // 2, max(1, N // 20)
        idx = list(range(0, n, n // m))[:m] if name == "copy_spread" else list(range(n // 2 - m // 2, n // 2 - m // 2 + m))
        truth = CopyTruth(den, cl, pairs=idx)
        truth.info = dict(pairs=len(idx), first=idx[:3], constrained_fraction=2 * len(idx) / N)
        return cl, TorchCopy([(i, n + i) for i in idx], N, den.S), truth
    if name in ("window", "two_window"):                  # at least k of a tenth of the positions are class 0: one contiguous
        import numpy as np                                # window, or two distant windows sharing the count (every position
        w = max(8, N // 10)                               # of one is tied to every position of the other until both are done)
        act = np.zeros(N, dtype=bool)
        if name == "window":
            lo = N // 2 - w // 2; act[lo:lo + w] = True
        else:
            w = 2 * (w // 2); lo = N // 4 - w // 4
            act[lo:lo + w // 2] = True; act[lo + N // 2:lo + N // 2 + w // 2] = True
        a, b = 0, max(1, round(0.75 * w))                 # smallest k with p(C) <= 1e-3 (the constraint clearly binds),
                                                          # capped at 3/4 of the window so the window is never fully forced
        while b - a > 1:
            k = (a + b) // 2
            if ConstrainedTruth(den, cl, C.dfa_at_least_k(0, k, den.S), active=act).log_pC > math.log(1e-3):
                a = k
            else:
                b = k
        d = C.dfa_at_least_k(0, b, den.S)
        truth = ConstrainedTruth(den, cl, d, active=act)
        pos = [int(i) for i in np.nonzero(act)[0]]
        truth.info = dict(window=[pos[0], pos[-1] + 1], positions=len(pos), k=b, p_C=math.exp(truth.log_pC),
                          expected_count=float(w * (den.pi @ den.emit[0])[0]), constrained_fraction=w / N)
        return cl, TorchSubset(TorchChain(d), pos, N), truth
    if name in GRAMMAR_CONSTRAINTS:                       # context-free constraints through the chart route
        from twistroute.models import GrammarTruth
        gname = GRAMMAR_CONSTRAINTS[name]
        g = {"dyck1_unamb": C.DYCK1_UNAMBIGUOUS, "dyck2": C.DYCK2, "dyck2_atoms": C.DYCK2_ATOMS}[gname]
        truth = GrammarTruth(den, cl, g)
        truth.info = dict(grammar=gname, route="chart", p_C=math.exp(truth.log_pC))
        return cl, build_route(dict(kind="cfg", A=den.S, N=N, grammar=gname)), truth
    d = {"forbidden": lambda: C.dfa_forbidden([(0, 0), (1, 2)], 3), "counting": lambda: C.dfa_exactly_k(0, N // 3, 3),
         "dyck": lambda: C.dfa_dyck1(N)}[name]()
    return cl, torch_route(d), ConstrainedTruth(den, cl, d)
