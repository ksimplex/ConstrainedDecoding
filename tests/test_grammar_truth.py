"""Exact answers for a chain model under a grammar (models.oracle.GrammarTruth).

* Dyck-1 as a grammar against Dyck-1 as an automaton (ConstrainedTruth): same log p(C) and marginals;
* Dyck-2 against the bounded-depth Dyck-2 automaton at a length where the bound is not active;
* Dyck-2 with atoms against brute-force enumeration, and against a bounded-depth automaton at a longer length;
* the grammars used are unambiguous: every accepted string has exactly one parse;
* the sampler through the chart is unbiased for the exact p(C) of Dyck-2 with atoms.
"""
import itertools, math
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.compile.automata import build_dfa, dfa_dyck2_depth
from twistroute.decode import twisted_smc, CouplingPolicy, DistanceKernel
from twistroute.models import ConstrainedTruth, GrammarTruth, lazy_chain, noisy_hmm
from twistroute.torch_routes import build_route

MODELS = {"markov": lambda N, S: lazy_chain(N, S, 0.6, seed=3, concentration=2.0),
          "hidden": lambda N, S: noisy_hmm(N, S, 0.85, 0.3, seed=1, concentration=2.0)}


def dfa_dyck2_atoms(D):
    """Dyck-2 with atoms and stack depth <= D as an automaton (non-empty strings)."""
    OPEN, CLOSE = {0: 0, 2: 1}, {1: 0, 3: 1}
    def step(st, y):
        stack, seen = st
        if y == 4:
            return (stack, True)
        if y in OPEN:
            return None if len(stack) >= D else (stack + (OPEN[y],), True)
        return (stack[:-1], True) if stack and stack[-1] == CLOSE[y] else None
    return build_dfa(((), False), step, lambda s: s == ((), True), 5, name=f"dyck2_atoms_D{D}")


def close(a, b, tol=1e-9):
    return abs(a.log_pC - b.log_pC) < tol and np.abs(a.marginals - b.marginals).max() < tol


@pytest.mark.parametrize("model", sorted(MODELS))
def test_dyck1_grammar_equals_dyck1_automaton(model):
    for N in (8, 20):
        den = MODELS[model](N, 2); cl = den.classes()
        assert close(GrammarTruth(den, cl, C.DYCK1_UNAMBIGUOUS), ConstrainedTruth(den, cl, C.dfa_dyck1(N)))


@pytest.mark.parametrize("model", sorted(MODELS))
def test_dyck2_grammar_equals_bounded_depth_automaton(model):
    N = 12
    den = MODELS[model](N, 4); cl = den.classes()
    assert close(GrammarTruth(den, cl, C.DYCK2), ConstrainedTruth(den, cl, dfa_dyck2_depth(N // 2)))


@pytest.mark.parametrize("model", sorted(MODELS))
def test_dyck2_atoms_equals_automaton_and_enumeration(model):
    N = 14
    den = MODELS[model](N, 5); cl = den.classes()
    assert close(GrammarTruth(den, cl, C.DYCK2_ATOMS), ConstrainedTruth(den, cl, dfa_dyck2_atoms(N // 2)))
    N = 7
    den = MODELS[model](N, 5); cl = den.classes()
    ys = torch.tensor(list(itertools.product(range(5), repeat=N)))
    ok = build_route(dict(kind="cfg", A=5, N=N, grammar="dyck2_atoms")).accepts(ys)
    p = torch.exp(den.log_joint(ys)) * ok
    g = GrammarTruth(den, cl, C.DYCK2_ATOMS)
    assert abs(g.log_pC - math.log(float(p.sum()))) < 1e-9
    mu = np.stack([[float(p[ys[:, i] == a].sum() / p.sum()) for a in range(5)] for i in range(N)])
    assert np.abs(mu - g.marginals).max() < 1e-9


@pytest.mark.parametrize("name,N", [("DYCK2_ATOMS", 6), ("DYCK2", 6), ("DYCK1_UNAMBIGUOUS", 10)])
def test_grammars_are_unambiguous(name, N):
    start, binary, lexical, A = getattr(C, name)
    ins = C.Inside(start, binary, lexical, A)
    n = 0
    for y in itertools.product(range(A), repeat=N):
        if ins.accepts(y):
            n += 1
            assert C.count_parses(start, binary, lexical, y) == 1
    assert n > 1


def test_sampler_through_the_chart_is_unbiased_for_dyck2_atoms():
    N = 10
    den = MODELS["markov"](N, 5); cl = den.classes()
    truth = GrammarTruth(den, cl, C.DYCK2_ATOMS)
    route = build_route(dict(kind="cfg", A=5, N=N, grammar="dyck2_atoms"))
    kern = DistanceKernel(unscreened=[0.0] + [1.0] * (N - 1), screened=[0.0] * N)
    out = twisted_smc(den, cl, route, N, N, nrun=1500, K=2, gen=torch.Generator().manual_seed(2), ess_min=0.0,
                      schedule=CouplingPolicy(kern, 0.0, priority="lex", mixed=False), return_particles=True)
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    assert out["frac_valid_finite"] == 1.0 and abs(z.mean() - math.exp(truth.log_pC)) < 4 * se
