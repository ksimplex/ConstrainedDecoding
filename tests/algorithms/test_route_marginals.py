"""The schedule's route query on every route family: exact constrained marginals by differentiating logZ.

decode.policy.constrained_marginals is checked against enumeration on automata, automaton products, the closed-form
copy, the subset and masked wrappers, factor graphs, the lifted automaton-and-factor-graph product, and charts
(Dyck-1, Dyck-2, automaton-and-grammar), with nothing revealed and with two positions revealed. Then the policy is
run through a chart: Dyck-1 as a grammar and as an automaton, same model and kernel, both unbiased for the exact p(C).
"""
import itertools, math
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.decode import twisted_smc, CouplingPolicy, DistanceKernel, constrained_marginals
from twistroute.models import ConstrainedTruth, lazy_chain
from twistroute.torch_routes import build_route, torch_route, TorchChain, TorchSubset

FB = dict(kind="forbidden", A=2, N=8, patterns=[[0, 0, 0]])
EK = dict(kind="exactly_k", A=2, N=8, cls=0, count=4)
FG = dict(kind="factor_graph", A=3, N=6, factors=[dict(scope=[0, 3], table=np.eye(3).tolist()),
                                                 dict(scope=[1, 4], table=(1 - np.eye(3)).tolist())])
DYCK1 = dict(kind="cfg", A=2, N=8, grammar="dyck1_unamb")
ROUTES = {
    "automaton": (lambda: build_route(dict(kind="forbidden", A=3, N=7, patterns=[[0, 0], [1, 2]])), 7, 3),
    "automaton product": (lambda: build_route(dict(kind="and", A=2, N=8, parts=[FB, EK])), 8, 2),
    "copy": (lambda: build_route(dict(kind="copy", A=3, pairs=3, N=6)), 6, 3),
    "subset wrapper": (lambda: TorchSubset(TorchChain(C.dfa_at_least_k(0, 2, 3)), [1, 2, 4, 5], 7), 7, 3),
    "masked wrapper": (lambda: torch_route(C.Masked(C.dfa_exactly_k(0, 2, 3), np.array([[1, 1, 0]] * 3 + [[1, 1, 1]] * 3, dtype=bool))), 6, 3),
    "factor graph": (lambda: build_route(FG), 6, 3),
    "automaton x factor graph": (lambda: build_route(dict(kind="and", A=3, N=6, parts=[dict(kind="exactly_k", A=3, N=6, cls=0, count=2), FG])), 6, 3),
    "chart Dyck-1": (lambda: build_route(DYCK1), 8, 2),
    "chart Dyck-2": (lambda: build_route(dict(kind="cfg", A=4, N=6, grammar="dyck2")), 6, 4),
    "chart automaton x grammar": (lambda: build_route(dict(kind="and", A=2, N=8, parts=[FB, DYCK1])), 8, 2),
}


def enumerate_marginals(route, lw):
    N, A = lw.shape
    ys = torch.tensor(list(itertools.product(range(A), repeat=N)))
    w = torch.exp(lw[torch.arange(N)[None], ys].sum(1).double()) * route.accepts(ys)
    mu = torch.stack([torch.stack([w[ys[:, i] == a].sum() for a in range(A)]) for i in range(N)])
    return (mu / w.sum()).numpy(), ys[w > 0]


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_constrained_marginals_match_enumeration(name):
    make, N, A = ROUTES[name]
    route = make()
    torch.manual_seed(1)
    lw = torch.log_softmax(torch.randn(N, A), -1)
    ref, valid = enumerate_marginals(route, lw)
    assert len(valid) > 1
    mu = constrained_marginals(route, lw[None].clone())[0].numpy()
    assert not np.isnan(mu).any() and np.abs(mu - ref).max() < 1e-5
    y = valid[len(valid) // 2]                                           # reveal two positions of a valid sequence
    lwc = lw.clone()
    for i in (0, N - 2):
        lwc[i] = float("-inf"); lwc[i, y[i]] = 0.0
    ref2, _ = enumerate_marginals(route, lwc)
    mu2 = constrained_marginals(route, lwc[None].clone())[0].numpy()
    assert not np.isnan(mu2).any() and np.abs(mu2 - ref2).max() < 1e-5


def test_policy_through_a_chart_matches_the_automaton():
    N = 12
    den = lazy_chain(N, 2, 0.6, seed=3, concentration=2.0); cl = den.classes()
    dfa = C.dfa_dyck1(N); truth = ConstrainedTruth(den, cl, dfa)
    kern = DistanceKernel(unscreened=[0.0] + [1.0] * (N - 1), screened=[0.0] * N)      # the chain's true structure
    res = {}
    for tag, route in (("automaton", torch_route(dfa)), ("chart", build_route(dict(kind="cfg", A=2, N=N, grammar="dyck1_unamb")))):
        out = twisted_smc(den, cl, route, N, N, nrun=1500, K=2, gen=torch.Generator().manual_seed(5), ess_min=0.0,
                          schedule=CouplingPolicy(kern, 0.0, priority="lex", mixed=False))
        z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
        assert out["frac_valid_finite"] == 1.0 and out["steps"].mean() < N
        assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se
        res[tag] = (z.mean(), se, out["steps"].mean())
    assert abs(res["automaton"][2] - res["chart"][2]) < 0.5              # the same schedule: about the same number of steps
