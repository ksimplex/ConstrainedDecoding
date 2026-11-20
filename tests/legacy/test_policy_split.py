"""Scoring and admission in CouplingPolicy are separate steps (decode/policy.py).

* the admission test `_fits`, rule by rule;
* each named static priority, rewritten as a function of the step context, reproduces the named one exactly over
  whole sampler runs: scoring reads the step context and nothing else;
* whatever the score, the admitted set obeys the admission rule: no coupled pair at threshold zero, the summed load
  within a budget, the set within its cap;
* a state-dependent score supplied as a function keeps Z_hat unbiased for the exact p(C) when only screened
  positions are co-revealed.
"""
import math
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.decode import twisted_smc, CouplingPolicy, DistanceKernel, fit_distance_kernel
from twistroute.decode.policy import RevealState, PRIORITIES, PULL_TOL, LEX_SCALE
from twistroute.models import ConstrainedTruth, lazy_chain, noisy_hmm
from twistroute.torch_routes import torch_route

D = lambda s: torch.where(s.masked, CouplingPolicy.distortion(s), torch.zeros_like(CouplingPolicy.distortion(s)))
SEP = lambda s: CouplingPolicy.neighbour_sum(s, s.masked.double())
AS_FUNCTION = {"bound": D, "separator": SEP, "sum": lambda s: D(s) + SEP(s), "lex": lambda s: LEX_SCALE * D(s) + SEP(s),
               "weighted": lambda s: D(s) + CouplingPolicy.neighbour_sum(s, D(s)), "confident": lambda s: -s.ent}


def test_fits_rules():
    k = DistanceKernel(unscreened=[0.0, 1.0])
    t = lambda *v: torch.tensor(v, dtype=torch.float64)
    spent, load, count = t(0.0, 0.5, 0.5), t(0.2, 0.2, 0.7), torch.tensor([0, 1, 1])
    fits = lambda pol, **kw: pol._fits(spent, load, count, **kw).tolist()
    assert fits(CouplingPolicy(k, 0.2, mixed=False)) == [True, True, False]                    # load <= theta
    assert fits(CouplingPolicy(k, 0.0, mixed=False, budget=0.8)) == [True, True, False]        # spent + load <= budget
    assert fits(CouplingPolicy(k, 0.0, mixed=False, rate=0.3)) == [True, False, False]         # ... <= rate * (count + 1)
    assert fits(CouplingPolicy(k, 0.0, mixed=False, rate=0.3, budget=9.0)) == [True, False, False]   # rate overrides budget
    assert fits(CouplingPolicy(k, 0.2, mixed=False, max_set=1)) == [True, False, False]        # and the step is below its cap
    spent, load, count = t(0.0, 0.0, 0.0), t(0.0, PULL_TOL / 2, 1e-3), torch.tensor([0, 0, 0])
    assert fits(CouplingPolicy(k, 0.0, mixed=False)) == [True, False, False]
    assert fits(CouplingPolicy(k, 0.0, mixed=False), pulled=True) == [True, True, False]       # pilot noise is not coupling
    spent = t(1.0, 1.0, 1.0)                                                                   # the rate is used up
    assert fits(CouplingPolicy(k, 0.0, mixed=False, rate=1e-9), pulled=True) == [True, True, False]
    assert fits(CouplingPolicy(k, 0.0, mixed=False, rate=1e-9, max_set=1), pulled=True) == [True, True, False]
    count = torch.tensor([1, 1, 1])
    assert fits(CouplingPolicy(k, 0.0, mixed=False, rate=1e-9, max_set=1), pulled=True) == [False, False, False]


@pytest.mark.parametrize("name", sorted(AS_FUNCTION))
@pytest.mark.parametrize("B", [None, 0.3])
def test_named_priority_equals_its_function(name, B):
    N = 24
    den = noisy_hmm(N, 3, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    route = torch_route(C.dfa_exactly_k(0, N // 3, 3))
    kern, _ = fit_distance_kernel(den, N, 100, torch.Generator().manual_seed(0))
    run = lambda pr: twisted_smc(den, cl, route, N, N, nrun=8, K=4, gen=torch.Generator().manual_seed(3), ess_min=0.0,
                                 schedule=CouplingPolicy(kern, 0.05, priority=pr, mixed=False, rate=None if B is None else B / N))   # theta = 0.05 when B is None
    a, b = run(name), run(AS_FUNCTION[name])
    assert np.array_equal(a["logZ"], b["logZ"]) and np.array_equal(a["steps"], b["steps"])
    assert a["steps"].mean() < N                                         # several positions per step were revealed


def fresh_state(den, cl, route, P=6):
    x = torch.full((P, den.N), den.mask_id)
    return RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)


@pytest.mark.parametrize("priority", list(PRIORITIES) + ["function"])
def test_admitted_set_obeys_the_rule_whatever_the_score(priority):
    N = 20
    den = noisy_hmm(N, 3, 0.85, 0.3, seed=2, concentration=2.0); cl = den.classes()
    route = torch_route(C.dfa_exactly_k(0, 6, 3))
    kern = DistanceKernel(unscreened=[0.0, 0.9, 0.4] + [0.0] * (N - 3))       # coupled within distance 2
    pr = (lambda s: torch.rand(s.masked.shape, generator=torch.Generator().manual_seed(7)).double()) if priority == "function" else priority
    gen = lambda: torch.Generator().manual_seed(11)
    chosen = lambda r, v: [sorted(r[p][v[p]].tolist()) for p in range(r.shape[0])]
    st = fresh_state(den, cl, route)
    sets = chosen(*CouplingPolicy(kern, 0.0, priority=pr, mixed=False).reveal(st, 0, N, gen()))
    assert all(b - a > 2 for s in sets for a, b in zip(s, s[1:])) and all(len(s) >= 1 for s in sets)
    pol = CouplingPolicy(kern, 0.0, priority=pr, mixed=False, budget=1.0)
    sets_b = chosen(*pol.reveal(st, 0, N, gen()))
    assert pol.load_sum <= 1.0 * len(sets_b) + 1e-12
    for s in sets_b:                                                     # the load recomputed from the kernel, in any order
        pair = sum(float(kern.ku[b - a]) for i, a in enumerate(s) for b in s[i + 1:])
        assert pair <= 1.0 + 1e-12
    sets_c = chosen(*CouplingPolicy(kern, 5.0, priority=pr, mixed=False, max_set=3).reveal(st, 0, N, gen()))
    assert all(len(s) == 3 for s in sets_c)


def test_function_priority_is_unbiased_on_screened_reveals():
    N = 12
    den = lazy_chain(N, 3, 0.6, seed=3, concentration=2.0); cl = den.classes()
    d = C.dfa_exactly_k(0, 4, 3); truth = ConstrainedTruth(den, cl, d)
    kern = DistanceKernel(unscreened=[0.0] + [1.0] * (N - 1), screened=[0.0] * N)      # the chain's true structure
    exposure = lambda s: torch.sqrt(D(s)) * (1.0 + SEP(s))                              # a state-dependent score
    out = twisted_smc(den, cl, torch_route(d), N, N, nrun=3000, K=2, gen=torch.Generator().manual_seed(4), ess_min=0.0,
                      schedule=CouplingPolicy(kern, 0.0, priority=exposure, mixed=False))
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    assert out["steps"].mean() < 0.8 * N
    assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se
