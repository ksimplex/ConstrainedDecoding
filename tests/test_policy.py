"""Constraint-aware reveal policy (decode/policy.py).

* route queries: exact constrained marginals and pairwise mutual information against enumeration, with some
  positions revealed;
* the distance kernel: screening logic, and the pilot fit on oracle models (zero for a mean-field model,
  exactly zero when a revealed position screens a Markov chain);
* the policy as a scheduler: a mean-field model is decoded in one step with a constant weight equal to p(C);
  on a Markov chain, co-revealing only screened positions leaves no parallel-reveal gap and Z_hat stays
  unbiased for the exact p(C) in fewer than N steps; with a positive threshold the sampler is still exact for
  its own native law (E[Z_hat] equals the hit rate of native sampling under the same policy);
* forced positions are considered first but still pass through the coupling test.
"""
import itertools
import math
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.decode import (twisted_smc, native_sample, CouplingPolicy, DistanceKernel, fit_distance_kernel,
                               constrained_marginals, pairwise_mi)
from twistroute.decode.policy import RevealState
from twistroute.decode.schedule import clamp_revealed
from twistroute.models import ConstrainedTruth, lazy_chain, noisy_hmm, independent, parallel_log_prob
from twistroute.torch_routes import build_route, torch_route

MASK = 3


def enum_pairs(lw, accepts, clamp):
    """Constrained product law by enumeration -> (marginals (N, A), pairwise MI (N, N))."""
    N, A = lw.shape
    law = {}
    for y in itertools.product(range(A), repeat=N):
        if accepts(y) and all(y[i] == c for i, c in clamp.items()):
            law[y] = math.exp(sum(lw[i, y[i]] for i in range(N) if i not in clamp))
    Z = sum(law.values())
    mu = np.zeros((N, A)); joint = np.zeros((N, N, A, A))
    for y, p in law.items():
        for i in range(N):
            mu[i, y[i]] += p / Z
            for j in range(N):
                joint[i, j, y[i], y[j]] += p / Z
    mi = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                ind = np.outer(mu[i], mu[j]); nz = joint[i, j] > 0
                mi[i, j] = (joint[i, j][nz] * np.log(joint[i, j][nz] / ind[nz])).sum()
    return mu, mi


@pytest.mark.parametrize("spec", [dict(kind="exactly_k", A=3, cls=0, count=2, N=6),
                                  dict(kind="forbidden", A=3, patterns=[[0, 0], [1, 2]], N=6),
                                  dict(kind="copy", A=3, pairs=3, N=6)])
def test_marginals_and_pairwise_mi_match_enumeration(spec):
    N, A = spec["N"], spec["A"]
    route = build_route(spec)
    torch.manual_seed(1)
    lw = torch.log_softmax(torch.randn(2, N, A), -1)
    x = torch.full((2, N), MASK); y = torch.zeros(2, N, dtype=torch.long)
    x[1, 4] = 1; y[1, 4] = 1                                              # second row: position 4 revealed as class 1
    lwc = clamp_revealed(lw, x, y, MASK)
    mu = constrained_marginals(route, lwc)
    mi = pairwise_mi(route, lwc, mu)
    acc = lambda s: bool(route.accepts(torch.tensor([s]))[0])
    for b, clamp in enumerate([{}, {4: 1}]):
        mu_e, mi_e = enum_pairs(lw[b].double().numpy(), acc, clamp)
        assert np.allclose(mu[b].numpy(), mu_e, atol=1e-9)
        keep = [i for i in range(N) if i not in clamp]
        assert np.allclose(mi[b].numpy()[np.ix_(keep, keep)], mi_e[np.ix_(keep, keep)], atol=1e-9)


def test_distance_kernel_screening():
    k = DistanceKernel(unscreened=[0, 5, 4, 3, 2], screened=[0, .5, .4, .3, .2])
    masked = torch.tensor([[True, True, False, True, True]])              # position 2 revealed
    M = k.matrix(masked)[0].numpy()
    assert M[0, 1] == 5 and M[3, 4] == 5                                  # same side: unscreened
    assert M[1, 3] == .4 and M[0, 4] == .2 and M[0, 3] == .3              # across the revealed position: screened
    assert np.allclose(M, M.T) and np.allclose(np.diag(M), 0)


def test_pilot_kernel_on_oracle_models():
    g = torch.Generator().manual_seed(0)
    N = 12
    ind = independent(np.random.default_rng(0).dirichlet(np.ones(3), N))
    k, cnt = fit_distance_kernel(ind, N, 64, g)
    assert float(k.ku.abs().max()) < 1e-12 and float(k.ks.abs().max()) < 1e-12
    chain = lazy_chain(N, 3, 0.7)
    k, cnt = fit_distance_kernel(chain, N, 400, g)
    assert float(k.ks.abs().max()) < 1e-12                               # a revealed position screens a chain exactly
    ku = k.ku.numpy()
    assert ku[1] > ku[2] > ku[3] > ku[4] > 0                              # decays with distance
    assert abs(ku[1] - chain.pair_mi(0, 1)) < 0.5 * chain.pair_mi(0, 1)   # same scale as the exact adjacent MI
    hmm = noisy_hmm(N, 3, 0.7, 0.3)
    k, cnt = fit_distance_kernel(hmm, N, 400, g)
    assert float(k.ks[2:5].min()) > 1e-4                                 # a hidden chain is not screened


def setup_chain(N=12, lam=0.6):
    den = lazy_chain(N, 3, lam, seed=7, concentration=2.0)
    d = C.dfa_exactly_k(0, N // 3, 3)
    return den, den.classes(), d, torch_route(d), ConstrainedTruth(den, den.classes(), d)


def test_mean_field_model_is_decoded_in_one_step():
    N = 10
    den = independent(np.random.default_rng(3).dirichlet(np.ones(3), N))
    d = C.dfa_exactly_k(0, 3, 3); cl = den.classes()
    truth = ConstrainedTruth(den, cl, d)
    kern, _ = fit_distance_kernel(den, N, 64, torch.Generator().manual_seed(0))
    out = twisted_smc(den, cl, torch_route(d), N, N, nrun=200, K=2, gen=torch.Generator().manual_seed(1),
                      schedule=CouplingPolicy(kern, 0.0), return_particles=True)
    assert (out["steps"] == 1).all()
    assert np.allclose(out["logZ"], truth.log_pC, atol=1e-6)             # constant weight: Z_hat = p(C) in every run
    assert out["frac_valid_finite"] == 1.0


def test_screened_reveals_have_no_gap_and_stay_unbiased():
    den, cl, d, route, truth = setup_chain()
    N = den.N
    kern = DistanceKernel(unscreened=[0.0] + [1.0] * (N - 1), screened=[0.0] * N)   # the chain's true structure
    pol = CouplingPolicy(kern, 0.0, priority="random", mixed=False)
    out = twisted_smc(den, cl, route, N, N, nrun=4000, K=1, gen=torch.Generator().manual_seed(2), schedule=pol,
                      return_particles=True, path_stat=lambda a, b: parallel_log_prob(den, a, b))
    r = den.log_joint(torch.tensor(out["tokens"]).view(-1, N)).numpy() - out["path_stat"].reshape(-1)
    assert np.abs(r).max() < 1e-8                                        # co-revealed positions were independent
    assert out["steps"].mean() < 0.7 * N and out["steps"].max() <= N
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se


@pytest.mark.parametrize("mixed", [False, True])
def test_policy_is_exact_for_its_own_native_law(mixed):
    den, cl, d, route, truth = setup_chain(N=9)
    N = den.N
    kern, _ = fit_distance_kernel(den, N, 400, torch.Generator().manual_seed(0))
    pol = CouplingPolicy(kern, 0.15, priority="bound", mixed=mixed)
    g = torch.Generator().manual_seed(5)
    out = twisted_smc(den, cl, route, N, N, nrun=4000, K=2, gen=g, schedule=pol, defensive=0.25)
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    n = 40000
    hits = route.accepts(native_sample(den, cl, N, N, n, torch.Generator().manual_seed(6), schedule=pol, route=route))
    h0 = float(hits.float().mean()); se0 = math.sqrt(h0 * (1 - h0) / n)
    assert abs(z.mean() - h0) < 4 * math.hypot(se, se0)
    assert out["steps"].mean() < N                                       # it did reveal several positions per step


def test_forced_positions_go_first_but_respect_coupling():
    N, A = 8, 3
    route = build_route(dict(kind="copy", A=A, pairs=4, N=N))
    den = lazy_chain(N, A, 0.5); cl = den.classes()
    x = torch.full((1, N), den.mask_id); x[0, 1] = 2                      # position 1 revealed: partner 5 is forced
    st = RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)
    dense = DistanceKernel(unscreened=[0.0] + [9.0] * (N - 1))            # everything strongly coupled
    r, valid = CouplingPolicy(dense, 0.0, priority="random", mixed=False).reveal(st, 0, N, torch.Generator().manual_seed(0))
    assert r[0][valid[0]].tolist() == [5]                                 # the forced partner goes first, alone
    local = DistanceKernel(unscreened=[0.0, 9.0] + [0.0] * (N - 2))        # only neighbours are coupled
    r, valid = CouplingPolicy(local, 0.0, priority="random", mixed=False).reveal(st, 0, N, torch.Generator().manual_seed(0))
    got = set(r[0][valid[0]].tolist())
    assert 5 in got and not ({4, 6} & got) and len(got) >= 3              # its neighbours wait, others join
    r, valid = CouplingPolicy(dense, 0.0, priority="confident", mixed=False, forced_first=False).reveal(st, 0, N)
    assert int(valid.sum()) == 1
