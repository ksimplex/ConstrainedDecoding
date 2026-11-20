"""Oracle denoisers (models/oracle.py): exact conditionals of a known hidden Markov model.

* `log_cond` and `log_joint` against a joint table built by enumerating hidden paths;
* the lazy chain's closed-form d-step predictive, and exact screening by a revealed position;
* the chain rule: one-at-a-time reveals in any order reproduce log p(x), so the parallel-reveal gap r is 0
  for single reveals, 0 for a mean-field model under any grouping, 0 for chain positions separated by a
  revealed one, and non-zero for adjacent chain positions and for a hidden chain;
* `ConstrainedTruth` (log p(C), marginals, exact sampler) against enumeration;
* end to end: `twisted_smc` with one reveal per step has mean Z_hat equal to the exact p(C).
"""
import itertools
import math
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.decode import twisted_smc
from twistroute.models import (OracleHMM, ConstrainedTruth, CopyTruth, lazy_chain, noisy_hmm, independent,
                               parallel_log_prob)
from twistroute.torch_routes import torch_route
from conftest import tv_vs_law, tv_threshold


def random_hmm(N, H, S, seed):
    r = np.random.default_rng(seed)
    return OracleHMM(r.dirichlet(np.ones(H)), r.dirichlet(np.ones(H), (N - 1, H)), r.dirichlet(np.ones(S), (N, H)))


def joint_table(den):
    """p(x) for every token sequence, by summing over every hidden path (independent of forward-backward)."""
    N, H, S = den.N, den.H, den.S
    pi, T, E = den.pi.numpy(), den.trans.numpy(), den.emit.numpy()
    J = np.zeros((S,) * N)
    for h in itertools.product(range(H), repeat=N):
        ph = pi[h[0]] * math.prod(T[i, h[i], h[i + 1]] for i in range(N - 1))
        rows = [E[i, h[i]] for i in range(N)]
        outer = rows[0]
        for rrow in rows[1:]:
            outer = np.multiply.outer(outer, rrow)
        J += ph * outer
    return J


MODELS = {"hmm": lambda: random_hmm(5, 3, 3, 1), "chain": lambda: lazy_chain(5, 3, 0.6, seed=2, concentration=2.0),
          "noisy": lambda: noisy_hmm(5, 3, 0.7, 0.3, seed=3, concentration=2.0),
          "indep": lambda: independent(np.random.default_rng(4).dirichlet(np.ones(3), 5))}


@pytest.mark.parametrize("name", list(MODELS))
def test_conditionals_and_joint_match_enumeration(name):
    den = MODELS[name](); J = joint_table(den); N, S = den.N, den.S
    assert abs(J.sum() - 1) < 1e-12
    r = np.random.default_rng(0)
    xs = torch.tensor(r.integers(0, S, (40, N)))
    mask = torch.tensor(r.random((40, N)) < 0.5)
    x = torch.where(mask, torch.full_like(xs, den.mask_id), xs)
    lc, lj = den.log_cond(x).numpy(), den.log_joint(x).numpy()
    for b in range(40):
        rev = [i for i in range(N) if not mask[b, i]]
        sl = tuple(int(xs[b, i]) if i in rev else slice(None) for i in range(N))
        assert abs(lj[b] - np.log(J[sl].sum())) < 1e-10                      # marginal of the revealed tokens
        for i in range(N):                                                   # leave-one-out predictive at i
            others = [j for j in rev if j != i]
            sl = tuple(int(xs[b, j]) if j in others else slice(None) for j in range(N))
            sub = J[sl]
            ax = tuple(k for k, j in enumerate([j for j in range(N) if j not in others]) if j != i)
            p = sub.sum(axis=ax); p = p / p.sum()
            assert np.allclose(np.exp(lc[b, i]), p, atol=1e-10)
    lg = den.logits(x)
    assert lg.shape == (40, N, S + 1) and torch.isneginf(lg[..., den.mask_id]).all()
    assert torch.allclose(lg[..., :S].exp().sum(-1), torch.ones(40, N), atol=1e-5)


def test_lazy_chain_closed_form_and_mutual_information():
    N, S, lam = 9, 4, 0.7
    den = lazy_chain(N, S, lam, seed=5, concentration=3.0)
    pi = den.pi.numpy()
    x = torch.full((1, N), den.mask_id); x[0, 2] = 1
    p = np.exp(den.log_cond(x)[0].numpy())
    for j in range(N):
        if j == 2:
            continue
        d = abs(j - 2)
        want = (1 - lam ** d) * pi + lam ** d * np.eye(S)[1]
        assert np.allclose(p[j], want, atol=1e-12)
    mi = [den.pair_mi(0, d) for d in range(1, N)]
    assert all(a > b > 0 for a, b in zip(mi, mi[1:]))                        # decays with distance
    assert lazy_chain(N, S, 0.0).pair_mi(0, 1) < 1e-12
    seg = lazy_chain(N, S, lam, breaks=(3,))
    assert seg.pair_mi(3, 4) < 1e-12 and seg.pair_mi(2, 3) > 0.1             # independent across the break


def reveal(den, x_full, sets):
    """Reveal `sets` in order; returns the summed parallel log-probability of the trajectory."""
    x = torch.full_like(x_full, den.mask_id); tot = torch.zeros(x_full.shape[0], dtype=torch.float64)
    for s in sets:
        x2 = x.clone(); x2[:, s] = x_full[:, s]
        tot += parallel_log_prob(den, x, x2); x = x2
    assert (x == x_full).all()
    return tot


def test_chain_rule_and_parallel_gap():
    g = torch.Generator().manual_seed(0)
    chain, hmm = lazy_chain(5, 3, 0.7, seed=1, concentration=2.0), noisy_hmm(5, 3, 0.7, 0.3, seed=1)
    ind = MODELS["indep"]()
    for den in (chain, hmm, ind):
        x = den.sample(64, g)
        lj = den.log_joint(x)
        for order in ([0, 1, 2, 3, 4], [3, 0, 4, 2, 1]):                     # single reveals: r = 0 in any order
            assert torch.allclose(reveal(den, x, [[i] for i in order]), lj, atol=1e-10)
    x = ind.sample(64, g)                                                    # mean-field: any grouping is exact
    assert torch.allclose(reveal(ind, x, [[0, 1, 2], [3, 4]]), ind.log_joint(x), atol=1e-10)
    x = chain.sample(64, g)
    r_sep = chain.log_joint(x) - reveal(chain, x, [[2], [1, 3], [0, 4]])     # every co-revealed pair is screened
    r_adj = chain.log_joint(x) - reveal(chain, x, [[2], [0, 1], [3, 4]])     # adjacent positions together
    assert r_sep.abs().max() < 1e-10 and r_adj.abs().max() > 0.05
    x = hmm.sample(64, g)                                                    # hidden chain: no screening
    assert (hmm.log_joint(x) - reveal(hmm, x, [[2], [1, 3], [0, 4]])).abs().max() > 1e-3


DFAS = {"exactly_k": lambda: C.dfa_exactly_k(0, 2, 3), "forbidden": lambda: C.dfa_forbidden([(0, 0), (1, 2)], 3)}


@pytest.mark.parametrize("model", ["hmm", "chain", "noisy"])
@pytest.mark.parametrize("dfa", list(DFAS))
def test_constrained_truth_matches_enumeration(model, dfa, rng):
    den, d = MODELS[model](), DFAS[dfa]()
    J = joint_table(den); N, S = den.N, den.S
    law = {y: J[y] for y in itertools.product(range(S), repeat=N) if d.accepts(y)}
    pC = sum(law.values()); law = {k: v / pC for k, v in law.items()}
    tr = ConstrainedTruth(den, den.classes(), d)
    assert abs(tr.log_pC - np.log(pC)) < 1e-10
    marg = np.zeros((N, S))
    for y, p in law.items():
        for i, c in enumerate(y):
            marg[i, c] += p
    assert np.allclose(tr.marginals, marg, atol=1e-10)
    draws = [tuple(int(v) for v in row) for row in tr.sample(4000, rng)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0 and tv < tv_threshold(law, 4000, rng)


@pytest.mark.parametrize("K", [1, 4])
def test_smc_matches_exact_constrained_probability(K):
    """One reveal per step: the native law is the model's joint, so E[Z_hat] = p(C) exactly."""
    N = 8
    den = lazy_chain(N, 3, 0.6, seed=7, concentration=2.0)
    d = C.dfa_exactly_k(0, 3, 3)
    cl = den.classes()
    truth = ConstrainedTruth(den, cl, d)
    out = twisted_smc(den, cl, torch_route(d), N, N, nrun=6000, K=K, gen=torch.Generator().manual_seed(11),
                      defensive=0.25)
    z = np.exp(out["logZ"])
    se = z.std(ddof=1) / math.sqrt(len(z))
    assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se
    assert out["frac_valid_finite"] == 1.0


@pytest.mark.parametrize("model", ["hmm", "chain"])
def test_pair_marginals_match_enumeration(model):
    den, d = MODELS[model](), DFAS["forbidden"]()
    J = joint_table(den); N, S = den.N, den.S
    law = {y: J[y] for y in itertools.product(range(S), repeat=N) if d.accepts(y)}
    pC = sum(law.values())
    want = np.zeros((N - 1, S, S))
    for y, p in law.items():
        for i in range(N - 1):
            want[i, y[i], y[i + 1]] += p / pC
    assert np.allclose(ConstrainedTruth(den, den.classes(), d).pair_marginals(), want, atol=1e-10)


@pytest.mark.parametrize("make", [lambda: random_hmm(6, 2, 3, 5), lambda: lazy_chain(6, 3, 0.6, seed=2, concentration=2.0),
                                  lambda: lazy_chain(6, 3, 0.7, seed=4, concentration=2.0, breaks=(1, 3))])
def test_copy_truth_matches_enumeration(make):
    den = make(); J = joint_table(den); N, S = den.N, den.S; n = N // 2
    law = {y: J[y] for y in itertools.product(range(S), repeat=N) if all(y[i] == y[n + i] for i in range(n))}
    pC = sum(law.values())
    marg = np.zeros((N, S))
    for y, p in law.items():
        for i, c in enumerate(y):
            marg[i, c] += p / pC
    tr = CopyTruth(den, den.classes())
    assert abs(tr.log_pC - np.log(pC)) < 1e-10 and np.allclose(tr.marginals, marg, atol=1e-10)


def test_new_automata_accept_the_right_strings():
    from twistroute.torch_routes import build_route
    chart = build_route(dict(kind="cfg", grammar="dyck1_unamb", A=2, N=8))
    d = C.dfa_dyck1(8)
    ys = list(itertools.product(range(2), repeat=8))
    assert [d.accepts(y) for y in ys] == chart.accepts(torch.tensor(ys)).tolist()
    assert sum(d.accepts(y) for y in ys) == 14                                  # Catalan number C_4
    par = C.dfa_parity(0, 3)
    seg = C.dfa_segment_counts(0, [3, 2], [1, 2], 3)
    for y in itertools.product(range(3), repeat=5):
        assert par.accepts(y) == (sum(v == 0 for v in y) % 2 == 0)
        assert seg.accepts(y) == (sum(v == 0 for v in y[:3]) == 1 and sum(v == 0 for v in y[3:]) == 2)
