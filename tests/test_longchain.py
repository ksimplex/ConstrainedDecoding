"""
Long-chain building blocks for the parallel-unmasking experiments, each checked against brute-force enumeration:
the saturating at-least-k automaton; TorchSubset (a route on a subset of positions, the rest free); exact
constrained truth restricted to a subset of positions (ConstrainedTruth(active=...)) and to a subset of copy
pairs (CopyTruth(pairs=...)); the anticorrelated chain; and the policy's on-demand kernel rows, which must give
the same reveal sets as the (P, N, N) matrix they replace.
"""
import itertools
import math
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.decode import twisted_smc, CouplingPolicy, DistanceKernel, fit_distance_kernel, constrained_marginals
from twistroute.decode.policy import RevealState
from twistroute.models import ConstrainedTruth, CopyTruth, lazy_chain, noisy_hmm, repel_chain, parallel_log_prob
from twistroute.torch_routes import TorchChain, TorchCopy, TorchSubset, torch_route
from conftest import tv_vs_law, tv_threshold
from test_oracle import joint_table, random_hmm


def test_at_least_k_automaton():
    d = C.dfa_at_least_k(1, 3, 3)
    assert d.n == 4                                                       # saturating counter: K + 1 states
    for y in itertools.product(range(3), repeat=6):
        assert d.accepts(y) == (sum(v == 1 for v in y) >= 3)


SUBSETS = {"window_count": (lambda: C.dfa_at_least_k(0, 2, 3), [2, 3, 4, 5]),
           "scattered_motif": (lambda: C.dfa_forbidden([(0, 0), (1, 2)], 3), [5, 0, 3]),      # order matters here
           "whole": (lambda: C.dfa_exactly_k(0, 2, 3), list(range(7)))}


@pytest.mark.parametrize("name", list(SUBSETS))
def test_subset_route_matches_enumeration(name, rng):
    make, pos = SUBSETS[name]
    d, N, A = make(), 7, 3
    route = TorchSubset(TorchChain(d), pos, N)
    g = torch.Generator().manual_seed(0)
    lw = torch.log_softmax(torch.randn(3, N, A, generator=g, dtype=torch.float64), -1)
    lw[1, 2] = torch.tensor([float("-inf"), 0.0, float("-inf")], dtype=torch.float64)       # a revealed position
    lw[2] = 1.7 * torch.randn(N, A, generator=g, dtype=torch.float64)                       # unnormalised weights
    seqs = list(itertools.product(range(A), repeat=N))
    ok = np.array([d.accepts(tuple(y[p] for p in pos)) for y in seqs])
    Y = torch.tensor(seqs)
    assert (route.accepts(Y).numpy() == ok).all()
    for b in range(3):
        w = np.exp(lw[b].numpy()[np.arange(N)[None], np.array(seqs)].sum(-1)) * ok
        assert abs(float(route.logZ(lw[b:b + 1])) - math.log(w.sum())) < 1e-10
        marg = np.zeros((N, A))
        for y, p in zip(seqs, w / w.sum()):
            marg[np.arange(N), y] += p
        assert np.allclose(constrained_marginals(route, lw[b:b + 1])[0].numpy(), marg, atol=1e-9)
        law = {y: p for y, p in zip(seqs, w / w.sum()) if p > 0}
        draws = [tuple(int(v) for v in row) for row in route.sample(lw[b:b + 1].expand(6000, N, A).float(), g)]
        tv, invalid = tv_vs_law(draws, law)
        assert invalid == 0 and tv < tv_threshold(law, 6000, rng)


def test_subset_route_rejects_bad_positions():
    inner = TorchChain(C.dfa_at_least_k(0, 1, 3))
    with pytest.raises(ValueError):
        TorchSubset(inner, [1, 1], 5)
    with pytest.raises(ValueError):
        TorchSubset(inner, [1, 5], 5)


WMODELS = {"hmm": lambda: random_hmm(6, 3, 3, 1), "chain": lambda: lazy_chain(6, 3, 0.6, seed=2, concentration=2.0),
           "noisy": lambda: noisy_hmm(6, 3, 0.7, 0.3, seed=3, concentration=2.0), "repel": lambda: repel_chain(6, 3, 0.4)}
WINDOWS = {"contiguous": [False, True, True, True, True, False], "split": [True, False, True, False, False, True],
           "all": [True] * 6}


@pytest.mark.parametrize("model", list(WMODELS))
@pytest.mark.parametrize("window", list(WINDOWS))
@pytest.mark.parametrize("dfa", ["at_least", "forbidden"])
def test_windowed_truth_matches_enumeration(model, window, dfa, rng):
    den = WMODELS[model]()
    d = C.dfa_at_least_k(0, 2, 3) if dfa == "at_least" else C.dfa_forbidden([(0, 0), (1, 2)], 3)
    act = np.array(WINDOWS[window]); pos = np.nonzero(act)[0]
    J = joint_table(den); N, S = den.N, den.S
    law = {y: J[y] for y in itertools.product(range(S), repeat=N) if d.accepts(tuple(y[p] for p in pos))}
    pC = sum(law.values()); law = {k: v / pC for k, v in law.items()}
    tr = ConstrainedTruth(den, den.classes(), d, active=act)
    assert abs(tr.log_pC - np.log(pC)) < 1e-10
    marg, pair = np.zeros((N, S)), np.zeros((N - 1, S, S))
    for y, p in law.items():
        marg[np.arange(N), y] += p
        for i in range(N - 1):
            pair[i, y[i], y[i + 1]] += p
    assert np.allclose(tr.marginals, marg, atol=1e-10)
    assert np.allclose(tr.pair_marginals(), pair, atol=1e-10)
    draws = [tuple(int(v) for v in row) for row in tr.sample(3000, rng)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0 and tv < tv_threshold(law, 3000, rng)


def test_windowed_truth_agrees_with_the_subset_route():
    """E[Z_hat] of the sampler with a TorchSubset route equals p(C) from the windowed truth (one reveal per step,
    no resampling, so the estimate has no hidden mass)."""
    N = 10
    den = lazy_chain(N, 3, 0.6, seed=7, concentration=2.0); cl = den.classes()
    d, pos = C.dfa_at_least_k(0, 3, 3), [3, 4, 5, 6, 7]
    act = np.zeros(N, dtype=bool); act[pos] = True
    truth = ConstrainedTruth(den, cl, d, active=act)
    out = twisted_smc(den, cl, TorchSubset(TorchChain(d), pos, N), N, N, nrun=6000, K=2,
                      gen=torch.Generator().manual_seed(11), ess_min=0.0)
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se and out["frac_valid_finite"] == 1.0


@pytest.mark.parametrize("make", [lambda: random_hmm(6, 2, 3, 5), lambda: lazy_chain(6, 3, 0.6, seed=2, concentration=2.0),
                                  lambda: repel_chain(6, 3, 0.4), lambda: noisy_hmm(6, 3, 0.7, 0.3, seed=3)])
@pytest.mark.parametrize("pairs", [[0], [1], [0, 2], [0, 1, 2], []])
def test_partial_copy_truth_matches_enumeration(make, pairs):
    den = make(); J = joint_table(den); N, S = den.N, den.S; n = N // 2
    law = {y: J[y] for y in itertools.product(range(S), repeat=N) if all(y[i] == y[n + i] for i in pairs)}
    pC = sum(law.values())
    marg = np.zeros((N, S))
    for y, p in law.items():
        marg[np.arange(N), y] += p / pC
    tr = CopyTruth(den, den.classes(), pairs=pairs)
    assert abs(tr.log_pC - np.log(pC)) < 1e-10 and np.allclose(tr.marginals, marg, atol=1e-10)
    route = TorchCopy([(i, n + i) for i in pairs], N, S) if pairs else None
    if route is not None:                                                 # the route encodes the same constraint
        Y = torch.tensor(list(itertools.product(range(S), repeat=N)))
        assert (route.accepts(Y).numpy() == np.array([y in law for y in map(tuple, Y.tolist())])).all()


def test_copy_truth_default_is_every_pair():
    den = lazy_chain(6, 3, 0.6, seed=2, concentration=2.0)
    a, b = CopyTruth(den, den.classes()), CopyTruth(den, den.classes(), pairs=[0, 1, 2])
    assert a.log_pC == b.log_pC and np.array_equal(a.marginals, b.marginals)


def test_repel_chain_is_anticorrelated():
    den = repel_chain(5, 3, 0.4)
    J = joint_table(den)
    pi = den.pi.numpy()
    for i in range(5):                                                    # stationary at every position
        assert np.allclose(J.sum(tuple(k for k in range(5) if k != i)), pi, atol=1e-12)
    for dgap in (1, 2, 3):                                                # correlation (-mu)^d of token indicators
        P2 = J.sum(tuple(k for k in range(5) if k not in (0, dgap)))
        for a in range(3):
            corr = (P2[a, a] - pi[a] ** 2) / (pi[a] * (1 - pi[a]))
            assert abs(corr - (-0.4) ** dgap) < 1e-12
    assert float(den.trans.min()) > 0                                     # every token keeps positive probability
    with pytest.raises(ValueError):
        repel_chain(5, 3, 0.5)                                            # the bound itself is excluded
    assert abs(repel_chain(5, 3, 0.0).pair_mi(0, 1)) < 1e-12              # mu = 0 is independence


def test_kernel_rows_match_the_matrix():
    g = torch.Generator().manual_seed(0)
    P, N = 7, 13
    k = DistanceKernel(torch.rand(N, generator=g).double(), torch.rand(N, generator=g).double())
    masked = torch.rand(P, N, generator=g) < 0.6
    M, counts = k.matrix(masked), k.counts(masked)
    for c in range(N):
        cc = torch.full((P,), c)
        assert torch.equal(k.row(counts, cc), M[:, c])
    c = torch.randint(0, N, (P,), generator=g)
    assert torch.equal(k.row(counts, c), M[torch.arange(P), c])


class _MatrixOnly:
    """A kernel without `row`: forces the policy onto the (P, N, N) path."""
    def __init__(self, k):
        self.k = k
    def matrix(self, masked):
        return self.k.matrix(masked)


@pytest.mark.parametrize("priority", ["bound", "confident", "random"])
@pytest.mark.parametrize("theta", [0.0, 0.02, 0.3])
def test_lean_policy_gives_the_same_reveal_sets(priority, theta):
    N = 14
    den = noisy_hmm(N, 3, 0.8, 0.3, seed=1, concentration=2.0); cl = den.classes()
    route = TorchSubset(TorchChain(C.dfa_at_least_k(0, 3, 3)), [4, 5, 6, 7, 8, 9], N)
    kern, _ = fit_distance_kernel(den, N, 300, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(3)
    x = den.sample(40, g)
    x = torch.where(torch.rand(40, N, generator=g) < 0.4, x, torch.full_like(x, den.mask_id))
    st = RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)
    a = CouplingPolicy(kern, theta, priority=priority, mixed=False).reveal(st, 0, N, torch.Generator().manual_seed(9))
    b = CouplingPolicy(_MatrixOnly(kern), theta, priority=priority, mixed=False).reveal(st, 0, N, torch.Generator().manual_seed(9))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    if theta >= 0.3:                                                     # a hidden chain has no exact screening, so
        assert a[1].sum(-1).float().mean() > 1                           # sets of several positions need theta > 0


def test_policy_on_a_windowed_constraint_is_gap_free_on_a_chain():
    """Markov chain, at least k of a window: with the chain's true structure as the kernel, theta = 0 reveals
    several positions per step, r is zero on every path, and Z_hat stays unbiased for p(C)."""
    N = 16
    den = lazy_chain(N, 3, 0.7, seed=7, concentration=2.0); cl = den.classes()
    d, pos = C.dfa_at_least_k(0, 4, 3), list(range(5, 11))
    act = np.zeros(N, dtype=bool); act[pos] = True
    truth = ConstrainedTruth(den, cl, d, active=act)
    kern = DistanceKernel(unscreened=[0.0] + [1.0] * (N - 1), screened=[0.0] * N)
    out = twisted_smc(den, cl, TorchSubset(TorchChain(d), pos, N), N, N, nrun=4000, K=1,
                      gen=torch.Generator().manual_seed(2), schedule=CouplingPolicy(kern, 0.0, mixed=False),
                      return_particles=True, path_stat=lambda a, b: parallel_log_prob(den, a, b))
    r = den.log_joint(torch.tensor(out["tokens"]).view(-1, N)).numpy() - out["path_stat"].reshape(-1)
    assert np.abs(r).max() < 1e-8 and out["steps"].mean() < 0.7 * N
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se and out["frac_valid_finite"] == 1.0


def test_kernel_apply_matches_the_matrix():
    g = torch.Generator().manual_seed(1)
    P, N = 6, 15
    ku = torch.rand(N, generator=g).double(); ku[9:] = 0.0                # a kernel with finite range
    k = DistanceKernel(ku, 0.3 * ku)
    masked = torch.rand(P, N, generator=g) < 0.6
    v = torch.rand(P, N, generator=g).double() * masked
    assert torch.allclose(k.apply(masked, v), (k.matrix(masked) @ v[..., None]).squeeze(-1), atol=1e-12)
    full = DistanceKernel(torch.rand(N, generator=g).double())           # range N - 1
    assert torch.allclose(full.apply(masked, v), (full.matrix(masked) @ v[..., None]).squeeze(-1), atol=1e-12)


def _state(N=15, reveal=()):
    den = lazy_chain(N, 3, 0.7, seed=7, concentration=2.0); cl = den.classes()
    route = TorchSubset(TorchChain(C.dfa_at_least_k(0, 2, 3)), [10, 11, 12, 13], N)
    x = torch.full((1, N), den.mask_id)
    for i, tok in reveal:
        x[0, i] = tok
    return den, RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)


def test_separator_priority_splits_the_longest_masked_stretch():
    N = 15
    kern = DistanceKernel(unscreened=[0.0] + [0.5 ** d for d in range(1, N)], screened=[0.0] * N)   # a Markov chain
    den, st = _state(N)
    r, valid = CouplingPolicy(kern, 0.0, priority="separator", mixed=False).reveal(st, 0, N)
    assert r[0][valid[0]].tolist() == [N // 2]                           # nothing revealed: the middle goes first, alone
    den, st = _state(N, reveal=[(7, 1)])
    r, valid = CouplingPolicy(kern, 0.0, priority="separator", mixed=False).reveal(st, 1, N)
    assert sorted(r[0][valid[0]].tolist()) == [3, 11]                    # then the middle of each half
    den, st = _state(N, reveal=[(7, 1), (3, 0), (11, 2)])
    r, valid = CouplingPolicy(kern, 0.0, priority="separator", mixed=False).reveal(st, 2, N)
    assert sorted(r[0][valid[0]].tolist()) == [1, 5, 9, 13]              # bisection: the set doubles each step


@pytest.mark.parametrize("priority", ["separator", "sum", "weighted"])
def test_new_priorities_on_both_kernel_paths_and_stay_exact(priority):
    N = 14
    den = noisy_hmm(N, 3, 0.8, 0.3, seed=1, concentration=2.0); cl = den.classes()
    route = TorchSubset(TorchChain(C.dfa_at_least_k(0, 3, 3)), [4, 5, 6, 7, 8, 9], N)
    kern, _ = fit_distance_kernel(den, N, 300, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(3)
    x = den.sample(40, g)
    x = torch.where(torch.rand(40, N, generator=g) < 0.4, x, torch.full_like(x, den.mask_id))
    st = RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)
    a = CouplingPolicy(kern, 0.05, priority=priority, mixed=False).reveal(st, 0, N)
    b = CouplingPolicy(_MatrixOnly(kern), 0.05, priority=priority, mixed=False).reveal(st, 0, N)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # exactness on a Markov chain with its true structure as the kernel: no gap, unbiased Z_hat
    N = 16
    den = lazy_chain(N, 3, 0.7, seed=7, concentration=2.0); cl = den.classes()
    d, pos = C.dfa_at_least_k(0, 4, 3), list(range(5, 11))
    act = np.zeros(N, dtype=bool); act[pos] = True
    truth = ConstrainedTruth(den, cl, d, active=act)
    kern = DistanceKernel(unscreened=[0.0] + [0.7 ** k for k in range(1, N)], screened=[0.0] * N)
    out = twisted_smc(den, cl, TorchSubset(TorchChain(d), pos, N), N, N, nrun=4000, K=1,
                      gen=torch.Generator().manual_seed(2), schedule=CouplingPolicy(kern, 0.0, priority=priority, mixed=False),
                      return_particles=True, path_stat=lambda a, b: parallel_log_prob(den, a, b))
    r = den.log_joint(torch.tensor(out["tokens"]).view(-1, N)).numpy() - out["path_stat"].reshape(-1)
    assert np.abs(r).max() < 1e-8 and out["steps"].mean() < 0.7 * N
    z = np.exp(out["logZ"]); se = z.std(ddof=1) / math.sqrt(len(z))
    assert abs(z.mean() - math.exp(truth.log_pC)) < 4 * se and out["frac_valid_finite"] == 1.0


def test_least_coupled_first_cannot_use_screening_but_separator_can():
    """Markov chain, theta = 0, the chain's true structure as the kernel. Least-coupled-first from the left-most
    position reveals one position per step (the left-most masked position never separates anything), so it needs
    N steps; the separator score bisects and needs about log2 N."""
    N = 15
    kern = DistanceKernel(unscreened=[0.0] + [0.5 ** d for d in range(1, N)], screened=[0.0] * N)
    den, st = _state(N)
    r, valid = CouplingPolicy(kern, 0.0, priority="least", mixed=False).reveal(st, 0, N)
    assert r[0][valid[0]].tolist() == [0]
    den, st = _state(N, reveal=[(0, 1)])
    r, valid = CouplingPolicy(kern, 0.0, priority="least", mixed=False).reveal(st, 1, N)
    assert r[0][valid[0]].tolist() == [1]
    den = lazy_chain(N, 3, 0.7, seed=7, concentration=2.0); cl = den.classes()
    route = TorchSubset(TorchChain(C.dfa_at_least_k(0, 2, 3)), [10, 11, 12, 13], N)
    steps = {}
    for pr in ("least", "separator"):
        out = twisted_smc(den, cl, route, N, N, nrun=50, K=1, gen=torch.Generator().manual_seed(0),
                          schedule=CouplingPolicy(kern, 0.0, priority=pr, mixed=False))
        steps[pr] = float(out["steps"].mean())
    assert steps["least"] == N and steps["separator"] <= 5


def test_least_coupled_first_uses_distance_and_budget_caps_the_total():
    N = 16
    kern = DistanceKernel(unscreened=[0.0, 0.4, 0.1, 0.02] + [0.0] * (N - 4))      # range 3, no screening
    den, st = _state(N)
    r, valid = CouplingPolicy(kern, 0.0, priority="least", mixed=False).reveal(st, 0, N)
    assert sorted(r[0][valid[0]].tolist()) == [0, 4, 8, 12]                 # every position more than 3 apart
    # a flat weak coupling of 0.01 between every pair: the per-candidate threshold 0.05 admits loads 0, 0.01, ..., 0.05
    # (six positions, total 0.15); a total budget of 0.05 admits loads 0, 0.01, 0.02 (three positions, total 0.03)
    flat = DistanceKernel(unscreened=[0.0] + [0.01] * (N - 1))
    for pr in ("least", "random"):
        r, valid = CouplingPolicy(flat, 0.05, priority=pr, mixed=False, forced_first=False).reveal(st, 0, N, torch.Generator().manual_seed(0))
        assert int(valid.sum()) == 6
        r, valid = CouplingPolicy(flat, 0.0, priority=pr, mixed=False, forced_first=False, budget=0.05).reveal(st, 0, N, torch.Generator().manual_seed(0))
        assert int(valid.sum()) == 3
    with pytest.raises(ValueError):
        CouplingPolicy(kern, 0.0, budget=-1.0)


@pytest.mark.parametrize("priority", ["least", "least_random"])
def test_dynamic_order_agrees_on_both_kernel_paths(priority):
    N = 14
    den = noisy_hmm(N, 3, 0.8, 0.3, seed=1, concentration=2.0); cl = den.classes()
    route = TorchSubset(TorchChain(C.dfa_at_least_k(0, 3, 3)), [4, 5, 6, 7, 8, 9], N)
    kern, _ = fit_distance_kernel(den, N, 300, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(3)
    x = den.sample(40, g)
    x = torch.where(torch.rand(40, N, generator=g) < 0.4, x, torch.full_like(x, den.mask_id))
    st = RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)
    for kw in (dict(theta=0.05), dict(theta=0.0, budget=0.1)):
        a = CouplingPolicy(kern, priority=priority, mixed=False, **kw).reveal(st, 0, N, torch.Generator().manual_seed(9))
        b = CouplingPolicy(_MatrixOnly(kern), priority=priority, mixed=False, **kw).reveal(st, 0, N, torch.Generator().manual_seed(9))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_graded_kernel_rows_and_apply_match_the_matrix():
    from twistroute.decode import COUNT_EDGES
    g = torch.Generator().manual_seed(2)
    P, N = 6, 40
    table = torch.rand(N, len(COUNT_EDGES) + 1, generator=g).double()
    k = DistanceKernel.from_table(table, COUNT_EDGES)
    masked = torch.rand(P, N, generator=g) < 0.5
    M, counts = k.matrix(masked), k.counts(masked)
    for c in (0, 7, 19, 39):
        assert torch.equal(k.row(counts, torch.full((P,), c)), M[:, c])
    v = torch.rand(P, N, generator=g).double() * masked
    assert torch.allclose(k.apply(masked, v), (M @ v[..., None]).squeeze(-1), atol=1e-12)
    # the entry is indexed by the number of revealed positions strictly in between
    m2 = torch.ones(1, N, dtype=torch.bool); m2[0, [3, 4, 5, 6, 7]] = False            # five revealed between 2 and 8
    assert float(k.matrix(m2)[0, 2, 8]) == float(table[6, 4]) and float(k.matrix(m2)[0, 8, 20]) == float(table[12, 0])
    with pytest.raises(ValueError):
        DistanceKernel.from_table(table, (1, 2))


def test_graded_pilot_kernel_sees_runs_screen_more_than_single_tokens():
    from twistroute.decode import COUNT_EDGES
    N = 24
    hmm = noisy_hmm(N, 3, 0.85, 0.3, seed=1, concentration=2.0)
    k, cnt = fit_distance_kernel(hmm, N, 3000, torch.Generator().manual_seed(0), count_edges=COUNT_EDGES)
    t = k.table.numpy()
    assert t.shape == (N, len(COUNT_EDGES) + 1)
    for d in (4, 6):                                                      # coupling falls as more is revealed in between
        assert t[d, 0] > t[d, 1] > t[d, 2] > t[d, 3] > 0
    b, _ = fit_distance_kernel(hmm, N, 3000, torch.Generator().manual_seed(0))          # the two-bin kernel, same pilot
    assert b.table.shape == (N, 2) and np.allclose(b.ku.numpy(), t[:, 0])
    assert t[4, 1] > b.ks.numpy()[4] > t[4, 3]                            # "at least one" pools the graded bins
    chain = lazy_chain(N, 3, 0.85)
    k, _ = fit_distance_kernel(chain, N, 500, torch.Generator().manual_seed(0), count_edges=COUNT_EDGES)
    assert float(k.table[:13, 1:].abs().max()) < 1e-12                    # a Markov chain is screened by one token
    assert float(k.table[1, 1:].abs().max()) == 0 and float(k.table[2, 2:].abs().max()) == 0     # bins that cannot occur


def test_rate_bounds_the_load_of_a_whole_trajectory():
    """rate = B / N: the average load per position in every step is <= rate, so a full decode spends <= B."""
    N, B = 24, 0.3
    den = noisy_hmm(N, 3, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    route = TorchSubset(TorchChain(C.dfa_at_least_k(0, 3, 3)), [8, 9, 10, 11, 12, 13], N)
    kern, _ = fit_distance_kernel(den, N, 500, torch.Generator().manual_seed(0))
    steps = {}
    for b in (0.0, B, 10 * B):
        pol = CouplingPolicy(kern, 0.0, priority="separator", mixed=False, rate=b / N)
        out = twisted_smc(den, cl, route, N, N, nrun=100, K=2, gen=torch.Generator().manual_seed(1), schedule=pol)
        assert pol.load_sum / 200 <= b + 1e-9                             # mean total load per particle path
        steps[b] = float(out["steps"].mean())
    assert steps[0.0] > steps[B] > steps[10 * B]                          # a larger budget buys fewer steps
    x = torch.full((1, N), den.mask_id)
    st = RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)
    flat = DistanceKernel(unscreened=[0.0] + [0.01] * (N - 1))            # loads 0, 0.01, 0.02, ...: average of n is 0.005 (n - 1)
    r, valid = CouplingPolicy(flat, 0.0, priority="random", mixed=False, forced_first=False, rate=0.021).reveal(st, 0, N, torch.Generator().manual_seed(0))
    assert int(valid.sum()) == 5                                          # 0.005 (n - 1) <= 0.021  <=>  n <= 5
    with pytest.raises(ValueError):
        CouplingPolicy(flat, 0.0, rate=-1.0)


def test_dilated_schedule_is_coarse_to_fine():
    from twistroute.decode import DilatedSchedule
    N = 8
    den = lazy_chain(N, 3, 0.6, seed=7, concentration=2.0); cl = den.classes()
    d = C.dfa_exactly_k(0, 3, 3); route = torch_route(d)
    x = torch.full((2, N), den.mask_id)
    sets = []
    for t in range(3):
        st = RevealState(x, cl.token_class[x], cl.class_logw(den.logits(x)), den.mask_id, cl, route)
        r, valid = DilatedSchedule(2).reveal(st, t, N)
        sets.append(sorted(r[0][valid[0]].tolist()))
        x = x.clone(); x[:, r[0][valid[0]]] = 0
    assert sets == [[0, 4], [2, 6], [1, 3, 5, 7]]                         # DUS eqs. 8-9 for a block of 8, zero-indexed
    out = twisted_smc(den, cl, route, N, N, nrun=300, K=2, gen=torch.Generator().manual_seed(0), schedule=DilatedSchedule(2))
    assert (out["steps"] == 3).all() and out["frac_valid_finite"] == 1.0
    out = twisted_smc(den, cl, route, N, N, nrun=50, K=2, gen=torch.Generator().manual_seed(0), schedule=DilatedSchedule(4))
    assert (out["steps"] == 2).all()                                      # base 4: strides 2 then 1
    with pytest.raises(ValueError):
        DilatedSchedule(1)
