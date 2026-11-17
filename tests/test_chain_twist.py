"""The first-order chain twist (decode/chain_twist.py).

* its value and its constrained marginals against brute-force enumeration of the chain law, with some positions
  revealed, on the whole sequence and on a subset of positions;
* against the reference implementation used in the twist-error study (experiments/parallel_unmasking/twist_error.py);
* with eps = 1 it is the product twist;
* in the sampler: without resampling the weights are those of the product twist (the intermediate twists
  telescope); with resampling Z_hat stays unbiased for the exact p(C).
"""
import itertools, math, os, sys
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.decode import twisted_smc, ChainTwist, fit_pair_kernel
from twistroute.decode.schedule import clamp_revealed
from twistroute.models import ConstrainedTruth, noisy_hmm, lazy_chain
from twistroute.torch_routes import torch_route, TorchChain, TorchSubset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "parallel_unmasking"))


def state(den, cl, n_rows, frac, seed):
    """Partly revealed states drawn from the model, and their clamped class log-weights."""
    g = torch.Generator().manual_seed(seed)
    x = den.sample(n_rows, g)
    x[torch.rand(x.shape, generator=g) > frac] = den.mask_id
    y = cl.token_class[torch.where(x == den.mask_id, torch.zeros_like(x), x)]
    return x, clamp_revealed(cl.class_logw(den.logits(x)), x, y, den.mask_id)


def enumerate_chain(tw, lwc, accepts):
    """log P_chain(C) and the constrained marginals of the chain law for one state, by enumeration."""
    mu = torch.exp(lwc.double()); mu = mu / mu.sum(-1, keepdim=True)
    Q = tw.transitions(mu[None])[0]
    N, A = mu.shape
    ys = torch.tensor(list(itertools.product(range(A), repeat=N)))
    p = mu[0, ys[:, 0]]
    for i in range(N - 1):
        p = p * Q[i, ys[:, i], ys[:, i + 1]]
    assert abs(float(p.sum()) - 1.0) < 1e-9                                # the chain law is normalised
    p = p * accepts(ys)
    if float(p.sum()) == 0.0:                                              # the revealed tokens already break C
        return -math.inf, np.zeros((N, A))
    marg = torch.stack([torch.stack([p[ys[:, i] == a].sum() for a in range(A)]) for i in range(N)]) / p.sum()
    return math.log(float(p.sum())), marg.numpy()


@pytest.mark.parametrize("case", ["forbidden", "exactly_k", "window"])
def test_value_and_marginals_match_enumeration(case):
    N, A = 7, 3
    den = noisy_hmm(N, A, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    K = fit_pair_kernel(den, cl, 4000, torch.Generator().manual_seed(0))
    if case == "window":
        act = np.zeros(N, dtype=bool); act[[1, 2, 4, 5]] = True
        d = C.dfa_at_least_k(0, 2, A); route = TorchSubset(TorchChain(d), [1, 2, 4, 5], N)
        tw = ChainTwist(d, K, active=act)
    else:
        d = C.dfa_forbidden([(0, 0), (1, 2)], A) if case == "forbidden" else C.dfa_exactly_k(0, 2, A)
        route = torch_route(d); tw = ChainTwist(d, K)
    x, lwc = state(den, cl, 12, 0.3, seed=3)
    val, mg = tw.log_chain(lwc), tw.marginals(lwc)
    alive = 0
    for b in range(len(x)):
        ref, ref_m = enumerate_chain(tw, lwc[b], route.accepts)
        if ref == -math.inf:
            assert float(val[b]) == -math.inf and float(mg[b].abs().max()) == 0.0
            continue
        alive += 1
        assert abs(float(val[b]) - ref) < 1e-9 and np.abs(mg[b].numpy() - ref_m).max() < 1e-8
    assert alive >= 3
    assert torch.allclose(tw.log_psi(lwc, route.logZ(lwc)), torch.logaddexp(val + math.log(0.95), route.logZ(lwc) + math.log(0.05)))
    assert torch.equal(ChainTwist(d, K, eps=1.0, active=tw.active).log_psi(lwc, route.logZ(lwc)), route.logZ(lwc))


def test_matches_the_reference_of_the_twist_error_study():
    import twist_error as te
    N, A = 20, 3
    den = noisy_hmm(N, A, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    d = C.dfa_exactly_k(0, N // 3, A)
    ref = te.Twists(den, d, np.ones(N, dtype=bool))
    K = ref.stationary_kernel(5000, seed=0)
    x, lwc = state(den, cl, 40, 0.25, seed=5)
    mu, rev = ref._marg(x)
    want = ref.chain_from(mu, ref.ratio_from(mu, rev, K))
    got = ChainTwist(d, K).log_chain(lwc)
    ok = torch.isfinite(want)
    assert ok.sum() > 10 and torch.equal(torch.isfinite(got), ok)
    assert (got[ok] - want[ok]).abs().max() < 1e-6                         # the sampler's weights are float32


@pytest.mark.parametrize("cname", ["counting", "forbidden"])
def test_in_the_sampler(cname):
    N, A = 24, 3
    den = noisy_hmm(N, A, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    d = C.dfa_exactly_k(0, N // 3, A) if cname == "counting" else C.dfa_forbidden([(0, 0), (1, 2)], A)
    route, truth = torch_route(d), ConstrainedTruth(den, cl, d)
    tw = ChainTwist(d, fit_pair_kernel(den, cl, 5000, torch.Generator().manual_seed(0)))
    run = lambda twist, ess: twisted_smc(den, cl, route, N, N, nrun=300, K=8, gen=torch.Generator().manual_seed(7),
                                         ess_min=ess, twist=twist, return_particles=True)
    a, b = run(None, 0.0), run(tw, 0.0)                                    # no resampling: the same weights
    assert np.array_equal(a["classes"], b["classes"]) and np.abs(a["logZ"] - b["logZ"]).max() < 1e-4
    for twist in (None, tw):                                               # resampling at every step: unbiased
        out = run(twist, 1.0)
        z = np.exp(out["logZ"] - truth.log_pC); se = z.std(ddof=1) / math.sqrt(len(z))
        assert out["n_res"].mean() > 5 and out["frac_valid_finite"] == 1.0
        assert abs(z.mean() - 1.0) < 4 * se


@pytest.mark.parametrize("temp", [0.7, 0.5])
def test_calibrated_chain_twist(temp):
    """temp < 1 builds the chain on flattened marginals. It changes the intermediate twist, leaves a revealed
    (one-hot) position alone, gives the same weights without resampling, and keeps Z_hat unbiased with it."""
    N, A = 24, 3
    den = noisy_hmm(N, A, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    d = C.dfa_forbidden([(0, 0), (1, 2)], A)
    route, truth = torch_route(d), ConstrainedTruth(den, cl, d)
    K = fit_pair_kernel(den, cl, 5000, torch.Generator().manual_seed(0))
    plain, cal = ChainTwist(d, K), ChainTwist(d, K, temp=temp)
    assert ChainTwist(d, K, temp=1.0).temp == 1.0
    with pytest.raises(ValueError):
        ChainTwist(d, K, temp=0.0)
    x = torch.full((4, N), den.mask_id); lw = cl.class_logw(den.logits(x)); lp = route.logZ(lw).double()
    assert (cal.log_psi(lw, lp) - plain.log_psi(lw, lp)).abs().min() > 1e-6          # the parameter is live
    y = torch.tensor(truth.sample(4, np.random.default_rng(0)))                      # fully revealed: one-hot rows
    onehot = torch.log(torch.nn.functional.one_hot(y, A).double())
    lp1 = route.logZ(onehot.float()).double()
    assert torch.allclose(cal.log_psi(onehot, lp1), plain.log_psi(onehot, lp1), atol=1e-9)
    run = lambda twist, ess: twisted_smc(den, cl, route, N, N, nrun=300, K=8, gen=torch.Generator().manual_seed(7),
                                         ess_min=ess, twist=twist, return_particles=True)
    a, b = run(None, 0.0), run(cal, 0.0)                                   # no resampling: the same weights
    assert np.array_equal(a["classes"], b["classes"]) and np.abs(a["logZ"] - b["logZ"]).max() < 1e-4
    out = run(cal, 1.0)                                                    # resampling at every step: unbiased
    z = np.exp(out["logZ"] - truth.log_pC); se = z.std(ddof=1) / math.sqrt(len(z))
    assert out["n_res"].mean() > 5 and out["frac_valid_finite"] == 1.0
    assert abs(z.mean() - 1.0) < 4 * se


@pytest.mark.parametrize("case", ["forbidden", "exactly_k", "window"])
def test_sample_from_the_chain_law(case):
    """`sample` draws from the chain law given the constraint: its normaliser is `log_chain`, its class frequencies
    are `marginals`, every draw satisfies the constraint, and the clamped masses sum to the normaliser."""
    N, A = 7, 3
    den = noisy_hmm(N, A, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    K = fit_pair_kernel(den, cl, 4000, torch.Generator().manual_seed(0))
    if case == "window":
        act = np.zeros(N, dtype=bool); act[[1, 2, 4, 5]] = True
        d = C.dfa_at_least_k(0, 2, A); route = TorchSubset(TorchChain(d), [1, 2, 4, 5], N); tw = ChainTwist(d, K, active=act)
    else:
        d = C.dfa_forbidden([(0, 0), (1, 2)], A) if case == "forbidden" else C.dfa_exactly_k(0, 2, A)
        route = torch_route(d); tw = ChainTwist(d, K)
    x, lwc = state(den, cl, 12, 0.3, seed=3)
    val = tw.log_chain(lwc); ok = torch.isfinite(val); assert ok.sum() >= 3
    lwc = lwc[ok]; val = val[ok]; mg = tw.marginals(lwc); B = len(lwc); n = 20000
    y, lc = tw.sample(lwc.repeat_interleave(n, 0), torch.Generator().manual_seed(1), chunk=7000)
    assert (lc.view(B, n)[:, 0] - val).abs().max() < 1e-9 and bool(route.accepts(y).all())
    freq = torch.nn.functional.one_hot(y, A).double().view(B, n, N, A).mean(1)
    se = torch.sqrt((mg * (1 - mg)).clamp_min(0.0) / n) + 1e-4
    assert ((freq - mg).abs() < 4.5 * se).all()
    r = torch.tensor([[2, 5]]).expand(B, 2)                                 # masses over the classes at two positions sum to P_chain(C)
    tot = torch.stack([tw.log_clamped(lwc, r, torch.tensor([[a, b]]).expand(B, 2)) for a in range(A) for b in range(A)]).logsumexp(0)
    assert (tot - val).abs().max() < 1e-9
    one = tw.log_clamped(lwc, r, torch.zeros(B, 2, dtype=torch.long), rvalid=torch.tensor([[True, False]]).expand(B, 2))
    pos = mg[:, 2, 0] > 1e-9                                               # a padded column does not count
    assert pos.any() and torch.allclose(one[pos], (val + torch.log(mg[:, 2, 0]))[pos], atol=1e-8) and bool(torch.isneginf(one[~pos]).all())


@pytest.mark.parametrize("cname", ["counting", "forbidden"])
@pytest.mark.parametrize("T", [24, 6])
def test_chain_proposal_in_the_sampler(cname, T):
    """Proposing from the chain law does not change what Z_hat estimates: p(C) with one token per step (T = N), and
    with several positions per step the same value as the product proposal (the probability of C under the parallel
    reveal process, which is not p(C)). Checked without resampling and with it, alone and with the chain twist as
    the intermediate twist; and a single particle is closer to the constrained law than under the product proposal."""
    N, A = 24, 3
    den = noisy_hmm(N, A, 0.85, 0.3, seed=1, concentration=2.0); cl = den.classes()
    d = C.dfa_exactly_k(0, N // 3, A) if cname == "counting" else C.dfa_forbidden([(0, 0), (1, 2)], A)
    route, truth = torch_route(d), ConstrainedTruth(den, cl, d)
    tw = ChainTwist(d, fit_pair_kernel(den, cl, 5000, torch.Generator().manual_seed(0)))
    ref, ref_se = 1.0, 0.0
    if T < N:                                                              # the value the product proposal estimates
        zr = np.exp(twisted_smc(den, cl, route, N, T, nrun=3000, K=8, gen=torch.Generator().manual_seed(11), ess_min=0.0)["logZ"] - truth.log_pC)
        ref, ref_se = zr.mean(), zr.std(ddof=1) / math.sqrt(len(zr))
        assert abs(ref - 1.0) > 0.1                                        # and it is not p(C)
    for kw in (dict(ess_min=0.0), dict(ess_min=1.0), dict(ess_min=1.0, twist=tw), dict(ess_min=0.0, mix_native=0.1)):
        out = twisted_smc(den, cl, route, N, T, nrun=400, K=8, gen=torch.Generator().manual_seed(7), proposal_twist=tw,
                          return_particles=True, **kw)
        z = np.exp(out["logZ"] - truth.log_pC); se = z.std(ddof=1) / math.sqrt(len(z))
        assert out["frac_valid_finite"] == 1.0 and abs(z.mean() - ref) < 4 * math.hypot(se, ref_se), (kw.keys(), z.mean(), se, ref)
        if kw.get("mix_native", 0) == 0:
            assert out["guard"]["proposal_off_support"] == 0 and out["guard"]["nan_weight"] == 0
    with pytest.raises(ValueError):
        twisted_smc(den, cl, route, N, T, nrun=2, K=2, proposal_twist=tw, twist_temp=0.7)
    if cname == "forbidden" and T == N:                                    # KL of one particle's law from the constrained law
        kl = lambda pt: -float(np.mean(twisted_smc(den, cl, route, N, N, nrun=4000, K=1, gen=torch.Generator().manual_seed(3),
                                                   ess_min=0.0, proposal_twist=pt)["logZ"] - truth.log_pC))
        assert kl(tw) < 0.6 * kl(None)

