"""Resampling primitives and the SMC options added for the resampling investigation (notes Sec. 7).

* every resampler draws ancestors with the requested probabilities; systematic resampling gives offspring
  counts within one of K * pi;
* a defensive resampling step preserves the weighted particle measure exactly (the identity in the proof of
  Thm. 6.2(iii)), and offspring weights are bounded by 1 / (1 - eps);
* the per-step weight trace agrees with Z_hat, and parameter validation rejects out-of-range options.
"""
import math
import numpy as np
import pytest
import torch
from twistroute.decode import twisted_smc
from twistroute.decode.smc import resample_indices, defensive_mixture, offspring_logweights
from twistroute.models import SyntheticMDLM
from twistroute.torch_routes import build_route

PROBS = torch.tensor([0.55, 0.30, 0.10, 0.05])


@pytest.mark.parametrize("resampler", ["multinomial", "cpu", "systematic"])
def test_ancestor_frequencies_match_probs(resampler):
    rows, K = 20_000, 4
    probs = PROBS.expand(rows, K).contiguous()
    idx = resample_indices(probs, K, resampler, torch.Generator().manual_seed(1), torch.Generator().manual_seed(2))
    assert idx.shape == (rows, K) and idx.min() >= 0 and idx.max() < K
    freq = torch.bincount(idx.reshape(-1), minlength=K).double() / idx.numel()
    se = torch.sqrt(PROBS.double() * (1 - PROBS.double()) / idx.numel())
    assert torch.all((freq - PROBS.double()).abs() < 4 * se + 1e-12), freq


def test_systematic_counts_are_floor_or_ceil():
    rng = np.random.default_rng(0)
    p = torch.tensor(rng.dirichlet(np.ones(8), 500), dtype=torch.float32)
    K = 8
    idx = resample_indices(p, K, "systematic", torch.Generator().manual_seed(3))
    counts = torch.zeros(500, K).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
    target = K * p
    assert torch.all(counts >= torch.floor(target) - 1e-6) and torch.all(counts <= torch.ceil(target) + 1e-6)


def test_unknown_resampler_rejected():
    with pytest.raises(ValueError):
        resample_indices(PROBS[None], 4, "stratified")


@pytest.mark.parametrize("eps", [0.0, 0.25, 0.6])
def test_defensive_step_preserves_weighted_measure(eps):
    """sum_a pi_a * (wn_a / pi_a) * f_a = sum_a wn_a f_a for every ancestor distribution pi with full support
    where wn > 0: the reweighted resampling step is conditionally unbiased."""
    rng = np.random.default_rng(4)
    wn = torch.tensor(rng.dirichlet(np.ones(5) * 0.3, 50), dtype=torch.float32)
    f = torch.tensor(rng.normal(size=(50, 5)), dtype=torch.float32)
    pi = defensive_mixture(wn, eps)
    assert torch.allclose(pi.sum(-1), torch.ones(50))
    all_idx = torch.arange(5).expand(50, 5)                               # every ancestor once
    w_off = torch.exp(offspring_logweights(wn, pi, all_idx, eps))
    if eps == 0:
        assert torch.all(w_off == 1)                                      # plain resampling resets weights
        return
    lhs = (pi * w_off * f).sum(-1)
    assert torch.allclose(lhs, (wn * f).sum(-1), atol=1e-6)
    assert torch.all(w_off <= 1 / (1 - eps) + 1e-6)                      # notes Prop. 7.6(ii)


def _setup(N=8):
    den = SyntheticMDLM(N, scale=4.0)
    spec = dict(kind="cfg", grammar="dyck1_amb", A=2, N=N)
    return den, den.classes(4).merge([0, 1, 0, 1]), build_route(spec)


def test_trace_last_row_gives_logZ_without_resampling():
    den, cl, route = _setup()
    out = twisted_smc(den, cl, route, 8, 4, 300, 4, torch.Generator().manual_seed(0), ess_min=0.0, return_trace=True)
    tr = out["trace"]
    assert tr.shape == (4, 300, 4)
    last = torch.tensor(tr[-1], dtype=torch.float64)
    lme = torch.logsumexp(last, -1) - math.log(4)
    fin = torch.isfinite(lme)
    assert np.allclose(lme[fin].numpy(), out["logZ"][fin.numpy()], atol=1e-5)


def test_trace_does_not_change_the_run():
    den, cl, route = _setup()
    a = twisted_smc(den, cl, route, 8, 4, 200, 4, torch.Generator().manual_seed(7))
    b = twisted_smc(den, cl, route, 8, 4, 200, 4, torch.Generator().manual_seed(7), return_trace=True)
    assert np.array_equal(a["logZ"], b["logZ"])


@pytest.mark.parametrize("bad", [dict(defensive=1.0), dict(defensive=-0.1), dict(twist_power=0.0), dict(twist_power=1.5)])
def test_invalid_options_rejected(bad):
    den, cl, route = _setup()
    with pytest.raises(ValueError):
        twisted_smc(den, cl, route, 8, 4, 2, 4, torch.Generator().manual_seed(0), **bad)


def test_defensive_runs_keep_low_weight_particles():
    """With eps > 0 every resampling keeps each particle with probability >= 1 - (1 - eps/K)^K, so runs that
    resample retain more distinct ancestors on average than with plain resampling."""
    den, cl, route = _setup()
    kw = dict(ess_min=0.95, return_particles=True)
    plain = twisted_smc(den, cl, route, 8, 4, 2000, 4, torch.Generator().manual_seed(1), **kw)
    dfn = twisted_smc(den, cl, route, 8, 4, 2000, 4, torch.Generator().manual_seed(1), defensive=0.5, **kw)
    distinct = lambda o: np.mean([len({tuple(r) for r in run}) for run in o["classes"]])
    assert dfn["n_res"].mean() > 0.5 and distinct(dfn) > distinct(plain)


@pytest.mark.parametrize("opts", [dict(lookahead=1), dict(lookahead=2, lookahead_draws=3), dict(lookahead=8),
                                  dict(mix_native=0.3), dict(lookahead=2, mix_native=0.2, defensive=0.25),
                                  dict(lookahead=2, lookahead_draws=3, lookahead_mean="geom"),
                                  dict(lookahead=1, lookahead_shrink=0.5)])
def test_lookahead_and_native_mixture_unbiased_and_valid(opts):
    """Look-ahead twists (random, positive, reused consistently) and the native-kernel mixture proposal keep
    E[Z_hat] = h0 (notes, free intermediate twists), and every finite-weight particle is valid."""
    from twistroute.decode import native_sample
    den, cl, route = _setup()
    g, hits, n = torch.Generator().manual_seed(77), 0, 200_000
    for _ in range(n // 20000):
        hits += int(route.accepts(native_sample(den, cl, 8, 4, 20000, g)).sum())
    h0, h0se = hits / n, math.sqrt(hits) / n
    out = twisted_smc(den, cl, route, 8, 4, 6000, 4, torch.Generator().manual_seed(5), ess_min=0.9, **opts)
    z = np.exp(out["logZ"])
    zscore = (z.mean() - h0) / math.sqrt(z.var() / len(z) + h0se ** 2)
    assert abs(zscore) < 4, f"{opts}: mean {z.mean():.4f} vs h0 {h0:.4f} (z={zscore:.2f})"
    assert not out["any_finite_invalid"] and np.isfinite(out["logZ"]).any()


def test_lookahead_rejects_bad_options():
    den, cl, route = _setup()
    for bad in (dict(lookahead=-1), dict(lookahead=1, lookahead_draws=0), dict(mix_native=1.0),
                dict(lookahead=1, lookahead_mean="median"), dict(lookahead=1, lookahead_shrink=1.5)):
        with pytest.raises(ValueError):
            twisted_smc(den, cl, route, 8, 4, 2, 4, torch.Generator().manual_seed(0), **bad)
