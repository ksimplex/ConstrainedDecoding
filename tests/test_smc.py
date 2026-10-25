"""Batched twisted SMC on a synthetic denoiser with strong re-conditioning.

* unbiasedness: mean Z_hat over runs matches h0 from native sampling (|z| < 4) at K = 1 and 4, with
  resampling active;
* two-sided identity: along native paths E_p[1_C / w] = 1;
* golden values: fixed-seed outputs of the configuration that produced E2 (regression guard for refactors).
"""
import math
import numpy as np
import pytest
import torch
from twistroute.decode import twisted_smc, native_sample, native_path_weights, select_particles
from twistroute.models import SyntheticMDLM
from twistroute.torch_routes import build_route

N, T = 8, 4
SPECS = {
    "forbidden": dict(kind="forbidden", A=4, map=[0, 1, 2, 3], patterns=[[3, 3], [1, 1]], N=N),
    "exactly_k": dict(kind="exactly_k", A=2, map=[0, 1, 0, 1], cls=0, count=2, N=N),
    "copy": dict(kind="copy", A=2, map=[0, 0, 1, 1], pairs=4, N=N),
    "dyck1_amb": dict(kind="cfg", grammar="dyck1_amb", A=2, map=[0, 1, 0, 1], N=N),
    "rna_pair": dict(kind="cfg", grammar="rna_pair", A=4, map=[0, 1, 2, 3], N=N),
}


def setup(name, scale=4.0):
    spec = SPECS[name]
    den = SyntheticMDLM(N, scale=scale)
    return den, den.classes(4).merge(spec["map"]), build_route(spec)


def h0_native(den, cl, route, n=200_000):
    hits = 0
    g = torch.Generator().manual_seed(77)
    for _ in range(n // 20000):
        hits += int(route.accepts(native_sample(den, cl, N, T, 20000, g)).sum())
    return hits / n, math.sqrt(hits) / n


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("K", [1, 4])
def test_zhat_unbiased(name, K):
    den, cl, route = setup(name)
    h0, h0se = h0_native(den, cl, route)
    out = twisted_smc(den, cl, route, N, T, 6000, K, torch.Generator().manual_seed(5), ess_min=0.9)
    z = np.exp(out["logZ"])
    zscore = (z.mean() - h0) / math.sqrt(z.var() / len(z) + h0se ** 2)
    assert abs(zscore) < 4, f"{name} K={K}: mean {z.mean():.4f} vs h0 {h0:.4f} (z={zscore:.2f})"
    assert out["frac_valid_finite"] == 1.0 and not out["any_finite_invalid"]
    if K > 1:
        assert out["n_res"].mean() > 0.1, "resampling path not exercised"


@pytest.mark.parametrize("name", ["exactly_k", "dyck1_amb"])
def test_two_sided_identity(name):
    den, cl, route = setup(name)
    lw, valid = native_path_weights(den, cl, route, N, T, 100_000, torch.Generator().manual_seed(9))
    inv = np.where(valid, np.exp(-np.where(valid, lw, 0.0)), 0.0)
    assert np.isfinite(lw[valid]).all() and not np.isfinite(lw[~valid]).any()
    assert abs(inv.mean() - 1.0) < 4 * inv.std() / math.sqrt(len(inv))


def test_selected_particles_are_valid():
    den, cl, route = setup("dyck1_amb")
    out = twisted_smc(den, cl, route, N, T, 200, 4, torch.Generator().manual_seed(3), return_particles=True)
    S = select_particles(out, np.random.default_rng(0))
    assert S.shape == (200, N) and route.accepts(torch.tensor(S)).all()


# Golden values: mean Z_hat with seed 11, 3000 runs, synthetic scale 4. Produced by the pre-refactor E2 code
# (bit-identical) — a change here means the sampler's behaviour changed, not just its statistics.
GOLDEN = {("forbidden", 1): 0.53414, ("forbidden", 4): 0.53687, ("exactly_k", 1): 0.07142, ("exactly_k", 4): 0.07310,
          ("copy", 1): 0.07179, ("copy", 4): 0.07354, ("dyck1_amb", 1): 0.07562, ("dyck1_amb", 4): 0.07675,
          ("rna_pair", 1): 0.16138, ("rna_pair", 4): 0.16225}


@pytest.mark.parametrize("key", list(GOLDEN))
def test_golden(key):
    name, K = key
    den, cl, route = setup(name)
    out = twisted_smc(den, cl, route, N, T, 3000, K, torch.Generator().manual_seed(11))
    assert abs(np.exp(out["logZ"]).mean() - GOLDEN[key]) < 5e-5


@pytest.mark.parametrize("name", ["forbidden", "dyck1_amb", "rna_pair"])
@pytest.mark.parametrize("opts", [dict(defensive=0.3), dict(twist_power=0.5), dict(defensive=0.3, twist_power=0.5),
                                  dict(resampler="systematic"), dict(resampler="cpu")])
def test_zhat_unbiased_remedies(name, opts):
    """Defensive resampling, tempered intermediate twists and the alternative resamplers keep E[Z_hat] = h0."""
    den, cl, route = setup(name)
    h0, h0se = h0_native(den, cl, route)
    out = twisted_smc(den, cl, route, N, T, 6000, 4, torch.Generator().manual_seed(5), ess_min=0.9,
                      res_gen=torch.Generator().manual_seed(6), **opts)
    z = np.exp(out["logZ"])
    zscore = (z.mean() - h0) / math.sqrt(z.var() / len(z) + h0se ** 2)
    assert abs(zscore) < 4, f"{name} {opts}: mean {z.mean():.4f} vs h0 {h0:.4f} (z={zscore:.2f})"
    assert out["n_res"].mean() > 0.1 and out["frac_valid_finite"] == 1.0


def test_defaults_unchanged():
    """defensive = 0 and twist_power = 1 give bit-identical output to the plain call."""
    den, cl, route = setup("dyck1_amb")
    a = twisted_smc(den, cl, route, N, T, 500, 4, torch.Generator().manual_seed(3), ess_min=0.9)
    b = twisted_smc(den, cl, route, N, T, 500, 4, torch.Generator().manual_seed(3), ess_min=0.9,
                    defensive=0.0, twist_power=1.0)
    assert np.array_equal(a["logZ"], b["logZ"])
