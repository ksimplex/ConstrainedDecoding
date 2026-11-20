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
# non-crossing pairs (0,7), (1,6), (3,4) with a compatibility table that is not the identity
_COMPAT = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
PAIR_FACTORS = [dict(scope=[i, j], table=_COMPAT) for i, j in [(0, 7), (1, 6), (3, 4)]]
SPECS = {
    "forbidden": dict(kind="forbidden", A=4, map=[0, 1, 2, 3], patterns=[[3, 3], [1, 1]], N=N),
    "exactly_k": dict(kind="exactly_k", A=2, map=[0, 1, 0, 1], cls=0, count=2, N=N),
    "copy": dict(kind="copy", A=2, map=[0, 0, 1, 1], pairs=4, N=N),
    "dyck1_amb": dict(kind="cfg", grammar="dyck1_amb", A=2, map=[0, 1, 0, 1], N=N),
    "rna_pair": dict(kind="cfg", grammar="rna_pair", A=4, map=[0, 1, 2, 3], N=N),
    "pairs_fg": dict(kind="factor_graph", A=4, map=[0, 1, 2, 3], N=N, factors=PAIR_FACTORS),
    "motif_and_pairs": dict(kind="and", A=4, map=[0, 1, 2, 3], N=N, parts=[
        dict(kind="forbidden", A=4, N=N, patterns=[[0, 0]]),
        dict(kind="factor_graph", A=4, N=N, factors=PAIR_FACTORS)]),
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

@pytest.mark.parametrize("name", ["forbidden", "copy", "dyck1_amb", "rna_pair", "motif_and_pairs"])
@pytest.mark.parametrize("tau", [0.5, 0.25])
@pytest.mark.parametrize("ess_min", [0.0, 0.9])
def test_zhat_unbiased_calibrated_twist(name, tau, ess_min):
    """The calibrated twist (route sum under q ∝ p^tau) with its locally optimal proposal keeps E[Z_hat] = h0,
    with and without resampling, and never weights an invalid output."""
    den, cl, route = setup(name)
    h0, h0se = h0_native(den, cl, route)
    out = twisted_smc(den, cl, route, N, T, 6000, 4, torch.Generator().manual_seed(9), ess_min=ess_min,
                      twist_temp=tau)
    z = np.exp(out["logZ"])
    zscore = (z.mean() - h0) / math.sqrt(z.var() / len(z) + h0se ** 2)
    assert abs(zscore) < 4, f"{name} tau={tau}: mean {z.mean():.4f} vs h0 {h0:.4f} (z={zscore:.2f})"
    assert not out["any_finite_invalid"]


def test_support_guard_clean_on_valid_runs():
    """Every invariant holds on ordinary runs, for each resampling mode and route kind."""
    for name in ("forbidden", "copy", "dyck1_amb", "rna_pair"):
        den, cl, route = setup(name)
        out = twisted_smc(den, cl, route, N, T, 300, 4, torch.Generator().manual_seed(2), ess_min=0.9,
                          defensive=0.25, guard="raise")
        assert all(v == 0 for v in out["guard"].values()), (name, out["guard"])


def test_support_guard_detects_twist_collapse():
    """A route whose mass pass loses support after the first step is reported (and raises when asked)."""
    den, cl, route = setup("forbidden")

    class Leaky:
        def __init__(self, r): self.r, self.calls = r, 0
        def sample(self, lw, gen=None): return self.r.sample(lw, gen)
        def accepts(self, y): return self.r.accepts(y)
        def logZ(self, lw):
            z = self.r.logZ(lw)
            self.calls += 1
            if self.calls == 5:                     # the successor-twist pass of step 1 returns zero mass
                z = torch.full_like(z, float("-inf"))
            return z

    out = twisted_smc(den, cl, Leaky(route), N, T, 50, 2, torch.Generator().manual_seed(3), ess_min=0.0,
                      guard="off")
    assert out["guard"] == {k: 0 for k in out["guard"]}          # off: nothing counted
    with pytest.warns(RuntimeWarning, match="twist_collapse"):
        out = twisted_smc(den, cl, Leaky(route), N, T, 50, 2, torch.Generator().manual_seed(3), ess_min=0.0)
    assert out["guard"]["twist_collapse"] > 0
    with pytest.raises(FloatingPointError):
        twisted_smc(den, cl, Leaky(route), N, T, 50, 2, torch.Generator().manual_seed(3), ess_min=0.0,
                    guard="raise")


def test_class_weights_exact_below_float_floor():
    """Class masses far below 1e-38 are exact (log-sum-exp), not floored; ordinary entries are unchanged."""
    from twistroute.models import TokenClasses
    cls = TokenClasses(torch.tensor([0, 0, 1, 1, 2]), 3, excluded=(4,))
    lp = torch.log_softmax(torch.tensor([[[0.0, 0.0, -150.0, -151.0, 0.0]]]), -1)
    got = cls.class_logw(lp)[0, 0]
    ref1 = torch.logsumexp(lp[0, 0, 2:4], 0)
    assert torch.isclose(got[1], ref1, atol=1e-4) and float(got[1]) < -140     # floor would give ~-87.5
    assert torch.isneginf(got[2])                                              # class 2 has only the excluded token
    plain = torch.log((torch.exp(lp) @ cls.onehot).clamp_min(1e-38))[0, 0]
    assert got[0] == plain[0]


def test_independent_schedule_reveals_everything_and_matches_mdlm_rates():
    """Each masked position reveals with probability 1/(T - t); at the last step every row finishes."""
    from twistroute.decode import reveal_indices
    T, N, B = 8, 16, 4000
    g = torch.Generator().manual_seed(4)
    x = torch.full((B, N), 1, dtype=torch.long)                    # 1 = mask id here
    masked = torch.ones(B, N, dtype=torch.bool)
    counts = []
    for t_ in range(T):
        r, valid = reveal_indices(masked, t_, T, "independent", g)
        hit = torch.zeros_like(masked).scatter_(1, r, valid)
        counts.append(float((hit & masked).sum()) / float(masked.sum().clamp_min(1)))
        masked = masked & ~hit
    assert not masked.any(), "some positions never revealed"
    for t_, frac in enumerate(counts):                             # empirical reveal rate vs 1/(T - t)
        assert abs(frac - 1.0 / (T - t_)) < 0.03, (t_, frac)


@pytest.mark.parametrize("name", ["forbidden", "exactly_k", "dyck1_amb", "pairs_fg"])
@pytest.mark.parametrize("opts", [dict(ess_min=0.0), dict(ess_min=0.9, defensive=0.25),
                                  dict(ess_min=0.0, twist_temp=0.5)])
def test_zhat_unbiased_independent_schedule(name, opts):
    """With the MDLM independent-unmasking schedule in both the proposal and the native kernel, E[Z_hat] = h0."""
    den, cl, route = setup(name)
    hits, g = 0, torch.Generator().manual_seed(21)
    n = 200_000
    for _ in range(n // 20000):
        hits += int(route.accepts(native_sample(den, cl, N, T, 20000, g, schedule="independent")).sum())
    h0, h0se = hits / n, math.sqrt(hits) / n
    out = twisted_smc(den, cl, route, N, T, 6000, 4, torch.Generator().manual_seed(22), schedule="independent",
                      res_gen=torch.Generator().manual_seed(23), **opts)
    z = np.exp(out["logZ"])
    zscore = (z.mean() - h0) / math.sqrt(z.var() / len(z) + h0se ** 2)
    assert abs(zscore) < 4, f"{name} {opts}: mean {z.mean():.4f} vs h0 {h0:.4f} (z={zscore:.2f})"
    assert not out["any_finite_invalid"] and all(v == 0 for v in out["guard"].values())


def test_independent_schedule_targets_a_different_process():
    """The two schedules give measurably different h0 on the same model and constraint, so they are different
    native processes: "exact" has to name which one. (Observed separation: copy 10.5 se, forbidden 9.4 se,
    exactly_k 3.9 se at n = 300k.)"""
    den, cl, route = setup("copy")
    g = torch.Generator().manual_seed(31)
    n = 300_000
    def h0(sched):
        hits = 0
        for _ in range(n // 20000):
            hits += int(route.accepts(native_sample(den, cl, N, T, 20000, g, schedule=sched)).sum())
        return hits / n
    a, b = h0("fixed"), h0("independent")
    assert abs(a - b) > 5 * math.sqrt(a * (1 - a) / n), (a, b)


def test_schedule_default_is_unchanged():
    den, cl, route = setup("dyck1_amb")
    a = twisted_smc(den, cl, route, N, T, 400, 4, torch.Generator().manual_seed(3), ess_min=0.9)
    b = twisted_smc(den, cl, route, N, T, 400, 4, torch.Generator().manual_seed(3), ess_min=0.9, schedule="fixed")
    assert np.array_equal(a["logZ"], b["logZ"])


def test_mean_field_denoiser_makes_the_correction_unnecessary():
    """Proposition 'mean-field collapse' in the notes: if the denoiser's predictions do not depend on the
    revealed values, the frozen twist equals the Doob function, every potential is 1, and a single particle
    is already an exact draw. This is the boundary case that delimits when the method is worth running."""
    import math
    import torch
    from twistroute.decode.smc import twisted_smc
    from twistroute.torch_routes.build import build_route
    from twistroute.models.classes import TokenClasses

    class MeanFieldDenoiser:
        """Predictions depend on how many positions are masked, never on what was revealed."""
        mask_id, V = 4, 5

        def logits(self, x):
            lp = torch.zeros(x.shape + (self.V,))
            lp[..., self.mask_id] = float("-inf")
            lp[..., 0] += 0.8                               # a fixed, context-free preference
            return torch.log_softmax(lp, -1)

    den = MeanFieldDenoiser()
    cl = TokenClasses(torch.tensor([0, 1, 2, 3, 0]), 4, excluded=(den.mask_id,))
    route = build_route(dict(kind="forbidden", A=4, N=6, patterns=[[0, 0]]), "cpu")
    out = twisted_smc(den, cl, route, 6, 3, 200, 1, torch.Generator().manual_seed(0), ess_min=0.0)
    # every run must return exactly eta_0 = h_0, with no spread at all
    z = out["logZ"]
    assert z.std() < 1e-6, z.std()
    assert abs(float(z.mean()) - float(out["eta0"])) < 1e-6      # eta0 is already a log mass

    # and eta_0 must itself be the true constraint probability under the (context-free) product law
    lp = den.logits(torch.full((1, 6), den.mask_id))
    w = cl.class_logw(lp)[0].exp()[0]                   # same at every position, by construction
    import itertools
    want = sum(float(w[y[0]] * w[y[1]] * w[y[2]] * w[y[3]] * w[y[4]] * w[y[5]])
               for y in itertools.product(range(4), repeat=6)
               if not any(y[i] == 0 and y[i + 1] == 0 for i in range(5)))
    assert abs(float(out["eta0"]) - math.log(want)) < 1e-6, (float(out["eta0"]), math.log(want))


def test_log_mean_exp_matches_the_naive_computation():
    """Shared by the Zhat accumulation and the look-ahead gain average; must be stable far below the float
    floor, where the naive exp-mean-log underflows to zero."""
    import math
    import torch
    from twistroute.decode.resampling import log_mean_exp
    x = torch.tensor([[-1.0, 0.0, 2.0], [-900.0, -901.0, -902.0]], dtype=torch.float64)
    got = log_mean_exp(x, -1)
    want0 = math.log(sum(math.exp(v) for v in [-1.0, 0.0, 2.0]) / 3)
    assert abs(float(got[0]) - want0) < 1e-12
    assert abs(float(got[1]) - (-900.0 + math.log((1 + math.exp(-1) + math.exp(-2)) / 3))) < 1e-9


def test_generator_device_mismatch_is_caught_with_a_clear_message():
    """A CPU generator against a CUDA route fails deep inside torch.rand, naming neither the caller nor the
    cause; a GPU job died that way after loading a 148M-parameter model. No CPU-only test can reproduce the
    real failure, so the guard is checked against a route that merely claims to be on another device."""
    import pytest
    import torch
    from twistroute.decode.smc import twisted_smc
    from twistroute.models.classes import TokenClasses
    from twistroute.torch_routes.build import build_route

    class Den:
        mask_id, V = 4, 5
        def logits(self, x):
            lp = torch.zeros(x.shape + (self.V,)); lp[..., self.mask_id] = float("-inf")
            return torch.log_softmax(lp, -1)

    route = build_route(dict(kind="forbidden", A=4, N=6, patterns=[[0, 0]]), "cpu")
    route.device = "cuda"                                  # pretend; no GPU needed to exercise the check
    cl = TokenClasses(torch.tensor([0, 1, 2, 3, 0]), 4, excluded=(4,))
    with pytest.raises(ValueError, match="generator is on cpu but the route is on cuda"):
        twisted_smc(Den(), cl, route, 6, 3, 2, 1, torch.Generator().manual_seed(0))
