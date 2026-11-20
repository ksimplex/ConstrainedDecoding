import numpy as np
from twistroute.diagnostics import psis_khat, pairwise_tv, subsample_medians


def test_khat_recovers_pareto_shape():
    rng = np.random.default_rng(1)
    for xi in (0.3, 0.5, 0.8):
        est = np.mean([psis_khat(np.log(rng.pareto(1 / xi, 4000) + 1)) for _ in range(30)])
        assert abs(est - xi) < 0.1


def test_pairwise_tv_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(2)
    S = rng.integers(0, 3, (5000, 6))
    assert pairwise_tv(S, S, 3) == 0.0
    S2 = S.copy(); S2[:, 0] = 0
    assert pairwise_tv(S, S2, 3) > 0.05


def test_subsample_medians_flat_for_light_tails():
    z = np.random.default_rng(3).normal(1, 0.1, 4000)
    m = subsample_medians(z, [50, 500, 2000])
    assert max(m.values()) - min(m.values()) < 0.01


def test_tv_with_floor_matches_floor_for_same_law_and_exceeds_it_otherwise():
    from twistroute.diagnostics import tv_with_floor
    rng = np.random.default_rng(5)
    R = rng.integers(0, 3, (6000, 6))
    S_same = rng.integers(0, 3, (1000, 6))
    tv, floor = tv_with_floor(S_same, R, 3)
    assert abs(tv - floor) < 0.5 * floor
    S_shift = S_same.copy(); S_shift[:, :3] = 0
    tv2, floor2 = tv_with_floor(S_shift, R, 3)
    assert tv2 > 3 * floor2


def _trace(T=4, R=1000, K=4, seed=0):
    """Random resampling-off log-weight trajectories (cumulative, so later steps refine earlier ones)."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 2.0, (T, R, K)), axis=0)


def test_first_trigger_finds_first_low_ess_step():
    from twistroute.diagnostics import first_trigger
    tr = np.zeros((3, 2, 3))                     # K = 3, so ESS < K/2 is reachable
    tr[1, 0] = [0.0, -30.0, -30.0]               # run 0: degenerate at step 1
    s, wn = first_trigger(tr)
    assert list(s) == [1, -1] and np.allclose(wn.sum(-1), 1)


def test_hidden_mass_on_constructed_trace():
    """One run triggers at step 0 (normalised weights ~[1, e^-40, e^-40]). The low-weight particles end with
    weights 1 and 0, the dominant one with 3, so the low-weight share of Z_hat is 1 / (3 + 1 + 0)."""
    from twistroute.diagnostics import hidden_mass
    tr = np.array([[[0.0, -40.0, -40.0]], [[np.log(3.0), 0.0, -np.inf]]])   # (T=2, R=1, K=3)
    h0 = 4.0 / 3.0                                                            # mean final weight
    out = hidden_mass(tr, h0, eps_grid=(1e-6, 0.9))
    assert out["frac_runs_triggered"] == 1.0
    assert np.isclose(out["Z_over_h0"], 1.0)
    assert np.isclose(out["mass_below"][1e-6], 0.25)
    assert np.isclose(out["mass_below"][0.9], 0.25)              # the other particle has wn ~ 1


def test_perfect_twist_hides_little():
    """Corollary in notes Sec. 7: if final contributions are proportional to the normalised weights at the
    trigger step (twist = Doob function), the mass below eps is at most K * eps of the total."""
    from twistroute.diagnostics import hidden_mass
    tr = _trace()
    tr[-1] = tr[0] + 0.3                     # every particle's future weight ratio is the same constant
    tr[1:-1] = tr[0]                         # so the first trigger step is 0
    out = hidden_mass(tr, h0=1.0, eps_grid=(1e-3, 1e-2, 0.1))
    for e, m in out["mass_below"].items():
        assert m <= 4 * e * out["Z_over_h0"] + 1e-12


def test_hidden_mass_detects_recovering_particles():
    """When low-weight particles recover (final weight independent of the trigger-step weight), the mass below
    eps exceeds the perfect-twist bound by a wide margin."""
    from twistroute.diagnostics import hidden_mass
    rng = np.random.default_rng(1)
    tr = _trace(seed=2)
    tr[-1] = rng.normal(0, 0.5, tr[-1].shape)                   # recovery: final weights ignore the past
    tr[1:-1] = tr[0]
    out = hidden_mass(tr, h0=1.0, eps_grid=(1e-2,))
    assert out["mass_below"][1e-2] > 2 * 4 * 1e-2 * out["Z_over_h0"]      # twice the perfect-twist bound
