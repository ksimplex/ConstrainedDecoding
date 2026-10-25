"""
Twist design on a CPU testbed that reproduces the OWT-130M resampling pathology: the synthetic denoiser with
strong context dependence (scale 8) on the TWISTER Fig. 1(a) constraint, where the frozen twist is
over-confident by up to ~e^-48 and plain resampling inflates the s.e. of Z_hat about 5x.

Arms (all unbiased; notes, free intermediate twists):
    frozen twist, no / plain / defensive resampling; K = 8 plain as a compute-matched comparator;
    look-ahead twists of horizon j in {1, 2, T} (m speculative step-exact rollouts), plain and defensive;
    native-kernel mixture proposal a = 0.2.
Per arm: Z_hat / h0 with s.e., denoiser calls per run (cost), efficiency 1 / (se_rel^2 * cost), and for a
subset the hidden mass of the resampling-off trace (diagnostics.hidden_mass).

usage: python experiments/twist_lab.py [--scale 8] [--runs 20000] [--native 200000] [--out twist_lab]
"""
import argparse, json, time
import numpy as np, torch
from twistroute.diagnostics import hidden_mass
from twistroute.experiment import load_denoiser, fig1a_setup, native_hits, run_smc, z_summary

ARMS = {  # name: (K, smc options)
    "frozen_off": (4, dict(ess_min=0.0)),
    "frozen_plain": (4, dict(ess_min=0.5)),
    "frozen_def25": (4, dict(ess_min=0.5, defensive=0.25)),
    "frozen_plain_K8": (8, dict(ess_min=0.5)),
    "frozen_def25_K8": (8, dict(ess_min=0.5, defensive=0.25)),
    "look1_plain": (4, dict(ess_min=0.5, lookahead=1)),
    "look1_def25": (4, dict(ess_min=0.5, lookahead=1, defensive=0.25)),
    "look2_plain": (4, dict(ess_min=0.5, lookahead=2)),
    "lookT_plain": (4, dict(ess_min=0.5, lookahead=1000)),
    "lookT_def25": (4, dict(ess_min=0.5, lookahead=1000, defensive=0.25)),
    "look1_geom4_plain": (4, dict(ess_min=0.5, lookahead=1, lookahead_draws=4, lookahead_mean="geom")),
    "look1_shrink50_plain": (4, dict(ess_min=0.5, lookahead=1, lookahead_shrink=0.5)),
    "lookT_geom4_plain": (4, dict(ess_min=0.5, lookahead=1000, lookahead_draws=4, lookahead_mean="geom")),
    "lookT_geom4_shrink50_plain": (4, dict(ess_min=0.5, lookahead=1000, lookahead_draws=4, lookahead_mean="geom",
                                           lookahead_shrink=0.5)),
    "look1_geom4_def25": (4, dict(ess_min=0.5, lookahead=1, lookahead_draws=4, lookahead_mean="geom", defensive=0.25)),
    "lookT_geom4_def25": (4, dict(ess_min=0.5, lookahead=1000, lookahead_draws=4, lookahead_mean="geom",
                                  defensive=0.25)),
    "mix20_plain": (4, dict(ess_min=0.5, mix_native=0.2)),
    "mix20_def25": (4, dict(ess_min=0.5, mix_native=0.2, defensive=0.25)),
}
TRACE_ARMS = {"frozen": dict(), "look1": dict(lookahead=1), "lookT": dict(lookahead=1000), "mix20": dict(mix_native=0.2),
              "look1_geom4": dict(lookahead=1, lookahead_draws=4, lookahead_mean="geom"),
              "lookT_geom4": dict(lookahead=1000, lookahead_draws=4, lookahead_mean="geom"),
              "look1_shrink50": dict(lookahead=1, lookahead_shrink=0.5)}


def denoiser_calls(T, K, lookahead=0, lookahead_draws=1, **_):  # noqa: D401
    """Denoiser rows evaluated per run: K per step for the real path (T - 1 successor calls plus the initial
    call), plus K * m * min(j, steps remaining) per step for look-ahead rollouts."""
    calls = K * T
    if lookahead:
        calls += K * lookahead_draws * sum(min(lookahead, T - 1 - t) for t in range(-1, T - 1))
    return calls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--runs", type=int, default=20_000)
    ap.add_argument("--native", type=int, default=200_000)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--trace-arms", default=",".join(TRACE_ARMS))
    ap.add_argument("--out", default="twist_lab")
    a = ap.parse_args(argv)
    M, dev = load_denoiser(a.N, synthetic=True, scale=a.scale)
    cl, route = fig1a_setup(M, a.N, dev, pool=8, seed=1)
    gen = torch.Generator().manual_seed(2024)
    t0 = time.time()
    h0 = native_hits(M, cl, route, a.N, a.T, a.native, gen) / a.native
    res = dict(args=vars(a), h0=h0, n0=a.native, native_s=time.time() - t0, arms={}, hidden={})
    arrays = {}
    for name in a.arms.split(","):
        K, kw = ARMS[name]
        r = run_smc(M, cl, route, a.N, a.T, a.runs, K, gen, **kw)
        zs, cost = z_summary(r["logZ"], h0), denoiser_calls(a.T, K, **kw)
        res["arms"][name] = dict(K=K, options=kw, cost=cost, seconds=r["seconds"],
                                 mean_resamplings=float(r["n_res"].mean()), any_finite_invalid=r["any_finite_invalid"],
                                 efficiency=1.0 / (zs["se_rel"] ** 2 * cost), **zs)
        arrays[f"logZ_{name}"] = r["logZ"]
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["arms"][name].items() if k != "options"},
              flush=True)
        json.dump(res, open(a.out + ".json", "w"), indent=1)
    for name in a.trace_arms.split(","):
        r = run_smc(M, cl, route, a.N, a.T, a.runs, 4, gen, ess_min=0.0, return_trace=True, **TRACE_ARMS[name])
        hm = hidden_mass(r["trace"], h0, eps_grid=(1e-8, 1e-6, 1e-4, 1e-2))
        res["hidden"][name] = dict(hm, min_log_weight=float(np.nanmin(np.where(np.isfinite(r["trace"]), r["trace"], np.nan))))
        print("hidden", name, {str(k): round(v, 4) for k, v in hm["mass_below"].items()}, flush=True)
        json.dump(res, open(a.out + ".json", "w"), indent=1, default=float)
    np.savez_compressed(a.out + ".npz", **arrays)


if __name__ == "__main__":
    main()
