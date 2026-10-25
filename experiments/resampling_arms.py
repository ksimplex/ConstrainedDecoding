"""
Resampling arms on the TWISTER Fig. 1(a) configuration ("a.*", OWT-130M, N = 32, pool = 8, partition seed 1,
T = 8). One script for the whole resampling investigation (notes Sec. 7):

    K4_plain           K = 4, multinomial resampling at ESS < K/2 (TWISTER default)      100k runs: 0.930
    K4_off             K = 4, no resampling                                              0.998
    K1                 single particle                                                    0.987
    K4_cpu, K4_sys     CPU multinomial / systematic resampling (rules out the CUDA RNG)   0.931 / 0.921
    K4_def25           defensive resampling, eps = 0.25                                   0.998
    K4_beta50, K4_beta25, K4_beta50_def25   tempered twists, alone and with eps = 0.25   0.932 / 0.953 / 1.000
    K4_off_trace       K4_off with per-step log-weights saved, for the hidden-mass decomposition
                       (diagnostics.hidden_mass)
(Z_hat / h0 at 100k runs; see resampling_check_100k.json, resampling_diag.json, remedy_check_summary.json.)

h0 is either passed (--h0, --n0; 0.0248375 from 400k native samples for the default config) or estimated
from --native native samples first. --rng-tests adds direct checks of the device generator (multinomial
calibration, independence of the next draws), which do not touch the SMC generator.

Supersedes resampling_check.py, resampling_diag.py, remedy_check.py and weight_trace.py; with the same
--gen-seed and arms it reproduces their outputs exactly (tests/test_experiments.py).

usage: python experiments/resampling_arms.py --arms K4_plain,K4_off,K1 --runs 100000 --native 400000
       python experiments/resampling_arms.py --arms K4_def25 --h0 0.0248375 --n0 400000
"""
import argparse, json
import numpy as np, torch
from twistroute.experiment import load_denoiser, fig1a_setup, native_hits, run_smc, z_summary

ARMS = {
    "K4_plain": dict(K=4, ess_min=0.5),
    "K4_off": dict(K=4, ess_min=0.0),
    "K1": dict(K=1, ess_min=0.5),
    "K4_cpu": dict(K=4, ess_min=0.5, resampler="cpu"),
    "K4_sys": dict(K=4, ess_min=0.5, resampler="systematic"),
    "K4_def25": dict(K=4, ess_min=0.5, defensive=0.25),
    "K4_beta50": dict(K=4, ess_min=0.5, twist_power=0.5),
    "K4_beta25": dict(K=4, ess_min=0.5, twist_power=0.25),
    "K4_beta50_def25": dict(K=4, ess_min=0.5, twist_power=0.5, defensive=0.25),
    "K4_off_trace": dict(K=4, ess_min=0.0, return_trace=True),
}


def rng_tests(dev, B=200_000, K=4):
    """Calibration of torch.multinomial on dev, and whether draws after it repeat or correlate with it."""
    g = torch.Generator(device=dev).manual_seed(1)
    p = torch.rand(B, K, device=dev, generator=g) ** 3
    p = p / p.sum(-1, keepdim=True)
    idx = torch.multinomial(p, K, replacement=True, generator=g)
    cnt = torch.zeros(B, K, device=dev).scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
    flat_p, flat_f = p.reshape(-1), (cnt / K).reshape(-1)
    b = torch.bucketize(flat_p, torch.linspace(0, 1, 11, device=dev))
    calib = [[float(flat_p[b == i].mean()), float(flat_f[b == i].mean())] for i in range(1, 11) if (b == i).sum() > 1000]
    g1 = torch.Generator(device=dev).manual_seed(5); g2 = torch.Generator(device=dev).manual_seed(5)
    u_ref = torch.rand(B, 8, device=dev, generator=g2)
    idx1 = torch.multinomial(p, K, replacement=True, generator=g1)
    u_after = torch.rand(B, 8, device=dev, generator=g1)
    corr = [abs(float(np.corrcoef(idx1[:, j].float().cpu().numpy(), u_after[:, k].cpu().numpy())[0, 1]))
            for j in range(K) for k in range(8)]
    return dict(calibration=calib, after_equals_unadvanced=float((u_after == u_ref).float().mean()),
                max_abs_corr_idx_next_uniform=max(corr))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="K4_plain,K4_off,K1")
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--pool", type=int, default=8); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=100_000)
    ap.add_argument("--h0", type=float, default=None); ap.add_argument("--n0", type=int, default=400_000)
    ap.add_argument("--native", type=int, default=400_000, help="native samples for h0 when --h0 is not given")
    ap.add_argument("--gen-seed", type=int, default=777)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--rng-tests", action="store_true")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="resampling_arms")
    a = ap.parse_args(argv)
    unknown = set(a.arms.split(",")) - set(ARMS)
    if unknown:
        ap.error(f"unknown arms {sorted(unknown)}; choose from {sorted(ARMS)}")

    M, dev = load_denoiser(a.N, a.synthetic, a.device)
    info = dict(args=vars(a), arms={k: ARMS[k] for k in a.arms.split(",")}, torch=torch.__version__, timing={})
    if a.rng_tests:
        info["rng_tests"] = rng_tests(dev)
    gen = torch.Generator(device=dev).manual_seed(a.gen_seed)
    res_gen = torch.Generator(device="cpu").manual_seed(99)
    cl, route = fig1a_setup(M, a.N, dev, pool=a.pool, seed=a.seed)
    if a.h0 is None:
        a.h0, a.n0 = native_hits(M, cl, route, a.N, a.T, a.native, gen, a.chunk) / a.native, a.native
    out = dict(h0=np.array([a.h0, a.n0]))
    for arm in a.arms.split(","):
        kw = dict(ARMS[arm]); K = kw.pop("K")
        if kw.get("resampler") in ("cpu", "systematic"):
            kw["res_gen"] = res_gen
        r = run_smc(M, cl, route, a.N, a.T, a.runs, K, gen, a.chunk, **kw)
        for k in ("logZ", "n_res", "ess_min", "trace"):
            if k in r:
                out[f"{k}_{arm}"] = r[k]
        info["timing"][arm] = r["seconds"]
        s = z_summary(r["logZ"], a.h0)
        print(arm, "Z/h0", round(s["ratio"], 4), "se_rel", round(s["se_rel"], 4), "resamplings",
              round(float(r["n_res"].mean()), 3), "s", round(r["seconds"]), flush=True)
        np.savez_compressed(a.out + ".npz", **out)
    json.dump(info, open(a.out + ".json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
