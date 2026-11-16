"""Is the one-reveal-per-step control unbiased in practice?  Many seeds of the `fixed 1/step` arm of compare.py,
varying only the resampling rule and the twist calibration. Every arm has E[Z_hat] = p(C) (notes Thm 4.2,
Props 6.2-6.4); the question is how far a finite number of runs falls short of it (hidden mass, notes Sec. 6).

Output: <out>.npz with the per-run ratio Z_hat / p(C) for every (model, constraint, arm), shape (seeds * nrun,),
and <out>.json with a summary. Saved after every finished task, so a pre-empted job keeps what it has."""
import argparse, json, math, os, time
import numpy as np
import torch

ARMS = {"off": dict(ess_min=0.0), "defensive": dict(defensive=0.25), "plain": dict(defensive=0.0),
        "defensive_tau0.5": dict(defensive=0.25, twist_temp=0.5)}


def run(job):
    m, c, N, arm, seed, nrun, K = job
    torch.set_num_threads(1)
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compare import make_model, make_constraint, n_symbols
    from twistroute.decode import twisted_smc
    den = make_model(m, N, n_symbols(c))
    cl, route, truth = make_constraint(c, den, N)
    t0 = time.time()
    out = twisted_smc(den, cl, route, N, N, nrun=nrun, K=K, gen=torch.Generator().manual_seed(seed),
                      schedule="fixed", **ARMS[arm])
    z = np.exp(np.asarray(out["logZ"], dtype=np.float64) - truth.log_pC)
    return (m, c, arm, seed), z, float(np.mean(out["n_res"])), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--nrun", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1000, help="generator seeds are seed0, seed0 + 1, ...")
    ap.add_argument("--cells", nargs="+", default=["chain_0.6:copy", "chain_0.85:copy", "hidden_0.85:copy",
                    "segments_0.85:forbidden", "hidden_0.85:forbidden", "chain_0.85:counting", "chain_0.6:dyck"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS)); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="control_check")
    a = ap.parse_args()
    jobs = [(*cell.split(":"), a.N, arm, s, a.nrun, a.K) for s in range(a.seed0, a.seed0 + a.seeds) for arm in a.arms for cell in a.cells]
    import multiprocessing as mp
    z, nres, secs, t0 = {}, {}, {}, time.time()

    def save():
        arrs = {f"{m}|{c}|{arm}": np.concatenate([z[k] for k in sorted(z) if k[:3] == (m, c, arm)])
                for (m, c, arm) in sorted({k[:3] for k in z})}
        np.savez_compressed(a.out + ".tmp.npz", **arrs); os.replace(a.out + ".tmp.npz", a.out + ".npz")
        summ = []
        for key, v in arrs.items():
            m, c, arm = key.split("|"); ks = [k for k in z if k[:3] == (m, c, arm)]
            per_seed = np.array([z[k].mean() for k in ks])
            summ.append(dict(model=m, constraint=c, arm=arm, N=a.N, K=a.K, seeds=len(ks), runs=len(v), z_ratio=float(v.mean()),
                             se=float(v.std(ddof=1) / math.sqrt(len(v))), seeds_below_1=int((per_seed < 1).sum()),
                             seed_min=float(per_seed.min()), seed_max=float(per_seed.max()), top_run_share=float(v.max() / v.sum()),
                             median=float(np.median(v)), var_logZ=float(np.log(v[v > 0]).var()), frac_zero=float((v == 0).mean()),
                             n_res=float(np.mean([nres[k] for k in ks])), seconds_per_task=float(np.mean([secs[k] for k in ks]))))
        json.dump(dict(args=vars(a), wall_seconds=time.time() - t0, tasks_done=len(z), tasks=len(jobs), summary=summ),
                  open(a.out + ".json", "w"), indent=1)

    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (key, zz, nr, s) in enumerate(pool.imap_unordered(run, jobs)):
            z[key], nres[key], secs[key] = zz, nr, s
            if (i + 1) % max(1, a.workers) == 0 or i + 1 == len(jobs):
                save(); print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)} tasks", flush=True)


if __name__ == "__main__":
    main()
