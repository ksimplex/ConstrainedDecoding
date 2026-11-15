"""Is the accuracy loss at long lengths hidden mass (notes Sec. 6), and which remedy keeps resampling?

For each cell (model, constraint, N) and each schedule (one reveal per step; the policy at theta = 0), the same
sampler is run under several resampling / twist settings. Every setting has E[Z_hat] = p(C); they differ in how
much of that expectation sits in rare runs. Reported per task: mean Z_hat / p(C) with its plain standard error,
the relative spread and upper tail of Z_hat over runs, marginal and adjacent-pair error against the exact
constrained law, steps, resampling count, and, for the settings without resampling, the hidden mass measured
directly from the weight traces: the share of the total final weight carried by particles whose normalised
weight fell below eps at some step (the particles a resampling step would have been likely to drop).

Kernels are fitted once per (model, N) and cached next to --out. Results are written after every task; rerun with
the same --out to resume."""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SETTINGS = {  # name: (smc keyword arguments, particles per run, runs relative to --nrun)
    "off": (dict(ess_min=0.0), 4, 1.0),
    "defensive": (dict(defensive=0.25), 4, 1.0),
    "defensive tau=0.7": (dict(defensive=0.25, twist_temp=0.7), 4, 1.0),
    "defensive tau=0.5": (dict(defensive=0.25, twist_temp=0.5), 4, 1.0),
    "defensive tau=0.35": (dict(defensive=0.25, twist_temp=0.35), 4, 1.0),
    "defensive trigger=0.25": (dict(defensive=0.25, ess_min=0.25), 4, 1.0),
    "defensive K=16": (dict(defensive=0.25), 16, 1.0),
    "defensive K=16 tau=0.5": (dict(defensive=0.25, twist_temp=0.5), 16, 1.0),
    "off K=16": (dict(ess_min=0.0), 16, 1.0),
}
CONTROL_SETTINGS = ("off", "defensive", "defensive tau=0.5", "defensive K=16")     # the N-step arm is expensive
EPS = (1e-1, 1e-2, 1e-3)


def kernel_path(out, m, N):
    return f"{os.path.splitext(out)[0]}.kernel_{m}_{N}.npz"


def fit_kernel(job):
    m, N, pilot, seed, out = job
    torch.set_num_threads(1)
    from compare import make_model, stream_seed
    from twistroute.decode import fit_distance_kernel
    k, _ = fit_distance_kernel(make_model(m, N, 3), N, pilot, torch.Generator().manual_seed(stream_seed(seed, m, N, "pilot")))
    np.savez(kernel_path(out, m, N), ku=k.ku.numpy(), ks=k.ks.numpy())
    return m, N


def hidden_mass(trace):
    """trace (steps, nrun, K): cumulative log-weights of a run without resampling."""
    lw = trace.astype(np.float64)
    wn = np.exp(lw - np.logaddexp.reduce(lw, axis=-1, keepdims=True))            # normalised within the run, per step
    low = wn[:-1].min(0) if len(wn) > 1 else np.ones_like(wn[0])                  # lowest share before the last step
    fin = np.exp(lw[-1] - lw[-1].max())
    return {f"hidden_mass_below_{e:g}": float((fin * (low < e)).sum() / fin.sum()) for e in EPS}


def run(job):
    m, c, N, sched, setting, nrun, seed, out = job
    torch.set_num_threads(1)
    from compare import make_model, make_constraint, metrics, stream_seed
    from twistroute.decode import twisted_smc, CouplingPolicy, DistanceKernel
    from twistroute.models import parallel_log_prob
    den = make_model(m, N, 3)
    cl, route, truth = make_constraint(c, den, N)
    pair_truth = truth.pair_marginals() if hasattr(truth, "pair_marginals") else None
    kw, K, rel = SETTINGS[setting]
    nr = max(20, int(round(nrun * rel)))
    if sched == "one per step" and K > 4:        # the N-step arm with 16 particles: half the runs, to bound its cost
        nr = max(20, nr // 2)
    if sched == "one per step":
        schedule = "fixed"
    else:
        kz = np.load(kernel_path(out, m, N))
        schedule = CouplingPolicy(DistanceKernel(kz["ku"], kz["ks"]), 0.0, priority=sched.split()[0], mixed=False)
    sd = stream_seed(seed, m, c, N, sched, setting)
    t0 = time.time()
    o = twisted_smc(den, cl, route, N, N, nrun=nr, K=K, gen=torch.Generator().manual_seed(sd), schedule=schedule,
                    return_particles=True, return_trace=("ess_min" in kw and kw["ess_min"] == 0.0),
                    path_stat=lambda a, b: parallel_log_prob(den, a, b), **kw)
    row = dict(model=m, constraint=c, N=N, schedule=sched, setting=setting, nrun=nr, K=K, seed=sd, log_pC=truth.log_pC,
               seconds=round(time.time() - t0, 1), n_res=float(np.mean(o["n_res"])), frac_valid=o["frac_valid_finite"],
               **metrics(o, den, truth, N, den.S, K, pair_truth))
    z = np.exp(np.asarray(o["logZ"], dtype=np.float64) - truth.log_pC)
    row["rel_sd_z"] = float(z.std() / z.mean())
    if "trace" in o:
        row.update(hidden_mass(o["trace"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--cells", nargs="+", default=["chain_0.6:window", "chain_0.85:window", "hidden_0.85:window", "chain_0.85:copy_block"])
    ap.add_argument("--schedules", nargs="+", default=["one per step", "bound theta=0", "separator theta=0"])
    ap.add_argument("--nrun", type=int, default=200); ap.add_argument("--pilot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="hidden_mass_check.json")
    a = ap.parse_args()
    import multiprocessing as mp
    cells = [tuple(x.split(":")) for x in a.cells]
    rows = json.load(open(a.out))["rows"] if os.path.exists(a.out) else []
    done = {(r["model"], r["constraint"], r["N"], r["schedule"], r["setting"]) for r in rows}
    jobs = [(m, c, N, s, st, a.nrun, a.seed, a.out) for N in sorted(a.N, reverse=True) for (m, c) in cells for s in a.schedules
            for st in (CONTROL_SETTINGS if s == "one per step" else SETTINGS)]
    jobs.sort(key=lambda j: (j[3] != "one per step", -j[2]))                      # longest tasks first
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4]) not in done]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        need = [(m, N, a.pilot, a.seed, a.out) for m in sorted({c[0] for c in cells}) for N in a.N if not os.path.exists(kernel_path(a.out, m, N))]
        for m, N in pool.imap_unordered(fit_kernel, need):
            print(f"[{time.time() - t0:7.0f}s] kernel {m} N={N}", flush=True)
        for i, row in enumerate(pool.imap_unordered(run, jobs)):
            rows.append(row)
            json.dump(dict(args=vars(a), rows=rows, wall_seconds=round(time.time() - t0, 1)), open(a.out + ".tmp", "w"), indent=1)
            os.replace(a.out + ".tmp", a.out)
            print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)} {row['model']} {row['constraint']} N={row['N']} | {row['schedule']} | {row['setting']}: "
                  f"Z {row['z_ratio']:.2f} margTV {row['marg_tv']:.3f} rel sd {row['rel_sd_z']:.2f} ({row['seconds']:.0f}s)", flush=True)
    print("done:", len(rows), "tasks")


if __name__ == "__main__":
    main()
