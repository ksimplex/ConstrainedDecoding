"""
Policy against the fixed schedule on oracle models, measured against the exact target p(. | C).

Cells are (model, constraint, N). Models: Markov chains with two correlation lengths (a revealed position
separates its two sides exactly), a hidden chain (dependence decays but nothing separates), and a chain cut
into independent segments. Constraints: forbidden motifs (local), exactly-k (global, weak), same-order copy
(hard, pairwise, long-range), Dyck-1 (nested). Tokens are classes.

Arms per cell
    fixed          the existing schedule at 1, 2, 4, 8 reveals per step (1 per step samples the target itself)
    policy         CouplingPolicy with a pilot kernel, priority "bound", thresholds THETAS, without the mixed term
    small pilot    the same with a kernel fitted on few traces
    priorities     "confident" and "random" at two thresholds
    mixed          with the mixed-path term, fewer runs (it costs N * A extra route passes per particle per step)

Metrics (pooled over runs; standard errors by a delete-one-batch jackknife over 10 batches of runs)
    steps          reveal steps per particle = denoiser calls
    marg_tv        mean over positions of TV(weighted class marginal, exact marginal)
    pair_tv        the same for adjacent pairs (automaton constraints only)
    z_ratio        mean Z_hat / exact p(C)
    sd_r, kl_r     spread of the parallel-reveal gap r over the output distribution, and Var(r) / 2
    ess            mean over runs of the minimum effective sample size fraction
    kept           effective fraction of outputs left if each were weighted by exp(r)
    *_corr         marg_tv and pair_tv after that weighting
    route_passes   batched route passes per particle per step requested by the policy
"""
import argparse, json, math, os, time
import numpy as np
import torch

THETAS = (0.0, 0.003, 0.01, 0.03, 0.1)
MODELS = ("chain_0.6", "chain_0.85", "hidden_0.85", "segments_0.85")
CONSTRAINTS = ("forbidden", "counting", "copy", "dyck")


def make_model(name, N, S):
    from twistroute.models import lazy_chain, noisy_hmm
    if name == "chain_0.6":
        return lazy_chain(N, S, 0.6, seed=1, concentration=2.0)
    if name == "chain_0.85":
        return lazy_chain(N, S, 0.85, seed=1, concentration=2.0)
    if name == "hidden_0.85":
        return noisy_hmm(N, S, 0.85, 0.3, seed=1, concentration=2.0)
    if name == "segments_0.85":
        return lazy_chain(N, S, 0.85, seed=1, concentration=2.0, breaks=tuple(range(7, N - 1, 8)))
    raise ValueError(name)


def make_constraint(name, den, N):
    from twistroute import compile as C
    from twistroute.models import ConstrainedTruth, CopyTruth
    from twistroute.torch_routes import torch_route, build_route
    cl = den.classes()
    if name == "copy":
        return cl, build_route(dict(kind="copy", A=den.S, pairs=N // 2, N=N)), CopyTruth(den, cl)
    d = {"forbidden": lambda: C.dfa_forbidden([(0, 0), (1, 2)], 3), "counting": lambda: C.dfa_exactly_k(0, N // 3, 3),
         "dyck": lambda: C.dfa_dyck1(N)}[name]()
    return cl, torch_route(d), ConstrainedTruth(den, cl, d)


def metrics(out, den, truth, N, A, K, pair_truth, nb=10):
    logZ = out["logZ"]; nrun = len(logZ)
    lw = out["logw"]; w = np.exp(lw - lw.max(1, keepdims=True)); w = w / w.sum(1, keepdims=True)
    cls = out["classes"]
    r = den.log_joint(torch.tensor(out["tokens"]).view(-1, N)).numpy().reshape(lw.shape) - out["path_stat"]
    oh = np.eye(A)[cls]                                                   # (nrun, K, N, A)
    pair = oh[:, :, :-1, :, None] * oh[:, :, 1:, None, :] if pair_truth is not None else None
    zrel = np.exp(logZ - truth.log_pC)

    def stat(idx):
        rw = (np.exp(logZ[idx] - logZ.max())[:, None] * w[idx]); rw = rw / rw.sum()
        rr = r[idx]; m = (rw * rr).sum(); var = (rw * (rr - m) ** 2).sum()
        e = np.exp(rr - rr.max()); cw = rw * e; kept = cw.sum() ** 2 / (rw * e * e).sum(); cw = cw / cw.sum()
        o = dict(steps=out["steps"][idx].mean(), z_ratio=zrel[idx].mean(), sd_r=math.sqrt(max(var, 0.0)), kl_r=var / 2,
                 mean_r=m, ess=(out["ess_min"][idx] / K).mean(), kept=kept, var_logZ=np.log(zrel[idx]).var())
        for tag, ww in (("", rw), ("_corr", cw)):
            mg = np.einsum("rk,rkna->na", ww, oh[idx])
            o["marg_tv" + tag] = 0.5 * np.abs(mg - truth.marginals).sum(-1).mean()
            if pair is not None:
                pm = np.einsum("rk,rknab->nab", ww, pair[idx])
                o["pair_tv" + tag] = 0.5 * np.abs(pm - pair_truth).sum((-1, -2)).mean()
        return o

    full = stat(np.arange(nrun))
    batches = np.array_split(np.arange(nrun), nb)
    jk = [stat(np.concatenate([b for j, b in enumerate(batches) if j != i])) for i in range(nb)]
    res = {}
    for k, v in full.items():
        vals = np.array([j[k] for j in jk])
        res[k] = float(v); res[k + "_se"] = float(math.sqrt((nb - 1) / nb * ((vals - vals.mean()) ** 2).sum()))
    return res


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    from twistroute.decode import twisted_smc, CouplingPolicy, fit_distance_kernel
    from twistroute.models import parallel_log_prob
    S = 2 if cname == "dyck" else 3
    den = make_model(mname, N, S)
    cl, route, truth = make_constraint(cname, den, N)
    pair_truth = truth.pair_marginals() if hasattr(truth, "pair_marginals") else None
    t0 = time.time()
    kern, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(0))
    kern_small, _ = fit_distance_kernel(den, N, a["pilot_small"], torch.Generator().manual_seed(1))
    pilot_s = time.time() - t0
    arms = [(f"fixed {N // T}/step", dict(T=T, schedule="fixed"), a["nrun"]) for T in (N, N // 2, N // 4, N // 8)]
    pol = lambda k, th, pr="bound", mx=False: CouplingPolicy(k, th, priority=pr, mixed=mx)
    arms += [(f"policy theta={th}", dict(T=N, schedule=pol(kern, th)), a["nrun"]) for th in THETAS]
    arms += [(f"small-pilot theta={th}", dict(T=N, schedule=pol(kern_small, th)), a["nrun"]) for th in (0.01, 0.03)]
    arms += [(f"{pr} theta={th}", dict(T=N, schedule=pol(kern, th, pr)), a["nrun"]) for pr in ("confident", "random") for th in (0.01, 0.03)]
    if a["nrun_mixed"] > 0:
        arms += [(f"mixed theta={th}", dict(T=N, schedule=pol(kern, th, mx=True)), a["nrun_mixed"]) for th in (0.0, 0.01, 0.03)]
        arms += [(f"policy-ref theta={th}", dict(T=N, schedule=pol(kern, th)), a["nrun_mixed"]) for th in (0.0, 0.01, 0.03)]
    ps = lambda xb, xa: parallel_log_prob(den, xb, xa)
    rows = []
    for name, kw, nrun in arms:
        t0 = time.time()
        out = twisted_smc(den, cl, route, N, kw["T"], nrun=nrun, K=a["K"], gen=torch.Generator().manual_seed(3),
                          schedule=kw["schedule"], defensive=0.25, return_particles=True, path_stat=ps)
        row = dict(model=mname, constraint=cname, N=N, arm=name, nrun=nrun, K=a["K"], log_pC=truth.log_pC,
                   seconds=round(time.time() - t0, 2), frac_valid=out["frac_valid_finite"],
                   **metrics(out, den, truth, N, S, a["K"], pair_truth))
        sch = kw["schedule"]
        row["route_passes"] = 0.0 if isinstance(sch, str) else sch.route_passes / max(sch.calls, 1)
        rows.append(row)
    meta = dict(model=mname, constraint=cname, N=N, pilot_seconds=round(pilot_s, 2),
                kernel_unscreened=kern.ku.tolist(), kernel_screened=kern.ks.tolist(),
                exact_mi=[0.0] + [den.pair_mi(0, d) for d in range(1, N)])
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--nrun", type=int, default=2000); ap.add_argument("--nrun-mixed", dest="nrun_mixed", type=int, default=500)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--pilot", type=int, default=2000)
    ap.add_argument("--pilot-small", dest="pilot_small", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=list(MODELS)); ap.add_argument("--constraints", nargs="+", default=list(CONSTRAINTS))
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default="compare.json")
    ap.add_argument("--resume", action="store_true", help="skip cells already present in --out")
    a = vars(ap.parse_args())
    jobs = [(m, c, N, a) for N in a["N"] for m in a["models"] for c in a["constraints"]]
    rows, metas, t0 = [], [], time.time()
    if a["resume"] and os.path.exists(a["out"]):          # keep finished cells; a cell is written only when complete
        prev = json.load(open(a["out"]))
        same = all(prev["args"].get(k) == a[k] for k in ("nrun", "nrun_mixed", "K", "pilot", "pilot_small"))
        if not same:
            raise SystemExit("--resume: the existing file was made with different settings")
        rows, metas = prev["rows"], prev["cells"]
        have = {(c["model"], c["constraint"], c["N"]) for c in metas}
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in have]
        print(f"resuming: {len(have)} cells kept, {len(jobs)} to run", flush=True)
    def collect(res):
        rows.extend(res[0]); metas.append(res[1])
        json.dump(dict(args=a, rows=rows, cells=metas, wall_seconds=round(time.time() - t0, 1)), open(a["out"], "w"), indent=1)
        print(f"[{time.time() - t0:7.0f}s] done {res[1]['model']} {res[1]['constraint']} N={res[1]['N']} "
              f"({sum(r['seconds'] for r in res[0]):.0f}s of arms)", flush=True)
    if a["workers"] <= 1:
        for j in jobs:
            collect(run_cell(j))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a["workers"]) as pool:
            for res in pool.imap_unordered(run_cell, jobs):
                collect(res)
    print("all cells done:", len(metas), "cells,", len(rows), "arms")


if __name__ == "__main__":
    main()
