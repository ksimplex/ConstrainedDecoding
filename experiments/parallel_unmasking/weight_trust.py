"""Do intermediate weights predict final weights? Runs without resampling, so every particle keeps its identity,
and compares each particle's log-weight part-way through with its final log-weight.

For each cell and schedule, at fractions of the run's steps:
  spearman   rank correlation between the log-weight so far and the final log-weight, over particles
  slope      regression slope of the final log-weight on the log-weight so far: the share of a weight difference
             seen now that survives to the end (the natural exponent for partial resampling)
  sd_now     spread of the log-weight so far; sd_final is the spread at the end
  hidden_share   share of the total final weight held by particles in the lower half by weight so far: the mass
             a resampling step at that point would tend to discard
A local constraint should give a high correlation early; a long-range one should not, if its intermediate
weights are mostly twist error that later steps revise.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as cp

FRACS = (0.1, 0.25, 0.5, 0.75, 0.9)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    from twistroute.decode import twisted_smc, CouplingPolicy, fit_distance_kernel, COUNT_EDGES
    S = 2 if cname == "dyck" else 3
    den = cp.make_model(mname, N, S); cl, route, truth = cp.make_constraint(cname, den, N)
    kern, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(cp.stream_seed(a["seed"], mname, cname, N, "pilot")),
                                  count_edges=COUNT_EDGES)
    rows = []
    for sname, sch in (("fixed 1/step", "fixed"), ("separator theta=0.0", CouplingPolicy(kern, 0.0, priority="separator", mixed=False)),
                       ("lex theta=0.0", CouplingPolicy(kern, 0.0, priority="lex", mixed=False))):
        out = twisted_smc(den, cl, route, N, N, nrun=a["n"], K=1, gen=torch.Generator().manual_seed(cp.stream_seed(a["seed"], mname, cname, N, sname)),
                          schedule=sch, ess_min=0.0, return_trace=True)
        tr = np.asarray(out["trace"], dtype=np.float64)[:, :, 0]            # (steps, particles): cumulative log-weights
        T = tr.shape[0]; fin = tr[-1]; ok = np.isfinite(fin)
        row = dict(model=mname, constraint=cname, N=N, schedule=sname, steps=float(np.mean(out["steps"])), trace_len=T,
                   n=int(ok.sum()), sd_final=float(fin[ok].std()), log_pC=truth.log_pC, z_ratio=float(np.exp(out["logZ"] - truth.log_pC).mean()))
        for f in FRACS:
            t = min(max(int(round(f * T)) - 1, 0), T - 1); now = tr[t]; m = ok & np.isfinite(now)
            v = now[m].var()
            row[f"spearman_{f}"] = spearman(now[m], fin[m]) if v > 1e-18 else float("nan")
            row[f"slope_{f}"] = float(np.cov(now[m], fin[m])[0, 1] / v) if v > 1e-18 else float("nan")
            row[f"sd_now_{f}"] = float(now[m].std())
            wf = np.exp(fin[m] - fin[m].max()); low = now[m] < np.median(now[m])
            row[f"hidden_share_{f}"] = float(wf[low].sum() / wf.sum())   # final weight held by the lower half by weight so far
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--models", nargs="+", default=["chain_0.6", "chain_0.85", "hidden_0.85", "segments_0.85"])
    ap.add_argument("--constraints", nargs="+", default=["forbidden", "counting", "copy", "dyck"])
    ap.add_argument("--pilot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out", default="weight_trust.json")
    args = ap.parse_args()
    a = dict(n=args.n, pilot=args.pilot, seed=args.seed)
    jobs = [(m, c, args.N, a) for m in args.models for c in args.constraints]
    rows, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_cell, jobs):
            rows += res
            print(f"[{time.time() - t0:7.0f}s] done {res[0]['model']} {res[0]['constraint']}", flush=True)
            json.dump(dict(args=vars(args), rows=rows), open(args.out, "w"), indent=1)
    print("all cells done")


if __name__ == "__main__":
    main()
