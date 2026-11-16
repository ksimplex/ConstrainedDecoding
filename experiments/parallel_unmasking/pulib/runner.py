"""Running the comparison: one cell at a time (`run_cell`), and the command line with resume (`main`)."""
import argparse, json, math, os, sys, time
import numpy as np
import torch

from .arms import ARM_SETS, Cell
from .cells import MODELS, CONSTRAINTS, make_model, make_constraint
from .metrics import metrics, Footprint


def run_cell(job):
    """Run every arm of one cell. job = (model, constraint, N, args). Returns (rows, cell meta, per-run log Z)."""
    mname, cname, N, a = job
    torch.set_num_threads(1)
    foot = Footprint()
    from twistroute.decode import twisted_smc
    from twistroute.models import parallel_log_prob
    S = 2 if cname == "dyck" else 3
    den = make_model(mname, N, S)
    cl, route, truth = make_constraint(cname, den, N)
    pair_truth = truth.pair_marginals() if hasattr(truth, "pair_marginals") else None
    cell = Cell(mname, cname, N, a, den, cl, route, truth)
    t0 = time.time()
    kern = cell.kernel()                           # always fitted: the cell meta records it
    if a["arms"] == "full":
        cell.kernel("small")
    pilot_s = time.time() - t0
    arms = ARM_SETS[a["arms"]](cell)
    if a.get("only"):                          # keep the arms whose name starts with one of the given prefixes
        arms = [x for x in arms if any(x.name.startswith(p) for p in a["only"])]
    ps = lambda xb, xa: parallel_log_prob(den, xb, xa)
    rows, zlog = [], {}
    part = os.path.join(a["partial"], f"{mname}__{cname}__{N}") if a.get("partial") else None
    if part and os.path.exists(part + ".json"):                       # arms finished before an interruption
        rows = json.load(open(part + ".json")); zlog = dict(np.load(part + ".npz"))
    done = {r["arm"] for r in rows}
    for arm in arms:
        name, nrun, sch = arm.name, arm.nrun, arm.schedule
        if name in done:
            continue
        t0 = time.time()
        K = arm.K or a["K"]; smc = dict(defensive=0.25); smc.update(arm.smc)
        out = twisted_smc(den, cl, route, N, arm.T, nrun=nrun, K=K, gen=torch.Generator().manual_seed(cell.seed(name)),
                          schedule=sch, return_particles=True, path_stat=ps, **smc)
        zlog[f"{mname}|{cname}|{N}|{name}"] = (np.asarray(out["logZ"], dtype=np.float64) - truth.log_pC).astype(np.float32)
        row = dict(model=mname, constraint=cname, N=N, arm=name, nrun=nrun, K=K, log_pC=truth.log_pC,
                   n_res=float(np.mean(out["n_res"])),               # resampling events per run
                   seed=cell.seed(name),
                   seconds=round(time.time() - t0, 2), frac_valid=out["frac_valid_finite"],
                   **metrics(out, den, truth, N, S, K, pair_truth))
        row["route_passes"] = 0.0 if isinstance(sch, str) else sch.route_passes / max(sch.calls, 1)
        if getattr(sch, "particle_calls", 0):          # mean summed kernel load per particle path: the predicted E[r]
            row["load_per_path"] = sch.load_sum * sch.calls / sch.particle_calls
        rows.append(row)
        if part:                                                      # each arm has its own random stream, so a
            json.dump(rows, open(part + ".tmp", "w"))                 # cell resumed arm by arm equals an uninterrupted one
            np.savez_compressed(part + ".tmp.npz", **zlog)
            os.replace(part + ".tmp", part + ".json"); os.replace(part + ".tmp.npz", part + ".npz")
    order = {arm.name: i for i, arm in enumerate(arms)}
    rows.sort(key=lambda r: order[r["arm"]])
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)   # MB: bytes on macOS, kB on Linux
    meta = dict(model=mname, constraint=cname, N=N, pilot_seconds=round(pilot_s, 2), peak_rss_mb=round(rss, 1), peak_footprint_mb=round(foot.peak, 1),
                resampling_gate=cell.gate_info,
                kernel_unscreened=kern.ku.tolist(), kernel_screened=kern.ks.tolist(),
                exact_mi=[0.0] + [den.pair_mi(0, d) for d in range(1, N)], constraint_info=getattr(truth, "info", {}))
    on = next((r for r in rows if r["arm"] == "fixed 1/step"), None)
    off = next((r for r in rows if r["arm"] == "fixed 1/step, no resampling"), None)
    if on is None or off is None:              # a filtered run without both controls
        return rows, meta, zlog
    meta["control_check"] = dict(on=on["z_ratio"], on_se=on["z_ratio_se"], off=off["z_ratio"], off_se=off["z_ratio_se"],
                                 on_over_off=on["z_ratio"] / off["z_ratio"],
                                 gap_in_se=(on["z_ratio"] - off["z_ratio"]) / math.hypot(on["z_ratio_se"], off["z_ratio_se"]),
                                 se_unreliable=bool(on["z_se_unreliable"] or off["z_se_unreliable"]))
    return rows, meta, zlog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--nrun", type=int, default=2000); ap.add_argument("--nrun-mixed", dest="nrun_mixed", type=int, default=500)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--pilot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="base seed; every cell and arm derives its own stream from it")
    ap.add_argument("--pilot-small", dest="pilot_small", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=list(MODELS)); ap.add_argument("--constraints", nargs="+", default=list(CONSTRAINTS))
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default="compare.json")
    ap.add_argument("--resume", action="store_true", help="skip cells already present in --out; within a cell, skip finished arms")
    ap.add_argument("--arms", choices=tuple(ARM_SETS), default="full",
                    help="core: 10 arms for long sequences; order: the controls and every priority score at --order-theta; "
                         "budget: whole-sequence budgets with graded and two-bin kernels, and dilated baselines")
    ap.add_argument("--order-theta", dest="order_theta", type=float, default=0.0)
    ap.add_argument("--resample-ks", dest="resample_ks", type=int, nargs="+", default=None)
    ap.add_argument("--resample-modes", dest="resample_modes", nargs="+", default=None, help="subset of the resampling modes, by name")
    ap.add_argument("--only", nargs="+", default=None, help="run only the arms whose name starts with one of these prefixes")
    a = vars(ap.parse_args())
    a["partial"] = os.path.splitext(a["out"])[0] + ".partial" if a["resume"] else None
    if a["partial"]:
        os.makedirs(a["partial"], exist_ok=True)
    jobs = [(m, c, N, a) for N in a["N"] for m in a["models"] for c in a["constraints"]]
    rows, metas, zall, t0 = [], [], {}, time.time()
    zpath = os.path.splitext(a["out"])[0] + ".logZ.npz"
    if a["resume"] and os.path.exists(a["out"]):          # keep finished cells; a cell is written only when complete
        prev = json.load(open(a["out"]))
        same = all(prev["args"].get(k, a[k]) == a[k] for k in ("nrun", "nrun_mixed", "K", "pilot", "pilot_small", "seed", "arms"))
        if not same:
            raise SystemExit("--resume: the existing file was made with different settings")
        rows, metas = prev["rows"], prev["cells"]
        if os.path.exists(zpath):
            zall = dict(np.load(zpath))
        have = {(c["model"], c["constraint"], c["N"]) for c in metas}
        jobs = [j for j in jobs if (j[0], j[1], j[2]) not in have]
        print(f"resuming: {len(have)} cells kept, {len(jobs)} to run", flush=True)
    def collect(res):
        rows.extend(res[0]); metas.append(res[1]); zall.update(res[2])
        np.savez_compressed(zpath + ".tmp.npz", **zall); os.replace(zpath + ".tmp.npz", zpath)
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
