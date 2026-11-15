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
    top1_share     share of the sum of Z_hat carried by the top 1% of runs
    hill_k         Hill estimate of the upper-tail index of Z_hat over runs (top 2%); above 1/2 the variance is not
                   finite and z_ratio_se is flagged as unreliable (z_se_unreliable)

Guards against hidden mass (notes Sec. 6). Every arm uses defensive resampling (eps = 0.25) except the arm
"fixed 1/step, no resampling", which cannot hide mass. Each cell records `control_check`: mean Z_hat of the
one-per-step arm with and without resampling. They estimate the same p(C); a gap is the truth-free sign that
resampling is hiding mass. Per-run log(Z_hat / p(C)) of every arm is saved to <out>.logZ.npz.
"""
import sys
import argparse, json, math, os, time, zlib
import numpy as np
import torch

THETAS = (0.0, 0.003, 0.01, 0.03, 0.1)
BUDGETS = (0.01, 0.03, 0.1, 0.3, 1.0)      # whole-sequence error budgets in nats; the policy gets rate = B / N
MODELS = ("chain_0.6", "chain_0.85", "hidden_0.85", "segments_0.85")
THICK_WIDTHS = (1, 2, 3, 4, 6)
THICK_SPACINGS = (8, 16, 24, 32, 48, 64)
RESAMPLE_KS = (4, 16, 64, 256)
INTERACT_BUDGETS = (0.03, 0.3)
INTERACT_ORDERS = ("separator", "sum", "weighted", "lex", "bound", "random", "least_tie")
CONSTRAINTS = ("forbidden", "counting", "copy", "dyck")


def stream_seed(base, *parts):
    """A generator seed for one (model, constraint, N, arm, purpose) stream. Stable across processes and
    platforms (crc32, not hash()), so no two cells or arms share random numbers and a run is reproducible."""
    return zlib.crc32("|".join(str(p) for p in (base, *parts)).encode()) & 0x7FFFFFFF


def make_model(name, N, S):
    from twistroute.models import lazy_chain, noisy_hmm, repel_chain
    if name == "repel_0.8":                               # neighbours anticorrelated, at 80% of the admissible strength
        pi = lazy_chain(2, S, 0.0, seed=1, concentration=2.0).pi.numpy()
        return repel_chain(N, S, 0.8 * pi.min() / (1 - pi.min()), seed=1, concentration=2.0)
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
    from twistroute.torch_routes import TorchChain, TorchCopy, TorchSubset
    cl = den.classes()
    if name == "copy":
        return cl, build_route(dict(kind="copy", A=den.S, pairs=N // 2, N=N)), CopyTruth(den, cl)
    if name in ("copy_spread", "copy_block"):             # copy on about 10% of the positions: N / 20 of the N / 2 pairs
        n, m = N // 2, max(1, N // 20)
        idx = list(range(0, n, n // m))[:m] if name == "copy_spread" else list(range(n // 2 - m // 2, n // 2 - m // 2 + m))
        truth = CopyTruth(den, cl, pairs=idx)
        truth.info = dict(pairs=len(idx), first=idx[:3], constrained_fraction=2 * len(idx) / N)
        return cl, TorchCopy([(i, n + i) for i in idx], N, den.S), truth
    if name in ("window", "two_window"):                  # at least k of a tenth of the positions are class 0: one contiguous
        import numpy as np                                # window, or two distant windows sharing the count (every position
        w = max(8, N // 10)                               # of one is tied to every position of the other until both are done)
        act = np.zeros(N, dtype=bool)
        if name == "window":
            lo = N // 2 - w // 2; act[lo:lo + w] = True
        else:
            w = 2 * (w // 2); lo = N // 4 - w // 4
            act[lo:lo + w // 2] = True; act[lo + N // 2:lo + N // 2 + w // 2] = True
        a, b = 0, max(1, round(0.75 * w))                 # smallest k with p(C) <= 1e-3 (the constraint clearly binds),
                                                          # capped at 3/4 of the window so the window is never fully forced
        while b - a > 1:
            k = (a + b) // 2
            if ConstrainedTruth(den, cl, C.dfa_at_least_k(0, k, den.S), active=act).log_pC > math.log(1e-3):
                a = k
            else:
                b = k
        d = C.dfa_at_least_k(0, b, den.S)
        truth = ConstrainedTruth(den, cl, d, active=act)
        pos = [int(i) for i in np.nonzero(act)[0]]
        truth.info = dict(window=[pos[0], pos[-1] + 1], positions=len(pos), k=b, p_C=math.exp(truth.log_pC),
                          expected_count=float(w * (den.pi @ den.emit[0])[0]), constrained_fraction=w / N)
        return cl, TorchSubset(TorchChain(d), pos, N), truth
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
    zs = np.sort(zrel)[::-1]; m = min(max(10, len(zs) // 50), len(zs) - 1)                  # upper tail of Z_hat over runs
    tail = dict(top1_share=float(zs[:max(1, len(zs) // 100)].sum() / zs.sum()),
                hill_k=float(np.mean(np.log(zs[:m] / zs[m]))) if zs[m] > 0 else float("nan"))

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
    nb = min(nb, nrun)
    batches = np.array_split(np.arange(nrun), nb)
    jk = [stat(np.concatenate([b for j, b in enumerate(batches) if j != i])) for i in range(nb)]
    res = {}
    for k, v in full.items():
        vals = np.array([j[k] for j in jk])
        res[k] = float(v); res[k + "_se"] = float(math.sqrt((nb - 1) / nb * ((vals - vals.mean()) ** 2).sum()))
    res["z_ratio_se"] = float(zrel.std(ddof=1) / math.sqrt(nrun))         # runs are independent: the plain standard error
    res.update(tail)                                                      # is exact here and steadier than 10 batches
    res["z_se_unreliable"] = bool(not tail["hill_k"] <= 0.5)               # tail index above 1/2: no finite variance
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
    kern, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, "pilot")))
    kern_small = None if a["arms"] in ("core", "order", "budget", "interact", "resample", "thick") else fit_distance_kernel(
        den, N, a["pilot_small"], torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, "pilot_small")))[0]
    pilot_s = time.time() - t0
    core = a["arms"] in ("core", "order", "budget", "interact", "resample", "thick")   # core: the controls, two naive parallel schedules, the theta sweep
    arms = [(f"fixed {N // T}/step", dict(T=T, schedule="fixed"), a["nrun"]) for T in ((N, N // 8, N // 16) if core else (N, N // 2, N // 4, N // 8))]
    arms += [("fixed 1/step, no resampling", dict(T=N, schedule="fixed", smc=dict(ess_min=0.0)), a["nrun"])]
    pol = lambda k, th, pr="bound", mx=False: CouplingPolicy(k, th, priority=pr, mixed=mx)
    if a["arms"] == "order":                  # which position next: every priority score at one threshold
        th = a["order_theta"]
        arms = [arms[0], ("fixed 1/step, no resampling", dict(T=N, schedule="fixed", smc=dict(ess_min=0.0)), a["nrun"])]
        arms += [(f"{pr} theta={th}", dict(T=N, schedule=pol(kern, th, pr)), a["nrun"])
                 for pr in ("bound", "random", "confident", "separator", "sum", "weighted", "lex", "least", "least_random", "least_tie")]
    elif a["arms"] == "budget":               # whole-sequence budgets, graded against two-bin kernel, fixed dilated baselines
        from twistroute.decode import DilatedSchedule, COUNT_EDGES
        kern_g, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, "pilot")),
                                        count_edges=COUNT_EDGES)       # same pilot traces as `kern`
        rated = lambda k, B, pr: CouplingPolicy(k, 0.0, priority=pr, mixed=False, rate=B / N)
        arms = [(f"fixed 8/step", dict(T=N // 8, schedule="fixed"), a["nrun"])]
        arms += [(f"dilated base={b}", dict(T=N, schedule=DilatedSchedule(b)), a["nrun"]) for b in (2, 4)]
        arms += [(f"{pr} theta=0.0 graded", dict(T=N, schedule=pol(kern_g, 0.0, pr)), a["nrun"]) for pr in ("separator", "bound")]
        arms += [(f"{pr} B={B} graded", dict(T=N, schedule=rated(kern_g, B, pr)), a["nrun"])
                 for pr in ("separator", "bound", "random", "least_random") for B in BUDGETS]
        arms += [(f"separator B={B} two-bin", dict(T=N, schedule=rated(kern, B, "separator")), a["nrun"]) for B in BUDGETS]
    elif a["arms"] == "interact":             # how the order interacts with the constraint: every score, exact threshold
        from twistroute.decode import COUNT_EDGES                 # and two budgets, graded kernel, resampling off throughout
        kern_g, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, "pilot")),
                                        count_edges=COUNT_EDGES)
        off = dict(ess_min=0.0)
        mk = lambda pr, B: CouplingPolicy(kern_g, 0.0, priority=pr, mixed=False, rate=None if B is None else B / N)
        arms = [(f"{pr} " + ("theta=0.0" if B is None else f"B={B}"), dict(T=N, schedule=mk(pr, B), smc=off), a["nrun"])
                for B in (None,) + INTERACT_BUDGETS for pr in INTERACT_ORDERS]
        arms += [("fixed 1/step, no resampling", dict(T=N, schedule="fixed", smc=off), a["nrun"])]
    elif a["arms"] == "thick":                # do thick separators first beat the greedy rule's single-token bisection?
        from twistroute.decode import DilatedSchedule, ThickSeparatorSchedule, COUNT_EDGES      # no resampling: r is the measure
        kern_g, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, "pilot")),
                                        count_edges=COUNT_EDGES)
        off = dict(ess_min=0.0)
        arms = [(f"separator B={B} graded", dict(T=N, schedule=CouplingPolicy(kern_g, 0.0, priority="separator", mixed=False, rate=B / N), smc=off), a["nrun"])
                for B in BUDGETS]
        arms += [("dilated base=2", dict(T=N, schedule=DilatedSchedule(2), smc=off), a["nrun"])]
        arms += [(f"thick w={w} spacing={sp}", dict(T=N, schedule=ThickSeparatorSchedule(w, sp), smc=off), a["nrun"])
                 for w in THICK_WIDTHS for sp in THICK_SPACINGS if sp >= 4 * w]
    elif a["arms"] == "resample":             # does resampling hurt only when K is small? Total particles nrun * K held
        from twistroute.decode import COUNT_EDGES                 # fixed across K; exact schedules only, so r = 0 throughout
        kern_g, _ = fit_distance_kernel(den, N, a["pilot"], torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, "pilot")),
                                        count_edges=COUNT_EDGES)
        modes = (("off", dict(ess_min=0.0)), ("defensive", dict()), ("plain", dict(defensive=0.0)),
                 ("systematic", dict(defensive=0.0, resampler="systematic")),
                 ("power 0.5", dict(defensive=0.0, resample_power=0.5)), ("power 0.25", dict(defensive=0.0, resample_power=0.25)),
                 ("power 0.5 systematic", dict(defensive=0.0, resample_power=0.5, resampler="systematic")))
        arms = [(f"{sn} K={K} resampling {rn}", dict(T=N, schedule=("fixed" if sn.startswith("fixed") else pol(kern_g, 0.0, "separator")), smc=smc, K=K),
                 max(a["nrun"] * a["K"] // K, 2))
                for K in (a.get("resample_ks") or RESAMPLE_KS) for sn in ("fixed 1/step", "separator theta=0.0") for rn, smc in modes]
    else:
        arms += [(f"policy theta={th}", dict(T=N, schedule=pol(kern, th)), a["nrun"]) for th in THETAS]
    if not core:
        arms += [(f"small-pilot theta={th}", dict(T=N, schedule=pol(kern_small, th)), a["nrun"]) for th in (0.01, 0.03)]
        arms += [(f"{pr} theta={th}", dict(T=N, schedule=pol(kern, th, pr)), a["nrun"]) for pr in ("confident", "random") for th in (0.01, 0.03)]
    if a["arms"] not in ("order", "budget", "interact", "resample", "thick"):
        arms += [("random theta=0.0", dict(T=N, schedule=pol(kern, 0.0, "random")), a["nrun"])]   # same sets rule, order not chosen
    if a["nrun_mixed"] > 0 and not core:
        arms += [(f"mixed theta={th}", dict(T=N, schedule=pol(kern, th, mx=True)), a["nrun_mixed"]) for th in (0.0, 0.01, 0.03)]
        arms += [(f"policy-ref theta={th}", dict(T=N, schedule=pol(kern, th)), a["nrun_mixed"]) for th in (0.0, 0.01, 0.03)]
    if a.get("only"):                          # keep the arms whose name starts with one of the given prefixes
        arms = [x for x in arms if any(x[0].startswith(p) for p in a["only"])]
    ps = lambda xb, xa: parallel_log_prob(den, xb, xa)
    rows, zlog = [], {}
    part = os.path.join(a["partial"], f"{mname}__{cname}__{N}") if a.get("partial") else None
    if part and os.path.exists(part + ".json"):                       # arms finished before an interruption
        rows = json.load(open(part + ".json")); zlog = dict(np.load(part + ".npz"))
    done = {r["arm"] for r in rows}
    for name, kw, nrun in arms:
        if name in done:
            continue
        t0 = time.time()
        K = kw.get("K", a["K"]); smc = dict(defensive=0.25); smc.update(kw.get("smc", {}))
        out = twisted_smc(den, cl, route, N, kw["T"], nrun=nrun, K=K, gen=torch.Generator().manual_seed(stream_seed(a["seed"], mname, cname, N, name)),
                          schedule=kw["schedule"], return_particles=True, path_stat=ps, **smc)
        zlog[f"{mname}|{cname}|{N}|{name}"] = (np.asarray(out["logZ"], dtype=np.float64) - truth.log_pC).astype(np.float32)
        row = dict(model=mname, constraint=cname, N=N, arm=name, nrun=nrun, K=K, log_pC=truth.log_pC,
                   n_res=float(np.mean(out["n_res"])),               # resampling events per run
                   seed=stream_seed(a["seed"], mname, cname, N, name),
                   seconds=round(time.time() - t0, 2), frac_valid=out["frac_valid_finite"],
                   **metrics(out, den, truth, N, S, K, pair_truth))
        sch = kw["schedule"]
        row["route_passes"] = 0.0 if isinstance(sch, str) else sch.route_passes / max(sch.calls, 1)
        if getattr(sch, "particle_calls", 0):          # mean summed kernel load per particle path: the predicted E[r]
            row["load_per_path"] = sch.load_sum * sch.calls / sch.particle_calls
        rows.append(row)
        if part:                                                      # each arm has its own random stream, so a
            json.dump(rows, open(part + ".tmp", "w"))                 # cell resumed arm by arm equals an uninterrupted one
            np.savez_compressed(part + ".tmp.npz", **zlog)
            os.replace(part + ".tmp", part + ".json"); os.replace(part + ".tmp.npz", part + ".npz")
    order = {name: i for i, (name, _, _) in enumerate(arms)}
    rows.sort(key=lambda r: order[r["arm"]])
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)   # MB: bytes on macOS, kB on Linux
    meta = dict(model=mname, constraint=cname, N=N, pilot_seconds=round(pilot_s, 2), peak_rss_mb=round(rss, 1),
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
    ap.add_argument("--arms", choices=("full", "core", "order", "budget", "interact", "resample", "thick"), default="full",
                    help="core: 10 arms for long sequences; order: the controls and every priority score at --order-theta; "
                         "budget: whole-sequence budgets with graded and two-bin kernels, and dilated baselines")
    ap.add_argument("--order-theta", dest="order_theta", type=float, default=0.0)
    ap.add_argument("--resample-ks", dest="resample_ks", type=int, nargs="+", default=None)
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


if __name__ == "__main__":
    main()
