"""Does exposure predict the variance of the weights?

Claim under test (docs/parallel_unmasking_status.md, section 32). When a step reveals the set R, the log-weight
increment is driven by how much the model's predictions at the positions j that stay masked move, weighted by the
constraint's stake there. To first order its spread is bounded by twice the step's exposure
    e = sum_{i in R} sum_{j still masked} sqrt(D_j * M(i, j)),
with D_j = KL(constrained marginal || model marginal) at j, in nats, and M the pilot kernel. If increments of
different steps are roughly uncorrelated, Var(log w_T) <= 4 E[sum_t e_t^2].

Measured, without resampling and one particle per run, for sequential and parallel schedules on each cell:
    per (particle, step)   the increment of the log-weight, the exposure e, and rival predictors computed from
                           the same state: coupling only (sum M(i, j)), stake revealed (sum_{i in R} D_i), the
                           linear form sum D_j M(i, j) (exposure added in quadrature over both i and j), and
                           exposure added in quadrature over the positions of R only
    step level             (particle, step) pairs binned by each predictor: spread of the increment per bin
    run level              Var(log w_T) over particles against E[sum_t e_t^2] and the rivals

usage: python exposure_check.py --N 256 --models hidden_0.85 --constraints copy_block window --n 500 --out exposure.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulib.arms import Cell                                                                       # noqa: E402
from pulib.cells import make_model, make_constraint                                               # noqa: E402
from pulib.summary import provenance                                                              # noqa: E402

PRED = ("exposure", "coupling", "stake", "linear", "exposure_pos")


class Probe:
    """Wraps a policy: after it has chosen a step's set, records the predictors for that set from the same state."""

    def __init__(self, inner, kernel):
        from twistroute.decode import DistanceKernel
        self.inner, self.kernel = inner, kernel
        self.root = DistanceKernel.from_table(kernel.table.sqrt(), kernel.edges)        # sqrt(M), entry by entry
        self.rec = []

    @torch.no_grad()
    def reveal(self, st, t, T, gen=None):
        from twistroute.decode import CouplingPolicy
        r, valid = self.inner.reveal(st, t, T, gen)
        masked = st.x == st.mask_id
        chosen = torch.zeros_like(masked)
        rows = torch.arange(masked.shape[0])[:, None].expand_as(r)
        chosen[rows[valid], r[valid]] = True
        chosen &= masked
        s = self.inner.step_context(st, masked)
        D = CouplingPolicy.distortion(s); D = torch.where(masked, D, torch.zeros_like(D))
        rest = (masked & ~chosen).double(); ch = chosen.double()
        a = self.root.apply(masked, D.sqrt() * rest)                       # per position i: sum_j sqrt(D_j M(i, j))
        self.rec.append(torch.stack([(a * ch).sum(-1),                                              # exposure
                                     (self.kernel.apply(masked, rest) * ch).sum(-1),                # coupling only
                                     (D * ch).sum(-1),                                              # stake revealed
                                     (self.kernel.apply(masked, D * rest) * ch).sum(-1),            # linear form
                                     (a * a * ch).sum(-1).sqrt(),               # exposure added in quadrature over R
                                     ch.sum(-1)]).float())
        return r, valid


def schedules(c):
    kg = c.kernel("graded")
    P = lambda pr, **kw: c.policy(kg, 0.0, pr, **kw)
    out = [("random sequential (base)", P("random", max_set=1, forced_first=False)), ("random sequential, forced first", P("random", max_set=1))]
    out += [(f"{pr} sequential", P(pr, max_set=1)) for pr in ("bound", "separator", "sum", "lex")]
    out += [(f"{pr} theta=0.0", P(pr)) for pr in ("separator", "sum", "lex")]
    out += [("lex theta=0.0 +pull", P("lex", pull_forced=8)), ("lex theta=0.0 +pull+guard", P("lex", pull_forced=8, guard=True))]
    out += [("sum B=0.03", P("sum", B=0.03)), ("lex B=0.03", P("lex", B=0.03)),
            ("lex B=0.03 +pull", P("lex", B=0.03, pull_forced=8)), ("lex B=0.03 +pull+guard", P("lex", B=0.03, pull_forced=8, guard=True))]
    return kg, out


def binned(x, inc, nb=10):
    """Spread of the increment in bins of a predictor: the zero bin, then quantile bins of the positive values."""
    out = []
    zero = x <= 0
    if zero.sum() > 20:
        out.append(dict(x=0.0, sd=float(inc[zero].std()), mean=float(inc[zero].mean()), n=int(zero.sum())))
    xp, ip = x[~zero], inc[~zero]
    if len(xp) > 20 * nb:
        edges = np.quantile(xp, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-12
        for a, b in zip(edges[:-1], edges[1:]):
            m = (xp >= a) & (xp < b)
            if m.sum() > 20:
                out.append(dict(x=float(xp[m].mean()), sd=float(ip[m].std()), mean=float(ip[m].mean()), n=int(m.sum())))
    return out


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    from twistroute.decode import twisted_smc
    S = 2 if cname == "dyck" else 3
    den = make_model(mname, N, S)
    cl, route, truth = make_constraint(cname, den, N)
    c = Cell(mname, cname, N, dict(a, nrun=a["n"], pilot_small=0), den, cl, route, truth)
    kg, scheds = schedules(c)
    rows = []
    for name, pol in scheds:
        if a.get("only") and not any(name.startswith(p) for p in a["only"]):
            continue
        t0 = time.time()
        probe = Probe(pol, kg)
        out = twisted_smc(den, cl, route, N, N, nrun=a["n"], K=1, gen=torch.Generator().manual_seed(c.seed("exposure", name)),
                          schedule=probe, ess_min=0.0, return_trace=True)
        tr = np.asarray(out["trace"], dtype=np.float64)[:, :, 0]                     # (steps, particles), cumulative
        rec = torch.stack(probe.rec).numpy().astype(np.float64)                      # (steps, 6, particles)
        T = min(len(tr), len(rec)); tr, rec = tr[:T], rec[:T]
        alive = np.isfinite(tr[-1])
        inc = np.diff(np.vstack([np.zeros((1, tr.shape[1])), tr]), axis=0)           # the first row carries a constant
        inc[0] -= np.nanmean(np.where(alive, inc[0], np.nan))                        # (the initial twist): remove it
        used = (rec[:, -1] > 0) & alive[None]                                         # steps that revealed something
        lw = tr[-1, alive] - truth.log_pC
        w = np.exp(lw)
        row = dict(model=mname, constraint=cname, N=N, schedule=name, n=int(a["n"]), dead=float(1 - alive.mean()),
                   calls=float(np.mean(out["steps"])), var_logw=float(lw.var(ddof=1)), spread=float(w.std(ddof=1) / w.mean()),
                   z_ratio=float(np.mean(np.where(alive, np.exp(tr[-1] - truth.log_pC), 0.0))), seconds=round(time.time() - t0, 1))
        for k, p in enumerate(PRED):
            x = rec[:, k]
            row[f"{p}_sum"] = float(x[:, alive].sum(0).mean())                       # E[sum_t x_t]
            row[f"{p}_sumsq"] = float((x[:, alive] ** 2).sum(0).mean())              # E[sum_t x_t^2]
            row[f"{p}_bins"] = binned(x[used], inc[used])
        # correlation between the increments of consecutive revealing steps (the run-level bound assumes it is small)
        ic = np.where(used, inc, np.nan)
        a0, a1 = ic[:-1].ravel(), ic[1:].ravel(); ok = np.isfinite(a0) & np.isfinite(a1)
        row["lag1_corr"] = float(np.corrcoef(a0[ok], a1[ok])[0, 1]) if ok.sum() > 50 else None
        row["sum_var_inc"] = float(np.nansum(np.nanvar(ic, axis=1)))                 # sum over steps of Var(increment)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256); ap.add_argument("--n", type=int, default=500, help="particles per schedule")
    ap.add_argument("--models", nargs="+", default=["hidden_0.85"]); ap.add_argument("--constraints", nargs="+", default=["copy_block", "copy_spread", "window", "two_window"])
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", default=None); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default="exposure.json")
    a = vars(ap.parse_args())
    jobs = [(m, c, a["N"], a) for m in a["models"] for c in a["constraints"]]
    rows, t0, prov = [], time.time(), provenance()
    def collect(res):
        rows.extend(res)
        json.dump(dict(args=a, provenance=prov, rows=rows), open(a["out"], "w"))
        print(f"[{time.time() - t0:6.0f}s] done {res[0]['model']} {res[0]['constraint']} N={res[0]['N']}", flush=True)
    if a["workers"] <= 1:
        for j in jobs:
            collect(run_cell(j))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a["workers"]) as pool:
            for res in pool.imap_unordered(run_cell, jobs):
                collect(res)


if __name__ == "__main__":
    main()
