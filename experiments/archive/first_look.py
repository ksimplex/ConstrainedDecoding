"""
First look at the coupling policy on oracle chains (CPU, small). For each (model, constraint) cell it runs
    the existing fixed schedule at 1, 2, 4 and 8 reveals per step, and
    the coupling policy over a grid of thresholds theta, with and without the mixed-path term,
and reports, against the exact constrained target (models.oracle.ConstrainedTruth):
    steps     mean number of reveal steps per particle (= denoiser calls)
    Z ratio   mean Z_hat / exact p(C), with its standard error (1 means the schedule's law has the same p(C))
    marg TV   mean over positions of the TV between the weighted class marginals and the exact ones
    sd r      weighted standard deviation of the parallel-reveal gap r over final particles
    ESS       mean of the minimum effective sample size fraction over the run
Results go to a JSON file; nothing here is tuned.
"""
import argparse, json, math, time
import numpy as np
import torch
from twistroute import compile as C
from twistroute.decode import twisted_smc, CouplingPolicy, fit_distance_kernel
from twistroute.models import ConstrainedTruth, lazy_chain, noisy_hmm, parallel_log_prob
from twistroute.torch_routes import torch_route


def summarise(out, den, truth, N, A, K):
    logZ = out["logZ"]; z = np.exp(logZ - truth.log_pC)
    lw = out["logw"]; w = np.exp(lw - lw.max(1, keepdims=True)); w = w / w.sum(1, keepdims=True)      # (nrun, K)
    run_w = np.exp(logZ - logZ.max())[:, None] * w                                                    # Z_hat-weighted
    cls = out["classes"]
    marg = np.stack([(run_w[..., None] * (cls == a)).sum((0, 1)) for a in range(A)], -1) / run_w.sum()
    r = den.log_joint(torch.tensor(out["tokens"]).view(-1, N)).numpy().reshape(lw.shape) - out["path_stat"]
    rw = run_w / run_w.sum(); rm = (rw * r).sum()
    return dict(steps=float(out["steps"].mean()), z_ratio=float(z.mean()), z_se=float(z.std(ddof=1) / math.sqrt(len(z))),
                marg_tv=float(0.5 * np.abs(marg - truth.marginals).sum(-1).mean()),
                sd_r=float(math.sqrt(max((rw * (r - rm) ** 2).sum(), 0.0))), mean_r=float(rm),
                ess=float((out["ess_min"] / K).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24); ap.add_argument("--nrun", type=int, default=1000)
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--out", default="first_look.json")
    ap.add_argument("--pilot", type=int, default=200)
    a = ap.parse_args()
    N, S, K = a.N, 3, a.K
    models = {"chain lam=0.3": lambda: lazy_chain(N, S, 0.3, seed=1, concentration=2.0),
              "chain lam=0.6": lambda: lazy_chain(N, S, 0.6, seed=1, concentration=2.0),
              "chain lam=0.8": lambda: lazy_chain(N, S, 0.8, seed=1, concentration=2.0),
              "hmm lam=0.8 noise=0.3": lambda: noisy_hmm(N, S, 0.8, 0.3, seed=1, concentration=2.0)}
    cons = {"exactly_k": lambda: C.dfa_exactly_k(0, N // 3, S), "forbidden": lambda: C.dfa_forbidden([(0, 0), (1, 2)], S)}
    rows = []
    for mname, mk in models.items():
        den = mk(); cl = den.classes()
        kern, _ = fit_distance_kernel(den, N, a.pilot, torch.Generator().manual_seed(0))
        for cname, ck in cons.items():
            d = ck(); route = torch_route(d); truth = ConstrainedTruth(den, cl, d)
            ps = lambda xb, xa: parallel_log_prob(den, xb, xa)
            arms = [(f"fixed {N // T}/step", dict(T=T, schedule="fixed")) for T in (N, N // 2, N // 4, N // 8)]
            for mixed in (False, True):
                for th in (0.0, 0.01, 0.03, 0.1, 0.3):
                    arms.append((f"policy theta={th} mixed={mixed}",
                                 dict(T=N, schedule=CouplingPolicy(kern, th, priority="bound", mixed=mixed))))
            for aname, kw in arms:
                t0 = time.time()
                out = twisted_smc(den, cl, route, N, kw["T"], nrun=a.nrun, K=K, gen=torch.Generator().manual_seed(3),
                                  schedule=kw["schedule"], defensive=0.25, return_particles=True, path_stat=ps)
                row = dict(model=mname, constraint=cname, arm=aname, log_pC=truth.log_pC, seconds=round(time.time() - t0, 1),
                           **summarise(out, den, truth, N, S, K))
                if not isinstance(kw["schedule"], str):
                    row["route_passes_per_step"] = kw["schedule"].route_passes / max(kw["schedule"].calls, 1)
                rows.append(row)
                print(f"{mname:22s} {cname:10s} {aname:32s} steps {row['steps']:5.1f}  Z {row['z_ratio']:.3f}±{row['z_se']:.3f}"
                      f"  margTV {row['marg_tv']:.4f}  sd_r {row['sd_r']:.3f}  ESS {row['ess']:.2f}  {row['seconds']}s", flush=True)
            json.dump(dict(args=vars(a), kernel_note="pilot kernel per model, 200 traces", rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
