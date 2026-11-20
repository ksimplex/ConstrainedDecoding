"""Where does the time of the schedule go over a whole decode? For one oracle cell, run a policy from the fully
masked state to completion and time each stage of every reveal: the step context (one constrained-marginals pass),
scoring, admission, and inside admission the forced-partner pull-in (one constrained-marginals pass each). Reports
steps (= denoiser calls), positions admitted per step and pull-in passes per step. The denoiser is not timed: these
are the schedule's own costs, on CPU. Tokens are drawn one at a time from the constrained marginals, outside the
timers, so every state stays feasible."""
import argparse, json, sys, os, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "studies"))
from pulib.cells import make_model, make_constraint, n_symbols
from pulib.summary import provenance
from twistroute.decode import CouplingPolicy, fit_distance_kernel, COUNT_EDGES, constrained_marginals
from twistroute.decode.policy import RevealState
from twistroute.decode.schedule import clamp_revealed


def decode(pol, den, cl, route, N, P, g):
    x = torch.full((P, N), den.mask_id); rows = torch.arange(P)
    acc = dict(context=0.0, score=0.0, admit=0.0, pull=0.0, pulls=0, steps=0, admitted=0.0)
    inner = pol._pull

    def pull(*a, **k):
        t = time.perf_counter(); inner(*a, **k); acc["pull"] += 1e3 * (time.perf_counter() - t); acc["pulls"] += 1
    pol._pull = pull
    while bool((x == den.mask_id).any()):
        masked = x == den.mask_id
        ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
        lw = cl.class_logw(den.logits(x)); st = RevealState(x, ycls, lw, den.mask_id, cl, route)
        with torch.no_grad():
            t0 = time.perf_counter(); s = pol.step_context(st, masked); t1 = time.perf_counter()
            sc = pol.score(s, g); t2 = time.perf_counter()
            chosen, _ = pol.admit(s, sc, g); t3 = time.perf_counter()
        acc["context"] += 1e3 * (t1 - t0); acc["score"] += 1e3 * (t2 - t1); acc["admit"] += 1e3 * (t3 - t2); acc["steps"] += 1
        live = masked.any(-1)
        acc["admitted"] += float(chosen[live].sum(-1).double().mean())
        for j in range(int(chosen.sum(-1).max())):                    # fill the chosen positions one at a time
            has = chosen.any(-1); i = chosen.to(torch.int8).argmax(-1)
            ycls = cl.token_class[torch.where(x == den.mask_id, torch.zeros_like(x), x)]
            mu = constrained_marginals(route, clamp_revealed(lw, x, ycls, den.mask_id))[rows, i]
            a = torch.multinomial(mu.clamp_min(1e-300), 1, generator=g).squeeze(-1)
            x[rows[has], i[has]] = a[has]; chosen[rows[has], i[has]] = False
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[64]); ap.add_argument("--model", default="hidden_0.85")
    ap.add_argument("--constraints", nargs="+", default=["window", "counting", "forbidden"])
    ap.add_argument("--P", type=int, default=16); ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--B", type=float, nargs="+", default=[0.03, 0.3]); ap.add_argument("--out", default="reveal_stages.json")
    a = ap.parse_args(); torch.set_num_threads(2)
    res = dict(args=vars(a), provenance=provenance(), rows=[])
    for N in a.N:
        for cname in a.constraints:
            den = make_model(a.model, N, n_symbols(cname)); cl, route, truth = make_constraint(cname, den, N)
            kern, _ = fit_distance_kernel(den, N, a.pilot, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
            arms = [("sequential lex", dict(max_set=1))]
            for B in a.B:
                arms += [(f"lex B={B}", dict(rate=B / N)), (f"lex B={B} +pull", dict(rate=B / N, pull_forced=8))]
            for name, kw in arms:
                tot = None
                for rep in range(a.reps):
                    acc = decode(CouplingPolicy(kern, 0.0, priority="lex", mixed=False, **kw), den, cl, route, N, a.P, torch.Generator().manual_seed(10 + rep))
                    tot = acc if tot is None else {k: tot[k] + v for k, v in acc.items()}
                n = a.reps; st = tot["steps"]
                row = dict(N=N, constraint=cname, arm=name, steps=st / n, admitted_per_step=tot["admitted"] / st, pull_passes_per_step=tot["pulls"] / st,
                           context_ms=tot["context"] / n, score_ms=tot["score"] / n, admit_other_ms=(tot["admit"] - tot["pull"]) / n, pull_ms=tot["pull"] / n,
                           schedule_ms=(tot["context"] + tot["score"] + tot["admit"]) / n)
                row["schedule_ms_per_step"] = row["schedule_ms"] / row["steps"]
                res["rows"].append(row); json.dump(res, open(a.out, "w"), indent=1)
                print(f"N={N} {cname:10s} {name:18s} steps {row['steps']:6.1f} admitted/step {row['admitted_per_step']:5.2f} pull passes/step {row['pull_passes_per_step']:4.2f} | "
                      f"whole decode ms: context {row['context_ms']:7.0f} score {row['score_ms']:6.0f} admit(other) {row['admit_other_ms']:7.0f} pull-in {row['pull_ms']:7.0f} total {row['schedule_ms']:7.0f} | per step {row['schedule_ms_per_step']:6.1f}", flush=True)


if __name__ == "__main__":
    main()
