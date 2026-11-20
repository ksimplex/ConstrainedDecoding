"""Planned orders and the chain proposal on a real masked diffusion language model (MDLM, OpenWebText).

  E  proposal   window constraints on three token classes, one token per step, resampling off, K particles:
                product proposal, chain proposal (ratio formula), chain proposal with matched marginals, under the
                uniformly random order and the `sum` order. Reported: log of the mean Z_hat, the spread of Z_hat over
                runs (relative standard deviation; lower is better), wall seconds. The three samplers of one
                schedule estimate the same p(C). No denoiser call is added by the chain law beyond the pilot samples.
  F  outputs    what the order does to the text: single-particle decodes (the proposal's own law) under the uniformly
                random order, `sum`, `lex` and left to right, with no constraint and with each window constraint
                (matched chain proposal). Reported per arm: distinct-token share, share of tokens equal to their left
                neighbour, and the mean log-probability per token of the samples scored along fresh uniformly random
                orders with no constraint (the model's reference law; higher is more typical).

--synthetic runs everything on the small synthetic denoiser on CPU (a code check, not a result).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twistroute import compile as C                                                                # noqa: E402
from twistroute.constraints import Constraint                                                      # noqa: E402
from twistroute.decode import CouplingPolicy, COUNT_EDGES                                          # noqa: E402  (the `lex` and `random` orders only)
from twistroute.resampling import NO_RESAMPLING                                                    # noqa: E402
from twistroute.sampler import weighted_sampler                                                    # noqa: E402
from twistroute.schedules import DistanceDependence, LeftToRight, RandomOrder, Sum, fit_distance_dependence   # noqa: E402
from twistroute.torch_routes import torch_route, TorchSubset                                       # noqa: E402
from twistroute.twists import ChainTwist, counted_pair_table                                       # noqa: E402
from mdlm_first_look import free_route, score_along, chunks                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--traces", type=int, default=32); ap.add_argument("--pilot-reps", dest="pilot_reps", type=int, default=4)
    ap.add_argument("--pair-pilot", dest="pair_pilot", type=int, default=64, help="native samples for the pair kernel of the chain law")
    ap.add_argument("--rows", type=int, default=96, help="rows per denoiser batch")
    ap.add_argument("--nrun", type=int, default=48); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--nout", type=int, default=48, help="sequences per arm in part F"); ap.add_argument("--score-orders", dest="score_orders", type=int, default=4)
    ap.add_argument("--parts", default="EF"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="mdlm_orders_output.json")
    a = ap.parse_args(); N = a.N; t00 = time.time()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res["args"] = vars(a)
    save = lambda: (json.dump(res, open(a.out + ".tmp", "w"), indent=1), os.replace(a.out + ".tmp", a.out))
    if a.synthetic:
        from twistroute.models import SyntheticMDLM
        dev = "cpu"; den = SyntheticMDLM(N, scale=3.0); base = den.classes(4)
    else:
        from twistroute.models import semantic_classes
        from twistroute.models.mdlm_owt import MDLM
        dev = "cuda"; den = MDLM(dev); base = semantic_classes(den.tok, den.V, den.mask_id).to(dev); res["model"] = den.info()
    G = lambda s: torch.Generator(device=dev).manual_seed(a.seed * 1000 + s)
    one = base.merge([0, 0, 0, 0]); fr = free_route(1, dev); cl = base.merge([0, 1, 2, 2]); A = 3
    free = Constraint(fr, family="finite automaton", cost="O(L |E|)", n_classes=1, name="no constraint")

    tabs = []                                                                   # the distance kernel of the schedule, as in mdlm_first_look
    for rep in range(a.pilot_reps):
        k = fit_distance_dependence(den, N, a.traces, G(rep), device=dev)
        tabs.append(k.table.double().cpu().numpy() if hasattr(k, "table") else np.stack([k.ku.numpy(), k.ks.numpy()], 1))
    kern = DistanceDependence.from_table(torch.tensor(np.mean(tabs, 0)), torch.tensor(list(COUNT_EDGES)))
    t0 = time.time(); Kp = counted_pair_table(den, cl, a.pair_pilot, G(50), length=N)
    res["pair_kernel"] = dict(table=Kp.cpu().tolist(), seconds=round(time.time() - t0, 1), samples=a.pair_pilot, denoiser_calls=N)
    save(); print(f"[{time.time() - t00:6.0f}s] kernels fitted; pair kernel {res['pair_kernel']['seconds']} s: {np.round(Kp.cpu().numpy(), 3).tolist()}", flush=True)

    x0 = torch.full((1, N), den.mask_id, dtype=torch.long, device=dev); lw0 = cl.class_logw(den.logits(x0)); w = max(8, N // 10); n2 = N // 2
    def window(pos):                                                            # at least k of class 2 on `pos`, k the smallest the product twist puts below 1e-3
        for k in range(1, len(pos)):
            d = C.dfa_at_least_k(2, k, A); rt = TorchSubset(torch_route(d, dev), pos, N, device=dev)
            if float(rt.logZ(lw0)[0]) <= math.log(1e-3) or k >= round(0.75 * len(pos)):
                break
        act = np.zeros(N, dtype=bool); act[pos] = True
        return Constraint(rt, family="subset of positions", cost="O(L |E|)", n_classes=A, length=N), k, d, act
    lo = N // 2 - w // 2; w2 = 2 * (w // 2); lo2 = N // 4 - w2 // 4
    cons = dict(window=list(range(lo, lo + w)), two_window=list(range(lo2, lo2 + w2 // 2)) + list(range(lo2 + n2, lo2 + n2 + w2 // 2)))
    built = {c: window(p) for c, p in cons.items()}
    pol = lambda pr: CouplingPolicy(kern, 0.0, priority=pr, mixed=False, max_set=1)
    ltr = lambda: LeftToRight(forced_first=True)

    if "E" in a.parts:
        res["proposal"] = res.get("proposal", {})
        for cname, (constraint, par, d, act) in built.items():
            tw = ChainTwist(d, Kp, active=act, device=dev, matching_sweeps=0); tm = ChainTwist(d, Kp, active=act, device=dev, matching_sweeps=30)
            for sname, mks in (("fixed 1/step", lambda: RandomOrder()), ("sum sequential", lambda: Sum(kern))):
                for pname, kw in (("product", dict()), ("chain", dict(twist=tw)), ("matched chain", dict(twist=tm))):
                    key = f"{cname}|{sname}|{pname}"
                    if key in res["proposal"]:
                        continue
                    t0 = time.time()
                    def go(n, i):
                        o = weighted_sampler(den, cl, constraint, N, N, runs=n, particles=a.K, schedule=mks(), resampling=NO_RESAMPLING, generator=G(300 + i), **kw)
                        return dict(logZ=np.asarray(o.log_Z_hat, dtype=np.float64))
                    try:
                        lz = chunks(go, a.nrun, max(1, a.rows // a.K))["logZ"]
                    except Exception as e:                                      # one failing arm must not end the job
                        res["proposal"][key + "|error"] = f"{type(e).__name__}: {str(e)[:300]}"; save(); print(f"[{time.time() - t00:6.0f}s] {key} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
                    fin = np.isfinite(lz); z = np.exp(lz[fin] - lz[fin].max())
                    res["proposal"][key] = dict(param=par, log_mean_Z=float(lz[fin].max() + math.log(z.mean())), mean_logZ=float(lz[fin].mean()), sd_logZ=float(lz[fin].std(ddof=1)),
                                                spread=float(z.std(ddof=1) / z.mean()), rel_se=float(z.std(ddof=1) / z.mean() / math.sqrt(len(z))), n=int(fin.sum()), seconds=round(time.time() - t0, 1))
                    save(); r = res["proposal"][key]
                    print(f"[{time.time() - t00:6.0f}s] {cname:11s} {sname:15s} {pname:14s} log Z {r['log_mean_Z']:8.3f} +- {r['rel_se']:.2f}  spread {r['spread']:.2f}  sd log Z {r['sd_logZ']:.2f}  {r['seconds']:.0f} s", flush=True)

    if "F" in a.parts:
        res["outputs"] = res.get("outputs", {}); rnd = lambda: pol("random")
        orders = (("fixed 1/step", lambda: RandomOrder()), ("sum sequential", lambda: Sum(kern)), ("lex sequential", lambda: pol("lex")), ("left to right", ltr))
        for cname in ("none", "window", "two_window"):
            if cname == "none":
                classes, constraint, kw = one, free, dict()
            else:
                constraint, par, d, act = built[cname]; tm = ChainTwist(d, Kp, active=act, device=dev, matching_sweeps=30); classes, kw = cl, dict(twist=tm)
            for oname, mks in orders:
                key = f"{cname}|{oname}"
                if key in res["outputs"]:
                    continue
                t0 = time.time()
                def go(n, i):
                    o = weighted_sampler(den, classes, constraint, N, N, runs=n, particles=1, schedule=mks(), resampling=NO_RESAMPLING, generator=G(700 + i), **kw)
                    return dict(tokens=np.asarray(o.tokens)[:, 0], logZ=np.asarray(o.log_Z_hat, dtype=np.float64))
                try:
                    o = chunks(go, a.nout, a.rows)
                except Exception as e:
                    res["outputs"][key + "|error"] = f"{type(e).__name__}: {str(e)[:300]}"; save(); print(f"[{time.time() - t00:6.0f}s] {key} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
                x = torch.as_tensor(o["tokens"], dtype=torch.long, device=dev); ok = np.isfinite(o["logZ"])
                lp = torch.zeros(len(x), dtype=torch.float64, device=dev)
                for r in range(a.score_orders):
                    for lo_ in range(0, len(x), a.rows):
                        lp[lo_:lo_ + a.rows] += score_along(den, one, fr, x[lo_:lo_ + a.rows], rnd(), G(900 + r)) / a.score_orders
                xs = x.cpu().numpy(); distinct = np.array([len(set(r_.tolist())) / N for r_ in xs]); rep = (xs[:, 1:] == xs[:, :-1]).mean(1); lpt = lp.cpu().numpy() / N
                se = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v)))
                ent = dict(n=int(len(xs)), satisfied=float(ok.mean()), distinct=float(distinct.mean()), distinct_se=se(distinct), repeat=float(rep.mean()), repeat_se=se(rep),
                           logp_per_token=float(lpt.mean()), logp_per_token_se=se(lpt), seconds=round(time.time() - t0, 1))
                if hasattr(den, "tok"):
                    ent["texts"] = [den.tok.decode(r_.tolist())[:600] for r_ in xs[:3]]
                res["outputs"][key] = ent; save()
                print(f"[{time.time() - t00:6.0f}s] {cname:11s} {oname:15s} distinct {ent['distinct']:.3f} +- {ent['distinct_se']:.3f}  repeat {ent['repeat']:.3f}  log p per token {ent['logp_per_token']:.3f} +- {ent['logp_per_token_se']:.3f}  satisfied {ent['satisfied']:.2f}", flush=True)
    print(f"done in {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
