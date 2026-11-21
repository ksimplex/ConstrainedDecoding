"""First look at parallel unmasking on a real masked diffusion language model (MDLM, OpenWebText), with no exact
answer to compare against. Four parts, each written to the output JSON as it finishes:

  A  kernel     the pilot dependence kernel: coupling (nats) by distance and by revealed tokens in between
  B  audit      unconstrained schedules: denoiser calls against the co-reveal gap. For every step the gap is
                log(sequential joint of the step's tokens, revealed one at a time in index order) minus
                log(product of their conditionals), so its mean is minus a KL and is well defined even though a
                trained denoiser's conditionals are not consistent across orders. Reported: steps, mean, spread and
                upper quantiles of the per-sequence gap (the audit distribution).
  C  orders     order inconsistency: the same sequences scored one token at a time along a random order, a second
                random order, the separator order and left to right
  D  constrained  window-count and class-copy constraints on three token classes, resampling off: random
                sequential (the baseline), smart sequential and smart parallel; log Z_hat, its spread, steps

--synthetic runs everything on the small synthetic denoiser on CPU (a code check, not a result).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from twistroute import compile as C
from twistroute.constraints import Constraint
from twistroute.decode import CouplingPolicy, COUNT_EDGES          # orders that were set aside (separator, least_tie, random, lex) use the earlier policy class
from twistroute.decode.policy import RevealState
from twistroute.resampling import NO_RESAMPLING
from twistroute.sampler import weighted_sampler
from twistroute.schedules import BudgetedCoReveal, Dilated, DistanceDependence, RandomOrder, Sum, fit_distance_dependence
from twistroute.torch_routes import torch_route, TorchChain, TorchCopy, TorchSubset
from twistroute.twists import ProductTwist

PRODUCT = ProductTwist(temperature=1.0)        # the product twist of the notes and of TWISTER: no tempering


def free_route(A, dev):
    return torch_route(C.build_dfa(0, lambda s, y: 0, lambda s: True, A, name="free"), dev)


def gap_stat(den):
    """The co-reveal gap of one step from xb to xa (see the module docstring); summed over the steps of a run by the
    `observe` callback of `weighted_sampler`."""
    def ps(xb, xa):
        new = (xb == den.mask_id) & (xa != den.mask_id)
        P = xb.shape[0]; rows = torch.arange(P, device=xb.device)
        cnt = new.sum(-1); kmax = int(cnt.max())
        order = torch.argsort(new.to(torch.int8), dim=-1, descending=True, stable=True)[:, :max(kmax, 1)]   # chosen, by index
        cur = xb.clone(); par = torch.zeros(P, dtype=torch.float64, device=xb.device); seq = torch.zeros_like(par)
        lp0 = None
        for k in range(kmax):
            lp = den.logits(cur)
            if k == 0:
                lp0 = lp
                tok = torch.where(new, xa, torch.zeros_like(xa))
                par = torch.where(new, torch.gather(lp0, -1, tok[..., None]).squeeze(-1).double(), torch.zeros_like(new, dtype=torch.float64)).sum(-1)
            pos = order[:, k]; live = k < cnt
            t = xa[rows, pos]
            seq = seq + torch.where(live, lp[rows, pos, torch.where(live, t, torch.zeros_like(t))].double(), torch.zeros_like(seq))
            cur[rows, pos] = torch.where(live, t, cur[rows, pos])
        return seq - par
    return ps


@torch.no_grad()
def score_along(den, cl, route, x_target, schedule, gen):
    """log-probability of x_target revealed along `schedule` (one token per step for a capped policy), values forced."""
    P, N = x_target.shape; dev = x_target.device
    x = torch.full_like(x_target, den.mask_id); out = torch.zeros(P, dtype=torch.float64, device=dev)
    for t in range(N):
        lp = den.logits(x)
        st = RevealState(x, cl.token_class[x], cl.class_logw(lp), den.mask_id, cl, route)
        pos, valid = schedule.reveal(st, t, N, gen)
        ch = torch.zeros_like(x, dtype=torch.bool); ch.scatter_(1, pos, valid); ch &= x == den.mask_id
        tok = torch.where(ch, x_target, torch.zeros_like(x))
        out += torch.where(ch, torch.gather(lp, -1, tok[..., None]).squeeze(-1).double(), torch.zeros_like(ch, dtype=torch.float64)).sum(-1)
        x = torch.where(ch, x_target, x)
        if not (x == den.mask_id).any():
            break
    return out


class LeftToRight:
    def __init__(self):
        self.calls, self.route_passes = 0, 0.0
    def reveal(self, st, t, T, gen=None):
        m = st.x == st.mask_id
        pos = torch.where(m, torch.arange(m.shape[1], device=m.device)[None], torch.full_like(m, m.shape[1], dtype=torch.long)).argmin(-1, keepdim=True)
        return pos, m.any(-1, keepdim=True)


def chunks(fn, total, per):
    outs = [fn(min(per, total - i), i) for i in range(0, total, per)]
    return {k: np.concatenate([np.asarray(o[k]) for o in outs]) for k in outs[0] if isinstance(outs[0][k], np.ndarray)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--traces", type=int, default=32); ap.add_argument("--pilot-reps", dest="pilot_reps", type=int, default=4)
    ap.add_argument("--audit", type=int, default=256, help="sequences per schedule in part B")
    ap.add_argument("--rows", type=int, default=96, help="rows per denoiser batch")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--nrun", type=int, default=48); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--parts", default="ABCD"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="mdlm_first_look.json")
    a = ap.parse_args()
    N = a.N; res = dict(args=vars(a)); t00 = time.time()
    save = lambda: (json.dump(res, open(a.out + ".tmp", "w"), indent=1), os.replace(a.out + ".tmp", a.out))
    if a.synthetic:
        from twistroute.models import SyntheticMDLM
        dev = "cpu"; den = SyntheticMDLM(N, scale=3.0); base = den.classes(4)
    else:
        from twistroute.models import semantic_classes
        from twistroute.models.mdlm_owt import MDLM
        dev = "cuda"; den = MDLM(dev); base = semantic_classes(den.tok, den.V, den.mask_id).to(dev); res["model"] = den.info()
    G = lambda s: torch.Generator(device=dev).manual_seed(a.seed * 1000 + s)
    one = base.merge([0, 0, 0, 0]); fr = free_route(1, dev)                       # one class, no constraint
    free = Constraint(fr, family="finite automaton", cost="O(L |E|)", n_classes=1, name="no constraint")

    # ------------------------------------------------------------------ A: kernel
    if os.path.exists(a.out):
        res.update({k: v for k, v in json.load(open(a.out)).items() if k in ("kernel", "audit", "orders", "constrained")})
    tabs = []
    for rep in range(a.pilot_reps):
        k = fit_distance_dependence(den, N, a.traces, G(rep), device=dev)
        tabs.append(k.table.double().cpu().numpy() if hasattr(k, "table") else np.stack([k.ku.numpy(), k.ks.numpy()], 1))
    table = torch.tensor(np.mean(tabs, 0)); kern = DistanceDependence.from_table(table, torch.tensor(list(COUNT_EDGES)))
    res["kernel"] = dict(count_edges=list(COUNT_EDGES), table=np.mean(tabs, 0).tolist(), rep_sd=np.std(tabs, 0).tolist(), seconds=round(time.time() - t00, 1))
    save(); print(f"[{time.time() - t00:6.0f}s] kernel: unscreened coupling at d = 1, 2, 4, 8, 16, 32: " + ", ".join(f"{table[d, 0]:.4f}" for d in (1, 2, 4, 8, 16, 32) if d < N), flush=True)

    # ------------------------------------------------------------------ B: unconstrained audit
    if "B" in a.parts:
        ps = gap_stat(den); res["audit"] = res.get("audit", {})
        arms = [(f"separator B={B}", lambda B=B: CouplingPolicy(kern, 0.0, priority="separator", mixed=False, rate=B / N)) for B in a.budgets]
        arms += [(f"least_tie B={B}", lambda B=B: CouplingPolicy(kern, 0.0, priority="least_tie", mixed=False, rate=B / N)) for B in a.budgets[2:5]]
        arms += [(f"random B={B}", lambda B=B: CouplingPolicy(kern, 0.0, priority="random", mixed=False, rate=B / N)) for B in a.budgets[2:5]]
        arms += [("dilated base=2", lambda: Dilated(2))] + [(f"fixed {k}/step", (lambda k=k: (RandomOrder(), N // k))) for k in (2, 4, 8, 16)]
        for name, mk in arms:
            if name in res["audit"]:
                continue
            t0 = time.time(); sch = mk(); T = N
            if isinstance(sch, tuple):
                sch, T = sch
            def go(n, i):
                gap = torch.zeros(n, dtype=torch.float64, device=dev)
                def observe(step_index, state, step, log_weights):
                    gap.add_(ps(state.tokens, step.tokens))
                o = weighted_sampler(den, one, free, N, T, runs=n, particles=1, twist=PRODUCT, schedule=sch, resampling=NO_RESAMPLING, generator=G(100 + i), observe=observe)
                return dict(r=gap.cpu().numpy().reshape(-1), steps=np.asarray(o.steps).reshape(-1).astype(float))
            o = chunks(go, a.audit, a.rows); r = o["r"]
            row = dict(steps=float(o["steps"].mean()), mean_r=float(r.mean()), sd_r=float(r.std()), kl=float(-r.mean()), kl_se=float(r.std() / math.sqrt(len(r))),
                       q50=float(np.quantile(-r, 0.5)), q90=float(np.quantile(-r, 0.9)), q99=float(np.quantile(-r, 0.99)), max=float((-r).max()), n=len(r), seconds=round(time.time() - t0, 1))
            if getattr(sch, "particle_calls", 0):                       # the schedule object is shared by all batches:
                row["load_per_path"] = sch.load_sum / len(r)                 # total admitted load over the audited paths
            res["audit"][name] = row; save()
            print(f"[{time.time() - t00:6.0f}s] audit {name:22s} steps {row['steps']:6.1f}  KL {row['kl']:8.3f} ± {row['kl_se']:.3f}  sd {row['sd_r']:.3f}  q90 {row['q90']:.3f}  q99 {row['q99']:.3f}", flush=True)

    # ------------------------------------------------------------------ C: order inconsistency
    if "C" in a.parts and "orders" not in res:
        n = min(a.rows, 128)
        x = torch.tensor(weighted_sampler(den, one, free, N, N, runs=n, particles=1, twist=PRODUCT, resampling=NO_RESAMPLING, generator=G(200)).tokens).view(n, N).to(dev)
        cap = lambda pr: CouplingPolicy(kern, 0.0, priority=pr, mixed=False, max_set=1)
        sc = dict(random_a=score_along(den, one, fr, x, cap("random"), G(201)), random_b=score_along(den, one, fr, x, cap("random"), G(202)),
                  separator=score_along(den, one, fr, x, cap("separator"), G(203)), left_to_right=score_along(den, one, fr, x, LeftToRight(), G(204)))
        ref = sc["random_a"]; res["orders"] = dict(n=n, mean_logp={k: float(v.mean()) for k, v in sc.items()},
                                                    diff_vs_random_a={k: dict(mean=float((v - ref).mean()), sd=float((v - ref).std()), se=float((v - ref).std() / math.sqrt(n))) for k, v in sc.items() if k != "random_a"})
        save(); print(f"[{time.time() - t00:6.0f}s] orders: log p along random order {float(ref.mean()):.1f}; differences " + ", ".join(f"{k} {d['mean']:+.2f} (sd {d['sd']:.2f})" for k, d in res["orders"]["diff_vs_random_a"].items()), flush=True)

    # ------------------------------------------------------------------ D: constrained
    if "D" in a.parts:
        cl = base.merge([0, 1, 2, 2]); A = 3; res["constrained"] = res.get("constrained", {})
        x0 = torch.full((1, N), den.mask_id, dtype=torch.long, device=dev); lw0 = cl.class_logw(den.logits(x0))
        w = max(8, N // 10); n2 = N // 2; m = max(1, N // 20)
        def window_route(pos):
            k_ok = None
            for k in range(1, len(pos)):                                  # smallest k the product twist puts below 1e-3
                rt = TorchSubset(torch_route(C.dfa_at_least_k(2, k, A), dev), pos, N, device=dev)
                if float(rt.logZ(lw0)[0]) <= math.log(1e-3) or k >= round(0.75 * len(pos)):
                    return rt, k
            return rt, k
        lo = N // 2 - w // 2; w2 = 2 * (w // 2); lo2 = N // 4 - w2 // 4
        cons = dict(window=lambda: window_route(list(range(lo, lo + w))),
                    two_window=lambda: window_route(list(range(lo2, lo2 + w2 // 2)) + list(range(lo2 + n2, lo2 + n2 + w2 // 2))),
                    copy_spread=lambda: (TorchCopy([(i, n2 + i) for i in list(range(0, n2, n2 // m))[:m]], N, A, device=dev), m))
        for cname, mk in cons.items():
            route, par = mk()
            constraint = Constraint(route, family=cname, cost="", n_classes=A, length=N)
            mkp = lambda pr, B=None, capn=None: CouplingPolicy(kern, 0.0, priority=pr, mixed=False, rate=None if B is None else B / N, max_set=capn)
            arms = [("fixed 1/step", lambda: RandomOrder()), ("sum sequential", lambda: Sum(kern)), ("lex sequential", lambda: mkp("lex", capn=1))]
            arms += [(f"sum B={B}", lambda B=B: BudgetedCoReveal(kern, B, pull_in=0)) for B in a.budgets[2:5]]
            arms += [(f"{pr} B={B}", lambda pr=pr, B=B: mkp(pr, B)) for pr in ("separator", "lex") for B in a.budgets[2:5]]
            for name, mks in arms:
                key = f"{cname}|{name}"
                if key in res["constrained"]:
                    continue
                t0 = time.time()
                def go(n, i):
                    o = weighted_sampler(den, cl, constraint, N, N, runs=n, particles=a.K, twist=PRODUCT, schedule=mks(), resampling=NO_RESAMPLING, generator=G(300 + i))
                    return dict(logZ=np.asarray(o.log_Z_hat, dtype=np.float64), steps=np.asarray(o.steps).astype(float).mean(-1))
                o = chunks(go, a.nrun, max(1, a.rows // a.K)); lz = o["logZ"]; fin = np.isfinite(lz); z = np.exp(lz[fin] - lz[fin].max())
                res["constrained"][key] = dict(param=par, steps=float(o["steps"].mean()), log_mean_Z=float(lz[fin].max() + math.log(z.mean())), mean_logZ=float(lz[fin].mean()),
                                               spread=float(z.std(ddof=1) / z.mean()), rel_se=float(z.std(ddof=1) / z.mean() / math.sqrt(len(z))), n=int(fin.sum()), seconds=round(time.time() - t0, 1))
                save(); r = res["constrained"][key]
                print(f"[{time.time() - t00:6.0f}s] {cname:11s} {name:18s} steps {r['steps']:6.1f}  log Z {r['log_mean_Z']:8.3f}  spread {r['spread']:.2f}  rel s.e. {r['rel_se']:.2f}", flush=True)
    print(f"done in {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
