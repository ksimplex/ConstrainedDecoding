"""Does the disagreement between the chain twist and the product twist estimate the product twist's error?

At a state x, revealing position i draws its class from the product twist's constrained marginal q0_i, while the
exact sampler would draw from pi_i = p(class at i | revealed tokens, C). The cost of revealing i now, in the KL
divergence of the sampler's law from the exact constrained law, is
    eps_i = KL(q0_i || pi_i)                      (true error; needs the exact twist, available on oracle models)
and the proposed estimate is
    est_i = KL(q0_i || q1_i)                      (q1_i: the constrained marginal under the chain twist; no denoiser call)
The identity eps = est + KL(q1 || pi) + sum (q0 - q1) log(q1 / pi) gives, to leading order,
    | sqrt(eps) - sqrt(est) | <= sqrt(KL(q1 || pi))
so the estimate should be good where the chain twist's own error is small beside the product twist's.

States are drawn from the exact constrained law and revealed at random at several fractions. Reported per constraint
and fraction, over masked positions on which the constraint acts or which neighbour such a position:
    rank correlation of est with eps, pooled and within states; the share of positions that satisfy the bound;
    the true error at the position each rule would reveal next: cheapest (ideal, uses eps), lowest estimate, highest
    stake D (constraint-first), lowest entropy of q0, highest separator score, random; among all masked positions
    and among those that are not forced (a forced position costs nothing, and every policy reveals it first).
Two versions of the chain: `ratio` (pilot pair kernel, as in the sampler) and `exact` (the model's own adjacent-pair
conditionals: the best a first-order chain can do).
"""
import argparse, json, math, os, sys, time, warnings
import numpy as np
import torch
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message="An input array is constant")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.arms import Cell                                                                        # noqa: E402
from pulib.cells import make_model, make_constraint, n_symbols                                     # noqa: E402
from pulib.orders import kl, exact_conditionals                                                    # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402

FRACS = (0.1, 0.25, 0.5, 0.75)
RULES = ("cheapest", "estimate_ratio", "estimate_exact", "stake", "separator", "entropy", "random")


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    from twistroute.decode import constrained_marginals
    from twistroute.decode.schedule import clamp_revealed
    den = make_model(mname, N, n_symbols(cname))
    cl, route, truth = make_constraint(cname, den, N)
    cell = Cell(mname, cname, N, dict(a, nrun=1, pilot_small=0), den, cl, route, truth)
    kern, twist = cell.kernel("graded"), cell.chain_twist()
    ref = te.Twists(den, truth.wdfa, truth.active)
    rng = np.random.default_rng(cell.seed("error_estimate"))
    g = torch.Generator().manual_seed(cell.seed("error_estimate"))
    y = torch.tensor(truth.sample(a["states"], rng))
    rows = []
    for f in FRACS:
        t0 = time.time()
        x = torch.where(torch.rand(y.shape, generator=g) < f, y, torch.full_like(y, den.mask_id))
        masked = x == den.mask_id
        lw = cl.class_logw(den.logits(x))
        lwc = clamp_revealed(lw, x, cl.token_class[torch.where(masked, torch.zeros_like(x), x)], den.mask_id)
        mu = torch.exp(den.log_cond(x))
        q0 = constrained_marginals(route, lwc)
        q1r = twist.marginals(lwc)
        q1x = twist.marginals(lwc, Qc=ref.pair_conditionals(x))
        pi = exact_conditionals(ref, den, x, mu)
        eps, est_r, est_x = kl(q0, pi), kl(q0, q1r), kl(q0, q1x)
        err_r, err_x = kl(q1r, pi), kl(q1x, pi)                            # the chain twists' own errors
        D = kl(q0, mu)
        sep = kern.apply(masked, masked.double())
        ent = -torch.xlogy(q0, q0).sum(-1)
        ok = masked & torch.isfinite(eps)
        nb = torch.zeros_like(ok); st = ok & (D > 1e-6)                     # stake positions and their neighbours
        nb[:, 1:] |= st[:, :-1]; nb[:, :-1] |= st[:, 1:]
        use = ok & (st | nb)
        row = dict(model=mname, constraint=cname, N=N, frac=f, states=int(a["states"]), positions=int(use.sum()),
                   off_support=float((masked & ~torch.isfinite(eps)).sum() / masked.sum().clamp_min(1)),
                   mean_eps=float(eps[use].mean()), median_eps=float(eps[use].median()))
        for tag, est, err in (("ratio", est_r, err_r), ("exact", est_x, err_x)):
            e, h, c = eps[use].numpy(), est[use].numpy(), err[use].numpy()
            row[f"spearman_{tag}"] = float(spearmanr(h, e)[0])
            within = [spearmanr(est[b][use[b]].numpy(), eps[b][use[b]].numpy())[0] for b in range(len(x)) if use[b].sum() >= 5]
            row[f"spearman_within_{tag}"] = float(np.nanmean(within)) if within else None
            row[f"bound_holds_{tag}"] = float(np.mean(np.abs(np.sqrt(e) - np.sqrt(h)) <= 1.1 * np.sqrt(c) + 1e-6))
            big = e > np.quantile(e, 0.5)
            row[f"ratio_est_over_eps_median_{tag}"] = float(np.median(h[big] / e[big]))
            row[f"under_10x_{tag}"] = float(np.mean(h[big] < e[big] / 10))      # share underestimated more than tenfold
            row[f"under_100x_{tag}"] = float(np.mean(h[big] < e[big] / 100))
            row[f"over_10x_{tag}"] = float(np.mean(h[big] > e[big] * 10))
            top = e > np.quantile(e, 0.9)                                    # the costliest tenth of positions:
            row[f"top_decile_flagged_{tag}"] = float(np.mean(h[top] > np.median(h)))    # does the estimate rank them above its median?
            row[f"chain_error_over_product_error_{tag}"] = float(np.median(np.sqrt(c[big] / e[big])))
        scores = dict(cheapest=-eps, estimate_ratio=-est_r, estimate_exact=-est_x, stake=D, separator=sep, entropy=-ent,
                      random=torch.rand(eps.shape, generator=g).double())
        free = ok & (ent > 1e-6)                                           # not forced: a forced position costs nothing
        row["forced_share"] = float((ok & ~free).sum() / ok.sum().clamp_min(1))
        for tag, pool in (("", ok), ("_unforced", free)):                  # the true error at the position each rule picks
            for r in RULES:
                sc = torch.where(pool, scores[r], torch.full_like(eps, -float("inf")))
                pick = sc.argmax(-1); has = pool.any(-1)
                row[f"eps_at_pick{tag}_{r}"] = float(eps[torch.arange(len(x)), pick][has].mean())
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=["hidden_0.85"])
    ap.add_argument("--constraints", nargs="+", default=["counting", "dyck", "forbidden", "window", "two_window"])
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default="error_estimate.json")
    a = vars(ap.parse_args())
    jobs = [(m, c, a["N"], a) for m in a["models"] for c in a["constraints"]]
    rows, prov, t0 = [], provenance(), time.time()
    def collect(res):
        rows.extend(res)
        json.dump(dict(args=a, provenance=prov, rows=rows), open(a["out"], "w"), indent=1)
        print(f"[{time.time() - t0:6.0f}s] done {res[0]['model']} {res[0]['constraint']}", flush=True)
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
