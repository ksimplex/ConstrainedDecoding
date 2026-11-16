"""Why is the pair account poor on a sum constraint? A test of one derivation, on `counting` (exactly k of the N
positions are class 0).

Revealing position i as class a leaves the other masked positions to supply k_a zeros (k_rem - 1 if a = 0, else
k_rem). The exact conditional is pi_i(a) proportional to p_i(a) P(S = k_a | x_i = a), with S the number of zeros among
the other masked positions; the product twist uses the same expression with S a sum of independent indicators at the
current marginals. Two things are then wrong, with opposite signs:
    mean     revealing a = 0 raises the expected count of the others (they depend on i): delta_i(a)
    spread   the true count is more dispersed than an independent sum: Var(S) = VIF x (independent variance), and
             VIF - 1 is the variance-weighted average over positions of delta_j(0) - delta_j(not 0)
With a normal approximation to the count, the exact conditional is predicted by
    pi_hat_i(a) proportional to p_i(a) Normal(k_a; mean_others + delta_i(a), VIF x independent variance)
Estimators of eps_i = KL(q_i || pi_i), all from the model's exact one-step update (a test of the derivation):
    mean_only    the mean shift with the independent variance (the lookahead of pair_cost_check, in normal form)
    spread_only  the inflated variance without the mean shift
    both         the two together
    centred      1/2 q_i(0) (1 - q_i(0)) (slope x (influence_i - average influence))^2: the closed form the
                 derivation gives when both are small
"""
import argparse, json, math, os, sys, time, warnings
import numpy as np
import torch
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message="An input array is constant")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, make_constraint                                                # noqa: E402
from pulib.orders import kl, exact_conditionals                                                    # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402

FRACS = (0.1, 0.25, 0.5, 0.75)
EST = ("mean_only", "spread_only", "both", "centred", "stake")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=["hidden_0.85"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="collective.json")
    a = ap.parse_args(); torch.set_num_threads(2)
    from twistroute.decode import constrained_marginals
    from twistroute.decode.schedule import clamp_revealed
    res = dict(args=vars(a), provenance=provenance(), rows=[])
    for mname in a.models:
        N = a.N; den = make_model(mname, N, 3); S = den.S; k = N // 3
        cl, route, truth = make_constraint("counting", den, N)
        ref = te.Twists(den, truth.wdfa, truth.active)
        rng = np.random.default_rng(a.seed); g = torch.Generator().manual_seed(a.seed)
        y = torch.tensor(truth.sample(a.states, rng))
        for f in FRACS:
            t0 = time.time()
            x = torch.where(torch.rand(y.shape, generator=g) < f, y, torch.full_like(y, den.mask_id))
            masked = x == den.mask_id
            ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
            lw = cl.class_logw(den.logits(x)); lwc = clamp_revealed(lw, x, ycls, den.mask_id)
            m = torch.exp(den.log_cond(x)); q = constrained_marginals(route, lwc)
            pi = exact_conditionals(ref, den, x, m); eps = kl(q, pi)
            ent = -torch.xlogy(q, q).sum(-1)
            krem = k - ((x == 0) & ~masked).sum(-1)                                       # zeros still to be supplied
            b, i = masked.nonzero(as_tuple=True); P = len(b)
            bb, ii = b.repeat_interleave(S), i.repeat_interleave(S); aa = torch.arange(S).repeat(P)
            xv = x[bb].clone(); xv[torch.arange(len(bb)), ii] = aa
            others = masked[bb].clone(); others[torch.arange(len(bb)), ii] = False
            p0 = m[..., 0]                                                                # P(class 0) at every position
            mv0 = torch.exp(den.log_cond(xv))[..., 0]                                     # after revealing i as a
            delta = ((mv0 - p0[bb]) * others).sum(-1).view(P, S)                          # shift of the others' expected count
            var_new = ((mv0 * (1 - mv0)) * others).sum(-1).view(P, S)                     # independent variance after the update
            mu_o = ((p0[bb]) * others).sum(-1).view(P, S)                                 # others' expected count before
            var_o = ((p0[bb] * (1 - p0[bb])) * others).sum(-1).view(P, S)
            pi0, qi = p0[b, i], q[b, i]
            infl = delta[:, 0] - (m[b, i][:, 1:] * delta[:, 1:]).sum(-1) / (1 - pi0).clamp_min(1e-12)   # influence of i on the others' count
            v = pi0 * (1 - pi0)
            num = torch.zeros(len(x), dtype=torch.float64).index_add_(0, b, v * infl)
            den_ = torch.zeros(len(x), dtype=torch.float64).index_add_(0, b, v)
            vif = 1 + num / den_.clamp_min(1e-12)                                         # Var(count) / independent variance, per state
            ka = torch.stack([krem[b] - 1, krem[b], krem[b]], 1).double()                 # zeros the others must supply
            lmi = torch.log(m[b, i].clamp_min(1e-300))
            def normal(mean, var):
                return -0.5 * (ka - mean) ** 2 / var.clamp_min(1e-9) - 0.5 * torch.log(var.clamp_min(1e-9))
            est = dict(mean_only=kl(qi, torch.softmax(lmi + normal(mu_o + delta, var_new), -1)),
                       spread_only=kl(qi, torch.softmax(lmi + normal(mu_o, vif[b][:, None] * var_o), -1)),
                       both=kl(qi, torch.softmax(lmi + normal(mu_o + delta, vif[b][:, None] * var_new), -1)),
                       stake=kl(q, m)[b, i])
            gap = (krem[b].double() - mu_o[:, 1])                                         # needed minus expected among the others
            slope = (2 * gap - 1) / (2 * vif[b] * var_o[:, 1].clamp_min(1e-9))
            est["centred"] = 0.5 * qi[:, 0] * (1 - qi[:, 0]) * (slope * (infl - (vif[b] - 1))) ** 2
            e = eps[b, i]; ok = torch.isfinite(e); free = ok & (ent[b, i] > 1e-6)
            en = e[free].numpy(); big = en > np.quantile(en, 0.5)
            row = dict(model=mname, frac=f, unforced=int(free.sum()), mean_eps=float(e[free].mean()), vif_median=float(vif.median()),
                       vif_q25=float(vif.quantile(0.25)), vif_q75=float(vif.quantile(0.75)), influence_median=float(infl[free].median()),
                       influence_q10=float(infl[free].quantile(0.1)), influence_q90=float(infl[free].quantile(0.9)))
            sc_true = torch.full((len(x), N), float("inf"), dtype=torch.float64); sc_true[b[free], i[free]] = e[free].double()
            has = torch.isfinite(sc_true).any(-1); rowsx = torch.arange(len(x))
            row["eps_at_pick_ideal"] = float(eps[rowsx, sc_true.argmin(-1)][has].mean())
            rnd = torch.where(torch.isfinite(sc_true), torch.rand(sc_true.shape, generator=g).double(), torch.full_like(sc_true, float("inf")))
            row["eps_at_pick_random"] = float(eps[rowsx, rnd.argmin(-1)][has].mean())
            for kname in EST:
                h_all = torch.nan_to_num(est[kname], nan=0.0, posinf=1e3); h = h_all[free].numpy()
                within = [spearmanr(h_all[(b == s) & free].numpy(), e[(b == s) & free].numpy())[0] for s in range(len(x)) if int(((b == s) & free).sum()) >= 5]
                row[f"spearman_within_{kname}"] = float(np.nanmean(within))
                row[f"spearman_{kname}"] = float(spearmanr(h, en)[0])
                row[f"ratio_median_{kname}"] = float(np.median(h[big] / en[big]))
                row[f"within_2x_{kname}"] = float(np.mean((h[big] > en[big] / 2) & (h[big] < en[big] * 2)))
                row[f"sum_ratio_{kname}"] = float(h.sum() / en.sum())
                sc = torch.full((len(x), N), float("inf"), dtype=torch.float64); sc[b[free], i[free]] = h_all[free].double()
                row[f"eps_at_pick_{kname}"] = float(eps[rowsx, sc.argmin(-1)][has].mean())
            row["seconds"] = round(time.time() - t0, 1)
            res["rows"].append(row); json.dump(res, open(a.out, "w"), indent=1)
            print(f"{mname} f={f}: eps {1000 * row['mean_eps']:.2f}e-3, VIF {row['vif_median']:.2f} [{row['vif_q25']:.2f}, {row['vif_q75']:.2f}], influence {row['influence_median']:.2f} "
                  f"[{row['influence_q10']:.2f}, {row['influence_q90']:.2f}] | rank within state: " + ", ".join(f"{kk} {row['spearman_within_' + kk]:.2f}" for kk in EST)
                  + " | size: " + ", ".join(f"{kk} {row['ratio_median_' + kk]:.2f}" for kk in EST[:4])
                  + f" | pick x1000: ideal {1000 * row['eps_at_pick_ideal']:.2f}, " + ", ".join(f"{kk} {1000 * row['eps_at_pick_' + kk]:.2f}" for kk in EST) + f", random {1000 * row['eps_at_pick_random']:.2f}", flush=True)


if __name__ == "__main__":
    main()
