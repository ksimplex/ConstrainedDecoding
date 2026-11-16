"""Does a pairwise account of the model's dependence predict the true cost of revealing a position?

At a state x, revealing position i draws its class from the product twist's constrained marginal q_i; the exact
sampler would draw from pi_i = p(class at i | revealed tokens, C). The true cost, in the KL divergence of the
sampler's law from the exact constrained law, is eps_i = KL(q_i || pi_i) (needs the exact twist: oracle models only).

Estimators, none of which uses the exact twist. Write m_j for the model's marginal at a masked position j, rho_j =
(constrained marginal) / m_j for the constraint's ratio there, and m_j(. | a) for the model's marginal at j once
position i is revealed as class a.
    pair      sum over masked j of 1/2 Var_{a ~ q_i} E[rho_j(x_j) | x_i = a]: each pair counted separately (leading order)
    tilt      KL(q_i || q_i exp(D_i) / norm), D_i(a) = sum_j (E[rho_j | a] - 1): the same first-order change of the
              log twist, summed over j before the variance is taken (keeps covariances between the j)
    lookahead KL(q_i || p_i(a) Z(successor) / norm): the product twist re-evaluated at the successor state
    chain     KL(q_i || chain twist's constrained marginal): the earlier estimate (section 36g), for reference
Each of the first three in versions: `exact`, with m_j(. | a) from the model itself (one denoiser call per
position and class: tests the derivation); `kernel`, with m_j(. | a) proportional to m_j times the model's
unconditional pair ratio at that distance (no denoiser call; ignores the screening by revealed tokens); `screen`, the
same with the log ratio scaled by sqrt(kernel given what is revealed in between / unscreened kernel), from the pilot.
    pairc, tiltc   the same as pair and tilt with rho_j taken after position i is fixed at a (the constrained marginal
              at j given x_i = a, over m_j): what the constraint asks of j once i is known. One route pass per (i, a).
    stake     D_i = KL(q_i || p_i): how far the constraint moves position i (a ranking, not a size)
    self_oracle    NOT an estimator: the cost if the exact tilt at i were the product twist's tilt scaled by one factor
              c per state (the median over the state's positions of the fitted slope of the true log tilt on the
              product twist's). Measures how much of the cost is a common over- or under-reaction of the twist.
    self_plus_pairc   self_oracle + pairc_exact

States are drawn from the exact constrained law and revealed at random at several fractions; every masked position
with a finite true cost is scored.
"""
import argparse, json, os, sys, time, warnings
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
EST = ("pair_exact", "tilt_exact", "lookahead_exact", "pairc_exact", "tiltc_exact", "pair_kernel", "tilt_kernel", "lookahead_kernel",
       "pair_screen", "pairc_screen", "tiltc_screen", "lookahead_screen", "chain", "stake", "self_oracle", "self_plus_pairc")


def pair_ratios(den):
    """R (N, N, S, S): the model's unconditional pair law of tokens (i, j) over the product of its marginals."""
    pi, T, E = den.pi.numpy(), den.trans.numpy(), den.emit.numpy(); N, H, S = E.shape
    alpha = [pi]
    for i in range(N - 1): alpha.append(alpha[-1] @ T[i])
    R = np.ones((N, N, S, S))
    for i in range(N):
        M = np.eye(H)
        for j in range(i + 1, N):
            M = M @ T[j - 1]
            P = np.einsum("h,ha,hg,gb->ab", alpha[i], E[i], M, E[j])
            r = P / np.outer(P.sum(1), P.sum(0))
            R[i, j], R[j, i] = r, r.T
    return torch.tensor(R)


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    from twistroute.decode import constrained_marginals
    from twistroute.decode.schedule import clamp_revealed
    den = make_model(mname, N, n_symbols(cname)); S = den.S
    cl, route, truth = make_constraint(cname, den, N)
    cell = Cell(mname, cname, N, dict(a, nrun=1, pilot_small=0), den, cl, route, truth)
    twist = cell.chain_twist(); kern = cell.kernel("graded")
    ku = kern.table[:, 0].clamp_min(1e-300)
    ref = te.Twists(den, truth.wdfa, truth.active)
    R = pair_ratios(den)
    rng = np.random.default_rng(cell.seed("pair_cost")); g = torch.Generator().manual_seed(cell.seed("pair_cost"))
    y = torch.tensor(truth.sample(a["states"], rng))
    act = torch.tensor(np.asarray(truth.active, dtype=bool)) if getattr(truth, "active", None) is not None else torch.ones(N, dtype=torch.bool)
    wpos = act.nonzero().squeeze(-1)
    dist = (torch.arange(N)[:, None] - wpos[None]).abs().min(-1).values                 # distance to the constrained set
    rows = []
    for f in FRACS:
        t0 = time.time()
        x = torch.where(torch.rand(y.shape, generator=g) < f, y, torch.full_like(y, den.mask_id))
        masked = x == den.mask_id
        ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
        lw = cl.class_logw(den.logits(x)); lwc = clamp_revealed(lw, x, ycls, den.mask_id)
        m = torch.exp(den.log_cond(x))                                                  # model marginals (B, N, S)
        q = constrained_marginals(route, lwc)                                           # proposal marginals
        pi = exact_conditionals(ref, den, x, m)
        eps = kl(q, pi)
        ent = -torch.xlogy(q, q).sum(-1)
        rho = torch.where(m > 0, q / m.clamp_min(1e-300), torch.zeros_like(q))          # constraint ratio (B, N, S)
        b, i = masked.nonzero(as_tuple=True); P = len(b)                                # every (state, masked position)
        bb, ii = b.repeat_interleave(S), i.repeat_interleave(S); aa = torch.arange(S).repeat(P)
        xv = x[bb].clone(); xv[torch.arange(len(bb)), ii] = aa                          # i revealed as class a
        others = masked[bb].clone(); others[torch.arange(len(bb)), ii] = False          # the masked positions left
        yv = cl.token_class[torch.where(xv == den.mask_id, torch.zeros_like(xv), xv)]
        qi = q[b, i]                                                                    # (P, S)
        lmi = torch.log(m[b, i].clamp_min(1e-300))
        est = {}
        Rk = R[ii, :, aa, :]                                                            # (V, N, S): ratio for (i = a, j = .)
        mk = m[bb] * Rk; mk = mk / mk.sum(-1, keepdim=True).clamp_min(1e-300)
        Mscr = kern.matrix(masked)                                                      # (B, N, N) kernel given what is revealed in between
        dd = (torch.arange(N)[None] - ii[:, None]).abs()                                # (V, N)
        gam = torch.sqrt((Mscr[bb, ii] / ku[dd]).clamp(0.0, 1.0))                       # share of the log ratio that survives screening
        ms = m[bb] * torch.exp(gam[..., None] * torch.log(Rk.clamp_min(1e-300))); ms = ms / ms.sum(-1, keepdim=True).clamp_min(1e-300)
        mu_c = constrained_marginals(route, clamp_revealed(lw[bb], xv, yv, den.mask_id))  # constrained marginals with i fixed at a
        rho_c = torch.where(m[bb] > 0, mu_c / m[bb].clamp_min(1e-300), torch.zeros_like(mu_c))
        for tag, mv in (("exact", torch.exp(den.log_cond(xv))), ("kernel", mk), ("screen", ms)):
            for rtag, rr in (("", rho[bb]), ("c", rho_c)):
                if tag == "kernel" and rtag == "c":
                    continue
                L = (rr * mv).sum(-1)                                                   # E[rho_j | x_i = a] (V, N)
                base = (rr * m[bb]).sum(-1)                                             # its value before the update (1 where the ratio is defined)
                L = torch.where(others, L - base, torch.zeros_like(L)).view(P, S, N)    # change of the expected ratio
                mean = (qi[:, :, None] * L).sum(1, keepdim=True)
                est[f"pair{rtag}_{tag}"] = 0.5 * (qi[:, :, None] * (L - mean) ** 2).sum(1).sum(-1)
                D = L.sum(-1)                                                           # (P, S)
                lt = torch.log(qi.clamp_min(1e-300)) + D; lt = torch.where(qi > 0, lt, torch.full_like(lt, -float("inf")))
                est[f"tilt{rtag}_{tag}"] = kl(qi, torch.softmax(lt, -1))
            lwv = clamp_revealed(torch.log(mv.clamp_min(1e-300)), xv, yv, den.mask_id)
            z = route.logZ(lwv).double().view(P, S)                                     # product twist at the successor
            la = lmi + torch.nan_to_num(z, nan=-float("inf"))
            est["lookahead_" + tag] = kl(qi, torch.softmax(la, -1))
        est = {k: v for k, v in est.items() if k in EST}
        est["chain"] = kl(q, twist.marginals(lwc))[b, i]
        est["stake"] = kl(q, m)[b, i]
        e = eps[b, i]; ok = torch.isfinite(e); free = ok & (ent[b, i] > 1e-6)
        # is the cost a common over- or under-reaction of the twist? fit the true log tilt on the product twist's
        mi_, pii = m[b, i], pi[b, i]
        sup = (qi > 1e-12) & (pii > 1e-300) & (mi_ > 0)
        t = torch.where(sup, torch.log(qi.clamp_min(1e-300)) - torch.log(mi_.clamp_min(1e-300)), torch.zeros_like(qi))
        ts = torch.where(sup, torch.log(pii.clamp_min(1e-300)) - torch.log(mi_.clamp_min(1e-300)), torch.zeros_like(qi))
        wq = torch.where(sup, qi, torch.zeros_like(qi)); wq = wq / wq.sum(-1, keepdim=True).clamp_min(1e-300)
        tm, tsm = (wq * t).sum(-1, keepdim=True), (wq * ts).sum(-1, keepdim=True)
        vt = (wq * (t - tm) ** 2).sum(-1)
        slope = (wq * (t - tm) * (ts - tsm)).sum(-1) / vt.clamp_min(1e-300)             # c_i: 1 = the twist's tilt is right
        good = free & (vt > 1e-8)
        cs = torch.ones(len(x), dtype=torch.float64)
        for st_ in range(len(x)):
            sel_ = good & (b == st_)
            if int(sel_.sum()) >= 3:
                cs[st_] = slope[sel_].median()
        lts = torch.log(mi_.clamp_min(1e-300)) + cs[b][:, None] * t
        lts = torch.where(qi > 0, lts, torch.full_like(lts, -float("inf")))
        est["self_oracle"] = kl(qi, torch.softmax(lts, -1))
        est["self_plus_pairc"] = est["self_oracle"] + est["pairc_exact"]
        row = dict(model=mname, constraint=cname, N=N, frac=f, states=int(a["states"]), positions=int(ok.sum()), unforced=int(free.sum()),
                   mean_eps=float(e[ok].mean()), sum_eps_per_state=float(e[ok].sum() / len(x)))
        row.update(slope_median=float(slope[good].median()), slope_q25=float(slope[good].quantile(0.25)), slope_q75=float(slope[good].quantile(0.75)),
                   slope_state_median=float(cs.median()), slope_state_q25=float(cs.quantile(0.25)), slope_state_q75=float(cs.quantile(0.75)),
                   slope_within_state_iqr=float(np.median([float(slope[good & (b == s_)].quantile(0.75) - slope[good & (b == s_)].quantile(0.25))
                                                           for s_ in range(len(x)) if int((good & (b == s_)).sum()) >= 4] or [float("nan")])))
        en = e[free].numpy(); big = en > np.quantile(en, 0.5); top = en > np.quantile(en, 0.9)
        grp = dict(inside=dist[i] == 0, near=(dist[i] > 0) & (dist[i] <= 6), far=dist[i] > 6)
        for k in EST:
            h_all = torch.nan_to_num(est[k], nan=0.0, posinf=1e3); h = h_all[free].numpy()
            row[f"spearman_{k}"] = float(spearmanr(h, en)[0])
            within = [spearmanr(h_all[(b == s) & free].numpy(), e[(b == s) & free].numpy())[0] for s in range(len(x)) if int(((b == s) & free).sum()) >= 5]
            row[f"spearman_within_{k}"] = float(np.nanmean(within)) if within else None
            row[f"ratio_median_{k}"] = float(np.median(h[big] / en[big]))
            row[f"within_2x_{k}"] = float(np.mean((h[big] > en[big] / 2) & (h[big] < en[big] * 2)))
            row[f"under_10x_{k}"] = float(np.mean(h[big] < en[big] / 10))
            row[f"over_10x_{k}"] = float(np.mean(h[big] > en[big] * 10))
            row[f"top_decile_flagged_{k}"] = float(np.mean(h[top] > np.median(h)))
            row[f"sum_ratio_{k}"] = float(h.sum() / en.sum())
            for gname, gm in grp.items():
                sel = gm & free
                row[f"mean_{gname}_{k}"] = float(h_all[sel].mean()) if int(sel.sum()) else None
            # the true cost at the position this estimate would reveal next (lowest estimate among the unforced)
            sc = torch.full((len(x), N), float("inf"), dtype=torch.float64); sc[b[free], i[free]] = h_all[free].double()
            has = torch.isfinite(sc).any(-1); pick = sc.argmin(-1)
            row[f"eps_at_pick_{k}"] = float(eps[torch.arange(len(x)), pick][has].mean())
        for gname, gm in grp.items():
            sel = gm & free
            row[f"mean_{gname}_true"] = float(e[sel].mean()) if int(sel.sum()) else None; row[f"n_{gname}"] = int(sel.sum())
        sc = torch.full((len(x), N), float("inf"), dtype=torch.float64); sc[b[free], i[free]] = e[free].double()
        has = torch.isfinite(sc).any(-1)
        row["eps_at_pick_ideal"] = float(eps[torch.arange(len(x)), sc.argmin(-1)][has].mean())
        rnd = torch.where(torch.isfinite(sc), torch.rand(sc.shape, generator=g).double(), torch.full_like(sc, float("inf")))
        row["eps_at_pick_random"] = float(eps[torch.arange(len(x)), rnd.argmin(-1)][has].mean())
        se = torch.full((len(x), N), float("inf"), dtype=torch.float64); se[b[free], i[free]] = ent[b, i][free].double()
        row["eps_at_pick_entropy"] = float(eps[torch.arange(len(x)), se.argmin(-1)][has].mean())
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=["hidden_0.85"])
    ap.add_argument("--constraints", nargs="+", default=["counting", "dyck", "forbidden", "window", "two_window"])
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default="pair_cost.json")
    a = vars(ap.parse_args())
    jobs = [(mm, c, a["N"], a) for mm in a["models"] for c in a["constraints"]]
    rows, prov, t0 = [], provenance(), time.time()
    def collect(res):
        rows.extend(res)
        json.dump(dict(args=a, provenance=prov, rows=rows), open(a["out"], "w"), indent=1)
        print(f"[{time.time() - t0:6.0f}s] done {res[0]['model']} {res[0]['constraint']}", flush=True)
    if a["workers"] <= 1:
        for j in jobs: collect(run_cell(j))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a["workers"]) as pool:
            for res in pool.imap_unordered(run_cell, jobs): collect(res)


if __name__ == "__main__":
    main()
