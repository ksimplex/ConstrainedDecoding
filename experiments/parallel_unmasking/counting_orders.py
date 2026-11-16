"""Where along a decode does each order pay on a sum constraint, and does revealing well-connected positions early
lower the spread of the remaining count for later steps?

`counting` (exactly k of N positions are class 0), one token per step. For each order: the KL divergence of the
sampler's law from the exact constrained law (the sum over steps of the true error at the revealed position), and by
step the mean true error and the mean spread ratio VIF = Var(remaining count) / (variance of an independent sum).
Orders: random; lex (constraint-first); separator (most coupled first); sum; cheapest (ideal greedy, exact twist);
centred (greedy on the closed form of section 36o, with the model's exact one-step update).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, make_constraint                                                # noqa: E402
from pulib.orders import cheapest_order, kl                                                        # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402
from twistroute.decode import CouplingPolicy, fit_distance_kernel, COUNT_EDGES, constrained_marginals  # noqa: E402
from twistroute.decode.policy import RevealState                                                   # noqa: E402
from twistroute.decode.schedule import clamp_revealed                                              # noqa: E402


def influence(den, x, q, k):
    """Closed-form cost (P, N) of revealing each masked position (inf elsewhere) and the spread ratio VIF (P,)."""
    S = den.S; masked = x == den.mask_id; m = torch.exp(den.log_cond(x)); p0 = m[..., 0]
    b, i = masked.nonzero(as_tuple=True); P = len(b)
    bb, ii = b.repeat_interleave(S), i.repeat_interleave(S); aa = torch.arange(S).repeat(P)
    xv = x[bb].clone(); xv[torch.arange(len(bb)), ii] = aa
    others = masked[bb].clone(); others[torch.arange(len(bb)), ii] = False
    mv0 = torch.exp(den.log_cond(xv))[..., 0]
    delta = ((mv0 - p0[bb]) * others).sum(-1).view(P, S)
    mu_o = (p0[bb] * others).sum(-1).view(P, S)[:, 1]; var_o = ((p0[bb] * (1 - p0[bb])) * others).sum(-1).view(P, S)[:, 1]
    pi0 = p0[b, i]
    infl = delta[:, 0] - (m[b, i][:, 1:] * delta[:, 1:]).sum(-1) / (1 - pi0).clamp_min(1e-12)
    v = pi0 * (1 - pi0)
    num = torch.zeros(len(x), dtype=torch.float64).index_add_(0, b, v * infl)
    den_ = torch.zeros(len(x), dtype=torch.float64).index_add_(0, b, v)
    vif = 1 + num / den_.clamp_min(1e-12)
    krem = k - ((x == 0) & ~masked).sum(-1)
    slope = (2 * (krem[b].double() - mu_o) - 1) / (2 * vif[b] * var_o.clamp_min(1e-9))
    qi0 = q[b, i][:, 0]
    cost = torch.full(x.shape, float("inf"), dtype=torch.float64)
    cost[b, i] = 0.5 * qi0 * (1 - qi0) * (slope * (infl - (vif[b] - 1))) ** 2
    return cost, vif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--P", type=int, default=300); ap.add_argument("--model", default="hidden_0.85")
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--out", default="counting_orders.json")
    ap.add_argument("--constraint", default="counting", help="counting, or dyck (class 0 is the opening bracket; the spread "
                    "ratio is then that of the number of opening brackets, and `centred` does not apply)")
    ap.add_argument("--orders", nargs="+", default=["random", "lex", "separator", "sum", "cheapest", "centred"])
    a = ap.parse_args(); torch.set_num_threads(2)
    from pulib.cells import n_symbols
    N, P = a.N, a.P; k = N // 3 if a.constraint == "counting" else N // 2
    den = make_model(a.model, N, n_symbols(a.constraint)); cl, route, truth = make_constraint(a.constraint, den, N)
    kern, _ = fit_distance_kernel(den, N, a.pilot, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
    ref = te.Twists(den, truth.wdfa, getattr(truth, "active", None) if getattr(truth, "active", None) is not None else [True] * N)
    res = dict(args=vars(a), provenance=provenance(), log_pC=float(truth.log_pC), rows=[])
    for oname in a.orders:
        t0 = time.time()
        if oname == "cheapest":
            pr = cheapest_order(ref, den)
        elif oname == "centred":
            pr = lambda s: -torch.nan_to_num(influence(den, s.st.x, s.mu, k)[0], posinf=1e30)
        else:
            pr = oname
        pol = CouplingPolicy(kern, 0.0, priority=pr, mixed=False, max_set=1)
        x = torch.full((P, N), den.mask_id); g = torch.Generator().manual_seed(7); rows = torch.arange(P)
        eps_t = np.zeros(N); vif_t = np.zeros(N); logw = torch.zeros(P, dtype=torch.float64)
        hist = np.zeros((4, N)); qmax_t = np.zeros(N); hq_t = np.zeros(N); hm_t = np.zeros(N); dist_t = np.zeros(N); tilt_t = np.zeros(N)
        for t in range(N):
            masked = x == den.mask_id
            ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
            lw = cl.class_logw(den.logits(x)); st = RevealState(x, ycls, lw, den.mask_id, cl, route)
            lwc = clamp_revealed(lw, x, ycls, den.mask_id); q = constrained_marginals(route, lwc)
            vif_t[t] = float(influence(den, x, q, k)[1].mean()) if t < N - 1 else 1.0
            idx, _ = pol.reveal(st, t, N, g); i = idx[:, 0]
            q0 = q[rows, i]; pm = torch.exp(den.log_cond(x))[rows, i]
            pi = torch.zeros(P, den.S, dtype=torch.float64)
            for c in range(den.S):
                xa = x.clone(); xa[rows, i] = c
                pi[:, c] = pm[:, c] * torch.exp(ref.true(xa))
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-300)
            eps_t[t] = float(kl(q0, pi).mean())
            # what was picked: where, how determined by the constraint, how uncertain under the model, how far from revealed tokens
            hist[min(3, 4 * t // N)] += np.bincount(i.numpy(), minlength=N) / P
            qmax_t[t] = float(q0.max(-1).values.mean()); hq_t[t] = float(-(q0 * torch.log(q0.clamp_min(1e-300))).sum(-1).mean())
            hm_t[t] = float(-(pm * torch.log(pm.clamp_min(1e-300))).sum(-1).mean()); tilt_t[t] = float(kl(q0, pm).mean())
            if t > 0:
                pos = torch.arange(N).view(1, N).expand(P, N).double()
                dd = torch.where(masked, torch.full((P, N), 1e9, dtype=torch.float64), (pos - i.view(P, 1)).abs())
                dist_t[t] = float(dd.min(-1).values.mean())
            else:
                dist_t[t] = float("nan")
            tok = torch.multinomial(q0.clamp_min(0), 1, generator=g).squeeze(-1)
            logw += torch.log(pi[rows, tok].clamp_min(1e-300)) - torch.log(q0[rows, tok].clamp_min(1e-300))
            x[rows, i] = tok
        row = dict(order=oname, kl_from_steps=float(eps_t.sum()), kl_from_weights=float(-logw.mean()), kl_weights_se=float(logw.std() / P ** 0.5),
                   eps_by_step=eps_t.tolist(), vif_by_step=vif_t.tolist(), pick_hist_by_quarter=hist.tolist(), pick_qmax_by_step=qmax_t.tolist(), pick_entropy_constrained=hq_t.tolist(),
                   pick_entropy_model=hm_t.tolist(), pick_tilt=tilt_t.tolist(), pick_dist_to_revealed=dist_t.tolist(), seconds=round(time.time() - t0, 1))
        res["rows"].append(row); json.dump(res, open(a.out, "w"), indent=1)
        q4 = [float(eps_t[j * N // 4:(j + 1) * N // 4].sum()) for j in range(4)]
        print(f"{oname:10s} KL {row['kl_from_steps']:.3f} (weights {row['kl_from_weights']:.3f} +- {row['kl_weights_se']:.3f}) | error paid in each quarter of the decode: "
              + " ".join(f"{v:.3f}" for v in q4) + " | spread ratio at steps 0, 8, 16, 32, 48: " + " ".join(f"{vif_t[j]:.2f}" for j in (0, 8, 16, 32, 48) if j < N) + f" | {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
