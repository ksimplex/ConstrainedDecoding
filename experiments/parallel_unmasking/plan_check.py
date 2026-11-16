"""Does one step of planning on an estimated cost beat the greedy use of the same estimate?

One token per step. An estimate c_j(x) of the cost of revealing each masked position j at state x is used in two ways:
    greedy   reveal the position with the lowest c_i(x)
    plan     reveal the position with the lowest c_i(x) + E_{a ~ q_i} [ sum over the remaining masked j of c_j(x with i = a) ]:
             its own cost plus the estimated cost of everything left afterwards, so a position that makes the others
             cheaper is credited for it. No weights are tuned.
Cost models (both use the model's exact one-step update: this tests the planning idea, not a kernel):
    closed   counting only: 1/2 q_i(0)(1 - q_i(0)) (slope x (influence_i - average influence))^2 (section 36o)
    pair     any automaton: sum over masked j of 1/2 Var_{a ~ q_i} E[rho_j | x_i = a] (section 36n)
Reference orders: random, lex (constraint-first), separator, sum, cheapest (ideal greedy with the exact twist).
The score of an order is the KL divergence of the sampler's law from the exact constrained law, as the sum over
steps of the true error at the revealed position.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, make_constraint, n_symbols                                     # noqa: E402
from pulib.orders import cheapest_order, kl                                                        # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402
from twistroute.decode import CouplingPolicy, fit_distance_kernel, COUNT_EDGES, constrained_marginals  # noqa: E402
from twistroute.decode.policy import RevealState                                                   # noqa: E402
from twistroute.decode.schedule import clamp_revealed                                              # noqa: E402


class Costs:
    def __init__(self, den, cl, route, model, k=None, chunk=60000):
        self.den, self.cl, self.route, self.model, self.k, self.chunk = den, cl, route, model, k, chunk

    def marginals(self, x):
        masked = x == self.den.mask_id
        ycls = self.cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
        lw = self.cl.class_logw(self.den.logits(x))
        q = constrained_marginals(self.route, clamp_revealed(lw, x, ycls, self.den.mask_id))
        return torch.exp(self.den.log_cond(x)), q, masked

    def __call__(self, x):
        """(B, N) estimated cost of revealing each masked position; inf at revealed ones; 0 at a forced one."""
        out = torch.full(x.shape, float("inf"), dtype=torch.float64)
        for lo in range(0, len(x), max(1, self.chunk // (x.shape[1] * self.den.S * x.shape[1]))):
            sl = slice(lo, lo + max(1, self.chunk // (x.shape[1] * self.den.S * x.shape[1])))
            out[sl] = self._cost(x[sl])
        return out

    def _cost(self, x):
        den, S = self.den, self.den.S
        m, q, masked = self.marginals(x)
        b, i = masked.nonzero(as_tuple=True); P = len(b)
        out = torch.full(x.shape, float("inf"), dtype=torch.float64)
        if P == 0:
            return out
        bb, ii = b.repeat_interleave(S), i.repeat_interleave(S); aa = torch.arange(S).repeat(P)
        xv = x[bb].clone(); xv[torch.arange(len(bb)), ii] = aa
        others = masked[bb].clone(); others[torch.arange(len(bb)), ii] = False
        mv = torch.exp(den.log_cond(xv))
        qi = q[b, i]
        if self.model == "pair":
            rho = torch.where(m > 0, q / m.clamp_min(1e-300), torch.zeros_like(q))
            L = ((rho[bb] * mv).sum(-1) - (rho[bb] * m[bb]).sum(-1)) * others
            L = L.view(P, S, -1); mean = (qi[:, :, None] * L).sum(1, keepdim=True)
            c = 0.5 * (qi[:, :, None] * (L - mean) ** 2).sum(1).sum(-1)
        else:
            p0 = m[..., 0]; mv0 = mv[..., 0]
            delta = ((mv0 - p0[bb]) * others).sum(-1).view(P, S)
            mu_o = (p0[bb] * others).sum(-1).view(P, S)[:, 1]; var_o = ((p0[bb] * (1 - p0[bb])) * others).sum(-1).view(P, S)[:, 1]
            pi0 = p0[b, i]
            infl = delta[:, 0] - (m[b, i][:, 1:] * delta[:, 1:]).sum(-1) / (1 - pi0).clamp_min(1e-12)
            v = pi0 * (1 - pi0)
            num = torch.zeros(len(x), dtype=torch.float64).index_add_(0, b, v * infl)
            dn = torch.zeros(len(x), dtype=torch.float64).index_add_(0, b, v)
            vif = 1 + num / dn.clamp_min(1e-12)
            krem = self.k - ((x == 0) & ~masked).sum(-1)
            slope = (2 * (krem[b].double() - mu_o) - 1) / (2 * vif[b] * var_o.clamp_min(1e-9))
            c = 0.5 * qi[:, 0] * (1 - qi[:, 0]) * (slope * (infl - (vif[b] - 1))) ** 2
        out[b, i] = torch.nan_to_num(c, nan=0.0, posinf=1e6)
        return out


def plan_scores(costs, x, q):
    """(B, N): c_i(x) + E_{a ~ q_i} sum_j c_j(x with i = a) at masked i; inf elsewhere."""
    den, S = costs.den, costs.den.S
    c = costs(x); masked = x == den.mask_id
    b, i = masked.nonzero(as_tuple=True); P = len(b)
    bb, ii = b.repeat_interleave(S), i.repeat_interleave(S); aa = torch.arange(S).repeat(P)
    live = q[bb, ii, aa] > 1e-12                                                       # classes the proposal can draw
    xv = x[bb[live]].clone(); xv[torch.arange(int(live.sum())), ii[live]] = aa[live]
    cs = costs(xv)
    G = torch.where(torch.isfinite(cs), cs, torch.zeros_like(cs)).sum(-1)              # estimated cost of everything left
    Gall = torch.zeros(len(bb), dtype=torch.float64); Gall[live] = G
    exp_G = (q[b, i] * Gall.view(P, S)).sum(-1)
    out = torch.full(x.shape, float("inf"), dtype=torch.float64)
    out[b, i] = c[b, i] + exp_G
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--P", type=int, default=64); ap.add_argument("--model", default="hidden_0.85")
    ap.add_argument("--constraint", default="counting"); ap.add_argument("--cost", choices=("closed", "pair"), default="closed")
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--out", default="plan_check.json")
    ap.add_argument("--orders", nargs="+", default=["random", "lex", "separator", "sum", "cheapest", "greedy", "plan"])
    a = ap.parse_args(); torch.set_num_threads(2)
    N, P = a.N, a.P
    den = make_model(a.model, N, n_symbols(a.constraint)); cl, route, truth = make_constraint(a.constraint, den, N)
    kern, _ = fit_distance_kernel(den, N, a.pilot, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
    act = getattr(truth, "active", None)
    ref = te.Twists(den, truth.wdfa, act if act is not None else [True] * N)
    costs = Costs(den, cl, route, a.cost, k=N // 3)
    res = dict(args=vars(a), provenance=provenance(), log_pC=float(truth.log_pC), rows=[])
    for oname in a.orders:
        t0 = time.time()
        if oname == "cheapest":
            pr = cheapest_order(ref, den)
        elif oname == "greedy":
            pr = lambda s: -torch.nan_to_num(costs(s.st.x), posinf=1e30)
        elif oname == "plan":
            pr = lambda s: -torch.nan_to_num(plan_scores(costs, s.st.x, s.mu), posinf=1e30)
        else:
            pr = oname
        pol = CouplingPolicy(kern, 0.0, priority=pr, mixed=False, max_set=1)
        x = torch.full((P, N), den.mask_id); g = torch.Generator().manual_seed(7); rows = torch.arange(P)
        eps_t = np.zeros(N); logw = torch.zeros(P, dtype=torch.float64)
        for t in range(N):
            masked = x == den.mask_id
            ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
            lw = cl.class_logw(den.logits(x)); st = RevealState(x, ycls, lw, den.mask_id, cl, route)
            q = constrained_marginals(route, clamp_revealed(lw, x, ycls, den.mask_id))
            idx, _ = pol.reveal(st, t, N, g); i = idx[:, 0]
            q0 = q[rows, i]; pm = torch.exp(den.log_cond(x))[rows, i]
            pi = torch.zeros(P, den.S, dtype=torch.float64)
            for c in range(den.S):
                xa = x.clone(); xa[rows, i] = c
                pi[:, c] = pm[:, c] * torch.exp(ref.true(xa))
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-300)
            e = kl(q0, pi); eps_t[t] = float(torch.nan_to_num(e, posinf=0.0).mean())
            tok = torch.multinomial(q0.clamp_min(0), 1, generator=g).squeeze(-1)
            logw += torch.log(pi[rows, tok].clamp_min(1e-300)) - torch.log(q0[rows, tok].clamp_min(1e-300))
            x[rows, i] = tok
        row = dict(order=oname, kl_from_steps=float(eps_t.sum()), kl_from_weights=float(-logw.mean()), kl_weights_se=float(logw.std() / P ** 0.5),
                   eps_by_step=eps_t.tolist(), seconds=round(time.time() - t0, 1))
        res["rows"].append(row); json.dump(res, open(a.out, "w"), indent=1)
        q4 = [float(eps_t[j * N // 4:(j + 1) * N // 4].sum()) for j in range(4)]
        print(f"{a.constraint} {a.cost:6s} {oname:10s} KL {row['kl_from_steps']:.3f} (weights {row['kl_from_weights']:.3f} +- {row['kl_weights_se']:.3f}) | by quarter: "
              + " ".join(f"{v:.3f}" for v in q4) + f" | {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
