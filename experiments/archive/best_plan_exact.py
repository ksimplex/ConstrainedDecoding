"""The ideal reveal order, exactly, at a length where every partially revealed state can be enumerated.

One token per step; the proposal at the revealed position is the product twist's constrained marginal q_i. The KL
divergence of the sampler's law from the exact constrained law is the expected sum over steps of the true error
eps_i(x) = KL(q_i || exact conditional) at the revealed position (expectation under the sampler). With all
(S + 1)^N states tabulated this is a finite decision problem:

    V(x) = min over masked i of [ eps_i(x) + sum_a q_i(a | x) V(x with i = a) ],   V(fully revealed) = 0.

Solved by one backward pass, it gives the best state-dependent order and its KL with no sampling noise, and the same
pass evaluates any other rule exactly:
    optimal, worst, random (uniform over masked positions)
    ltr                 left to right
    greedy              lowest true error now (the ideal greedy order, `cheapest` elsewhere)
    look1..look4        depth-d lookahead on the true error: lowest [error now + expected best depth-(d-1) value next]
    sum1                lowest [true error now + expected sum of the current true errors of everything left]
    rollout_random      lowest [true error now + expected KL of finishing in random order]; rollout_greedy likewise
    most/least_dependent, most/least_tilted, most/least_certain   single-feature rules: the model's dependence of the
                        position on the rest (sum of mutual informations), how far the constraint moves its marginal,
                        the entropy of its constrained marginal
    lex, separator, sum the repository's orders (pilot kernel)
    est_greedy          lowest estimated cost now            (estimate: `closed` for counting, `pair` for any automaton)
    est_plan1           lowest [estimate now + expected sum of the estimates of everything left afterwards]
    est_dp              the best plan for the estimated cost (full-depth planning on the estimate), scored on the true error
Also reported: where along the decode each rule pays, what the optimal rule picks (how often it is the greedy pick, the
rank of its pick by current error, how cheap it leaves the remaining positions), and at the empty state the error of
each first reveal next to its total cost under optimal play afterwards.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "studies"))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, make_constraint, n_symbols                                     # noqa: E402
from pulib.orders import kl                                                                        # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402
from twistroute.decode import CouplingPolicy, fit_distance_kernel, COUNT_EDGES, constrained_marginals  # noqa: E402
from twistroute.decode.policy import RevealState                                                   # noqa: E402
from twistroute.decode.schedule import clamp_revealed                                              # noqa: E402
from plan_check import Costs                                                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10); ap.add_argument("--model", default="hidden_0.85")
    ap.add_argument("--constraint", default="counting"); ap.add_argument("--cost", choices=("closed", "pair", "none"), default="none")
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--depth", type=int, default=4); ap.add_argument("--features", type=int, default=1)
    ap.add_argument("--repo-orders", nargs="*", default=["lex", "separator", "sum"]); ap.add_argument("--out", default="best_plan_exact.json")
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time()
    N = a.N; S = n_symbols(a.constraint)
    den = make_model(a.model, N, S)
    if a.constraint in ("window_small", "two_window_small"):          # at least 3 of 4 positions are class 0: one block in the
        from twistroute import compile as C                           # middle, or two pairs a few positions apart (the
        from twistroute.models import ConstrainedTruth                # repository's windows are at least 8 wide)
        from twistroute.torch_routes import TorchChain, TorchSubset
        act = np.zeros(N, dtype=bool)
        if a.constraint == "window_small":
            act[N // 2 - 2:N // 2 + 2] = True
        else:
            act[1:3] = True; act[N - 3:N - 1] = True
        d = C.dfa_at_least_k(0, 3, S); cl = den.classes(); truth = ConstrainedTruth(den, cl, d, active=act); truth.wdfa = d
        truth.active = act; route = TorchSubset(TorchChain(d), [int(i) for i in np.nonzero(act)[0]], N)
    else:
        cl, route, truth = make_constraint(a.constraint, den, N)
    assert den.mask_id == S
    act = getattr(truth, "active", None)
    ref = te.Twists(den, truth.wdfa, act if act is not None else [True] * N)
    B = (S + 1) ** N; pw = (S + 1) ** np.arange(N)
    codes = np.arange(B); dig = ((codes[:, None] // pw[None]) % (S + 1)).astype(np.int64)        # digit S = masked
    masked = dig == S; nm = masked.sum(1)
    x_all = torch.from_numpy(dig)
    pm = np.zeros((B, N, S)); q = np.zeros((B, N, S)); lt = np.zeros(B)
    for lo in range(0, B, a.chunk):
        x = x_all[lo:lo + a.chunk]; mk = x == S
        ycls = cl.token_class[torch.where(mk, torch.zeros_like(x), x)]
        lw = cl.class_logw(den.logits(x))
        q[lo:lo + a.chunk] = torch.nan_to_num(constrained_marginals(route, clamp_revealed(lw, x, ycls, S)), nan=0.0).numpy()
        pm[lo:lo + a.chunk] = torch.exp(den.log_cond(x)).numpy(); lt[lo:lo + a.chunk] = ref.true(x).numpy()
    alive = np.isfinite(lt) & (lt > -1e200)
    print(f"{a.constraint} {a.model} N={N}: {B} states, {int(alive.sum())} consistent with the constraint, tables in {time.time() - t0:.0f} s", flush=True)

    def succ(sel, i):
        return sel[:, None] + (np.arange(S)[None] - S) * pw[i]

    eps = np.full((B, N), np.inf)
    for i in range(N):
        sel = codes[masked[:, i] & alive]; sc = succ(sel, i)
        pi = pm[sel, i] * np.exp(np.where(alive[sc], lt[sc], -np.inf)); pi = pi / np.clip(pi.sum(1, keepdims=True), 1e-300, None)
        e = kl(torch.from_numpy(q[sel, i]), torch.from_numpy(pi)).numpy()
        eps[sel, i] = np.nan_to_num(e, nan=0.0, posinf=0.0)
    levels = [codes[(nm == m) & alive] for m in range(N + 1)]

    def table_min(tab):                                     # per state: min and sum over masked positions of a (B, N) table
        t = np.where(masked & alive[:, None], tab, np.inf)
        mn = t.min(1); mn[~np.isfinite(mn)] = 0.0
        return mn, np.where(np.isfinite(t), t, 0.0).sum(1)

    def lookahead(tab, nxt):                                # tab_i(x) + E_{a ~ q_i} nxt(x with i = a)
        out = np.full((B, N), np.inf)
        for i in range(N):
            sel = codes[masked[:, i] & alive]; sc = succ(sel, i)
            out[sel, i] = tab[sel, i] + (q[sel, i] * nxt[sc]).sum(1)
        return out

    def solve(cost, mode="min"):
        """Backward pass on `cost`: value and pick of the rule that minimises (maximises) total expected cost."""
        V = np.zeros(B); pick = np.full(B, -1)
        for m in range(1, N + 1):
            idx = levels[m]; best = np.full(len(idx), np.inf if mode == "min" else -np.inf); bi = np.full(len(idx), -1)
            for i in range(N):
                w = masked[idx, i]; sel = idx[w]
                tot = cost[sel, i] + (q[sel, i] * V[succ(sel, i)]).sum(1)
                better = tot < best[w] if mode == "min" else tot > best[w]
                bw = best[w]; bw[better] = tot[better]; best[w] = bw
                biw = bi[w]; biw[better] = i; bi[w] = biw
            V[idx] = best; pick[idx] = bi
        return V, pick

    def evaluate(pick):
        """Exact KL of the rule `pick` (B,) and the error it pays at each step (expectation under its own sampler)."""
        V = np.zeros(B)
        for m in range(1, N + 1):
            idx = levels[m]
            for i in range(N):
                sel = idx[pick[idx] == i]
                V[sel] = eps[sel, i] + (q[sel, i] * V[succ(sel, i)]).sum(1)
        reach = np.zeros(B); reach[B - 1] = 1.0; paid = np.zeros(N)
        for m in range(N, 0, -1):
            idx = levels[m]
            for i in range(N):
                sel = idx[pick[idx] == i]
                paid[N - m] += (reach[sel] * eps[sel, i]).sum()
                reach[succ(sel, i)] += reach[sel, None] * q[sel, i]
        return V, reach, paid

    def argmin_pick(score):
        s = np.where(masked & alive[:, None], score, np.inf); p = s.argmin(1); p[(nm == 0) | ~alive] = -1
        return p

    rules = {}
    Vopt, pick_opt = solve(eps, "min"); rules["optimal"] = pick_opt
    Vworst, pick_worst = solve(eps, "max"); rules["worst"] = pick_worst
    rules["ltr"] = argmin_pick(np.broadcast_to(np.arange(N, dtype=float), (B, N)))
    rules["greedy"] = argmin_pick(eps)
    mineps, sumeps = table_min(eps)
    Qd = eps                                                # depth-d lookahead on the true error: Q_d = eps + E min_j Q_{d-1}(successor)
    for d in range(1, a.depth + 1):
        Qd = lookahead(eps, table_min(Qd)[0]); rules[f"look{d}"] = argmin_pick(Qd)
    rules["sum1"] = argmin_pick(lookahead(eps, sumeps))    # error now + expected sum of the current errors of everything left
    feats = {"error_now": eps}
    if a.features:
        # model dependence of i on the rest: sum over masked j of I(x_i; x_j | revealed), from the tabulated conditionals
        mi = np.full((B, N), np.inf); tilt = np.full((B, N), np.inf); hq = np.full((B, N), np.inf)
        for i in range(N):
            sel_all = codes[masked[:, i] & alive]
            for lo in range(0, len(sel_all), 100000):
                sel = sel_all[lo:lo + 100000]; sc = succ(sel, i)                     # (n, S)
                pj = pm[sc]                                                          # (n, S, N, S): law of j given i = a
                base = pm[sel][:, None]                                              # (n, 1, N, S)
                klj = np.where(pj > 0, pj * (np.log(np.clip(pj, 1e-300, None)) - np.log(np.clip(base, 1e-300, None))), 0.0).sum(-1)   # (n, S, N)
                others = masked[sel].copy(); others[:, i] = False
                mi[sel, i] = (pm[sel, i][:, :, None] * klj * others[:, None, :]).sum((1, 2))
                qi = q[sel, i]; mi_ = pm[sel, i]
                tilt[sel, i] = np.where(qi > 0, qi * (np.log(np.clip(qi, 1e-300, None)) - np.log(np.clip(mi_, 1e-300, None))), 0.0).sum(1)
                hq[sel, i] = -(qi * np.log(np.clip(qi, 1e-300, None))).sum(1)
        feats.update(model_dependence=mi, tilt=tilt, constrained_entropy=hq)
        rules["most_dependent"] = argmin_pick(-mi); rules["least_dependent"] = argmin_pick(mi)
        rules["most_tilted"] = argmin_pick(-tilt); rules["least_tilted"] = argmin_pick(tilt)
        rules["most_certain"] = argmin_pick(hq); rules["least_certain"] = argmin_pick(-hq)
    if a.repo_orders:
        kern, _ = fit_distance_kernel(den, N, a.pilot, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
        g = torch.Generator().manual_seed(7)
        for name in a.repo_orders:
            pol = CouplingPolicy(kern, 0.0, priority=name, mixed=False, max_set=1); p = np.full(B, -1)
            for m in range(1, N + 1):
                idx = levels[m]
                for lo in range(0, len(idx), 20000):
                    sl = idx[lo:lo + 20000]; x = x_all[sl]; mk = x == S
                    ycls = cl.token_class[torch.where(mk, torch.zeros_like(x), x)]
                    st = RevealState(x, ycls, cl.class_logw(den.logits(x)), S, cl, route)
                    p[sl] = pol.reveal(st, N - m, N, g)[0][:, 0].numpy()
            assert (masked[codes[p >= 0], p[p >= 0]]).all()
            rules[name] = p
    if a.cost != "none":
        costs = Costs(den, cl, route, a.cost, k=N // 3); c = np.full((B, N), np.inf)
        for lo in range(0, B, a.chunk):
            c[lo:lo + a.chunk] = costs(x_all[lo:lo + a.chunk]).numpy()
        c = np.where(masked & alive[:, None], np.nan_to_num(c, nan=0.0, posinf=1e6), np.inf)
        rules["est_greedy"] = argmin_pick(c)
        rules["est_plan1"] = argmin_pick(lookahead(c, table_min(c)[1]))
        rules["est_dp"] = solve(np.where(np.isfinite(c), c, 0.0), "min")[1]

    # random: average over masked positions
    Vr = np.zeros(B)
    for m in range(1, N + 1):
        idx = levels[m]; acc = np.zeros(len(idx))
        for i in range(N):
            w = masked[idx, i]; sel = idx[w]
            acc[w] += eps[sel, i] + (q[sel, i] * Vr[succ(sel, i)]).sum(1)
        Vr[idx] = acc / m
    e0 = B - 1
    rules["rollout_random"] = argmin_pick(lookahead(eps, Vr))       # error now + expected KL of finishing in random order
    Vg_ = np.zeros(B)
    for m in range(1, N + 1):
        idx = levels[m]
        for i in range(N):
            sel = idx[rules["greedy"][idx] == i]
            Vg_[sel] = eps[sel, i] + (q[sel, i] * Vg_[succ(sel, i)]).sum(1)
    rules["rollout_greedy"] = argmin_pick(lookahead(eps, Vg_))      # error now + expected KL of finishing greedily
    Qopt = lookahead(eps, Vopt)
    feats["after_random"] = lookahead(np.where(np.isfinite(eps), 0.0, np.inf), Vr)      # expected KL of finishing in random order after revealing i
    feats["total_if_random_after"] = lookahead(eps, Vr)
    res = dict(args=vars(a), provenance=provenance(), log_pC=float(truth.log_pC), states=int(B), kl={"random": float(Vr[e0])}, paid_by_step={}, rules=list(rules))
    print(f"exact KL of the sampler's law from the constrained law (nats, lower is better); log p(C) = {truth.log_pC:.3f}")
    print(f"  {'random':15s} {Vr[e0]:.4f}")
    reach_opt = None; gp = rules["greedy"]
    for name, p in rules.items():
        V, reach, paid = evaluate(p)
        res["kl"][name] = float(V[e0]); res["paid_by_step"][name] = paid.tolist()
        q4 = [paid[j * N // 4:(j + 1) * N // 4].sum() for j in range(4)]
        agree = sum((reach[levels[m]] * (Qopt[levels[m], p[levels[m]]] <= Vopt[levels[m]] + 1e-12)).sum() for m in range(1, N + 1)) / N
        res.setdefault("optimal_pick_share", {})[name] = float(agree)
        print(f"  {name:15s} {V[e0]:.4f} | paid by quarter " + " ".join(f"{v:.4f}" for v in q4) + f" | an optimal pick in {100 * agree:.0f}% of its steps", flush=True)
        if name == "optimal":
            reach_opt = reach
    # what the optimal rule picks, by step (under its own sampler)
    diag = []
    for m in range(N, 0, -1):
        idx = levels[m]; r = reach_opt[idx]; i = pick_opt[idx]; e = np.where(masked[idx], eps[idx], np.inf)
        ei = eps[idx, i]; rank = (e < ei[:, None] - 1e-15).sum(1) / max(1, m - 1) if m > 1 else np.zeros(len(idx))
        qi = q[idx, i]; hq = -(qi * np.log(np.clip(qi, 1e-300, None))).sum(1)
        mean_now = np.where(np.isfinite(e), e, 0).sum(1) / m
        diag.append(dict(step=N - m, same_as_greedy=float((r * (ei <= e.min(1) + 1e-12)).sum()), rank_of_pick=float((r * rank).sum()),
                         paid=float((r * ei).sum()), cheapest_available=float((r * e.min(1)).sum()), mean_available=float((r * mean_now).sum()),
                         entropy_of_pick=float((r * hq).sum()), mean_position=float((r * i).sum())))
    res["optimal_by_step"] = diag
    # where the optimal pick sits in the ordering of the masked positions by each feature (0 = lowest value, 1 = highest), by step
    fr = {k: [] for k in feats}
    for m in range(N, 1, -1):
        idx = levels[m]; r = reach_opt[idx]; keep = r > 0; idx = idx[keep]; r = r[keep]; i = pick_opt[idx]
        for k, tab in feats.items():
            f = np.where(masked[idx], tab[idx], np.nan); fi = tab[idx, i]
            below = np.nansum(f < fi[:, None] - 1e-12, axis=1); ties = np.nansum(np.abs(f - fi[:, None]) <= 1e-12, axis=1) - 1
            fr[k].append(float((r * (below + 0.5 * ties) / (m - 1)).sum()))
    res["optimal_feature_rank_by_step"] = fr
    print("rank of the optimal pick among masked positions by each feature (0 = lowest, 1 = highest; 0.5 = no relation), steps 0.." + str(N - 2))
    for k, v in fr.items():
        print(f"  {k:22s} " + " ".join(f"{x:.2f}" for x in v))
    print("optimal rule by step: step | same pick as greedy | rank of its pick by current error (0 = cheapest, 1 = dearest) | error paid | cheapest available | mean available")
    for d in diag:
        print(f"  {d['step']:2d} | {d['same_as_greedy']:.2f} | {d['rank_of_pick']:.2f} | {d['paid']:.4f} | {d['cheapest_available']:.4f} | {d['mean_available']:.4f}")
    # the first reveal: error now, and total under optimal play afterwards
    first = []
    for i in range(N):
        sc = succ(np.array([e0]), i)[0]
        first.append(dict(position=i, error_now=float(eps[e0, i]), total_then_optimal=float(eps[e0, i] + (q[e0, i] * Vopt[sc]).sum()),
                          total_then_greedy=None))
    Vg = evaluate(gp)[0]
    for i in range(N):
        sc = succ(np.array([e0]), i)[0]; first[i]["total_then_greedy"] = float(eps[e0, i] + (q[e0, i] * Vg[sc]).sum())
    res["first_reveal"] = first
    print("first reveal: position | error now | total KL if play is optimal afterwards | total KL if play is greedy afterwards")
    for f in first:
        print(f"  {f['position']:2d} | {f['error_now']:.4f} | {f['total_then_optimal']:.4f} | {f['total_then_greedy']:.4f}")
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1); print(f"{res['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
