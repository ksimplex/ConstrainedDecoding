"""Can the ideal reveal order be found without the exact twist? Exact test at a length where every state is tabulated.

Two facts make the order problem independent of the exact twist (one token per step, product-twist proposal q_i, an
order-consistent model, a sampler whose every sample satisfies the constraint):

    KL(sampler || constrained law) = log p(C) + E_sampler[ sum over steps of KL(q_i || m_i) ],

where m_i is the model's marginal at the revealed position and KL(q_i || m_i), the *tilt*, is how far the proposal moves
it. The tilt is known at decision time (both q_i and m_i are in hand), and log p(C) does not depend on the order, so the
best order is the one with the lowest expected total tilt: a planning problem stated in the model's conditionals and the
route alone. The true per-step error differs from the tilt by a telescoping term,
    eps_i(x) = tilt_i(x) + log psi(x) - E_{a ~ q_i} log psi(x with i = a),          psi = exact twist,
so both give the same best order but different greedy orders.

This script evaluates, exactly on the true model:
    optimal / random / greedy / rollout_greedy     as in best_plan_exact.py
    static                the best fixed (state-independent) order found by local search over permutations
    sur:optimal           the best plan computed inside a surrogate model fitted to pilot samples (no further calls),
    sur:rollout_random    one step of improvement over random order inside the surrogate,
    sur:static            the best fixed order found inside the surrogate,
    mix:optimal           true tilt now + the surrogate's expected total tilt afterwards (optimal play in the surrogate),
    mix:rollout_random    true tilt now + the surrogate's expected total tilt of finishing in random order,
for surrogates markov1 (first-order chain) and markov2 (second-order chain) fitted to `--pilot` unconditional samples.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, make_constraint, n_symbols                                     # noqa: E402
from pulib.orders import kl                                                                        # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402
from twistroute.decode import constrained_marginals                                                # noqa: E402
from twistroute.decode.schedule import clamp_revealed                                              # noqa: E402
from twistroute.models.oracle import OracleHMM                                                     # noqa: E402


def constraint_for(name, den, N):
    S = den.S
    if name in ("window_small", "two_window_small"):
        from twistroute import compile as C
        from twistroute.models import ConstrainedTruth
        from twistroute.torch_routes import TorchChain, TorchSubset
        act = np.zeros(N, dtype=bool)
        if name == "window_small":
            act[N // 2 - 2:N // 2 + 2] = True
        else:
            act[1:3] = True; act[N - 3:N - 1] = True
        d = C.dfa_at_least_k(0, 3, S); cl = den.classes(); truth = ConstrainedTruth(den, cl, d, active=act); truth.wdfa = d
        truth.active = act
        return cl, TorchSubset(TorchChain(d), [int(i) for i in np.nonzero(act)[0]], N), truth
    return make_constraint(name, den, N)


def fit_markov(x, S, order, smooth=0.5):
    """A first- or second-order Markov chain fitted to samples x (P, N), as an OracleHMM (hidden state = last 1 or 2 tokens)."""
    P, N = x.shape
    pi0 = np.bincount(x[:, 0], minlength=S) + smooth; pi0 = pi0 / pi0.sum()
    c1 = np.full((S, S), smooth); np.add.at(c1, (x[:, :-1].ravel(), x[:, 1:].ravel()), 1); T1 = c1 / c1.sum(1, keepdims=True)
    if order == 1:
        return OracleHMM(pi0, np.tile(T1, (N - 1, 1, 1)), np.tile(np.eye(S), (N, 1, 1)), name="markov1")
    c2 = np.full((S, S, S), smooth); np.add.at(c2, (x[:, :-2].ravel(), x[:, 1:-1].ravel(), x[:, 2:].ravel()), 1); T2 = c2 / c2.sum(2, keepdims=True)
    H = S * S                                              # hidden state (previous token a, current token b) -> index a * S + b
    pi = np.zeros(H); pi[np.arange(S)] = pi0               # position 0: previous token fixed at 0 (unused by the first transition)
    tr = np.zeros((N - 1, H, H)); emit = np.zeros((N, H, S))
    for a in range(S):
        for b in range(S):
            emit[:, a * S + b, b] = 1.0
            for c in range(S):
                tr[0, a * S + b, b * S + c] = T1[b, c]; tr[1:, a * S + b, b * S + c] = T2[a, b, c]
    return OracleHMM(pi, tr, emit, name="markov2")


class World:
    """All (S + 1)^N states of one model under one constraint: marginals, proposals, true error, tilt."""

    def __init__(self, den, name, N, chunk=50000, proposal="product", pilot=2000):
        self.N, self.S = N, den.S; S = den.S
        cl, route, truth = constraint_for(name, den, N); self.log_pC = float(truth.log_pC)
        act = getattr(truth, "active", None); ref = te.Twists(den, truth.wdfa, act if act is not None else [True] * N)
        chain = None
        if proposal == "chain":                             # propose from the first-order chain twist's constrained marginals
            from twistroute.decode import ChainTwist, fit_pair_kernel
            chain = ChainTwist(truth.wdfa, fit_pair_kernel(den, cl, pilot, torch.Generator().manual_seed(5)), active=act)
        B = self.B = (S + 1) ** N; self.pw = (S + 1) ** np.arange(N); self.codes = np.arange(B)
        dig = ((self.codes[:, None] // self.pw[None]) % (S + 1)).astype(np.int64); self.masked = dig == S; self.nm = self.masked.sum(1)
        x_all = torch.from_numpy(dig); pm = np.zeros((B, N, S)); q = np.zeros((B, N, S)); lt = np.zeros(B)
        for lo in range(0, B, chunk):
            x = x_all[lo:lo + chunk]; mk = x == S
            ycls = cl.token_class[torch.where(mk, torch.zeros_like(x), x)]
            lwc = clamp_revealed(cl.class_logw(den.logits(x)), x, ycls, S)
            qq = constrained_marginals(route, lwc) if chain is None else chain.marginals(lwc)
            q[lo:lo + chunk] = torch.nan_to_num(qq / qq.sum(-1, keepdim=True).clamp_min(1e-300), nan=0.0).numpy()
            pm[lo:lo + chunk] = torch.exp(den.log_cond(x)).numpy(); lt[lo:lo + chunk] = ref.true(x).numpy()
        self.q, self.lt = q, lt; self.alive = alive = np.isfinite(lt) & (lt > -1e200)
        self.eps = np.full((B, N), np.inf); self.tilt = np.full((B, N), np.inf)
        for i in range(N):
            sel = self.codes[self.masked[:, i] & alive]; sc = self.succ(sel, i)
            pi = pm[sel, i] * np.exp(np.where(alive[sc], lt[sc], -np.inf)); pi = pi / np.clip(pi.sum(1, keepdims=True), 1e-300, None)
            self.eps[sel, i] = np.nan_to_num(kl(torch.from_numpy(q[sel, i]), torch.from_numpy(pi)).numpy(), nan=0.0, posinf=0.0)
            self.tilt[sel, i] = np.nan_to_num(kl(torch.from_numpy(q[sel, i]), torch.from_numpy(pm[sel, i])).numpy(), nan=0.0, posinf=0.0)
        self.levels = [self.codes[(self.nm == m) & alive] for m in range(N + 1)]; self.e0 = B - 1

    def succ(self, sel, i):
        return sel[:, None] + (np.arange(self.S)[None] - self.S) * self.pw[i]

    def lookahead(self, tab, nxt, q=None):
        q = self.q if q is None else q; out = np.full((self.B, self.N), np.inf)
        for i in range(self.N):
            sel = self.codes[self.masked[:, i] & self.alive]
            out[sel, i] = tab[sel, i] + (q[sel, i] * nxt[self.succ(sel, i)]).sum(1)
        return out

    def argmin_pick(self, score):
        s = np.where(self.masked & self.alive[:, None], score, np.inf); p = s.argmin(1); p[(self.nm == 0) | ~self.alive] = -1
        return p

    def solve(self, cost):
        V = np.zeros(self.B); pick = np.full(self.B, -1)
        for m in range(1, self.N + 1):
            idx = self.levels[m]; best = np.full(len(idx), np.inf); bi = np.full(len(idx), -1)
            for i in range(self.N):
                w = self.masked[idx, i]; sel = idx[w]
                tot = cost[sel, i] + (self.q[sel, i] * V[self.succ(sel, i)]).sum(1)
                better = tot < best[w]; bw = best[w]; bw[better] = tot[better]; best[w] = bw
                biw = bi[w]; biw[better] = i; bi[w] = biw
            V[idx] = best; pick[idx] = bi
        return V, pick

    def value(self, pick, cost=None):
        cost = self.eps if cost is None else cost; V = np.zeros(self.B)
        for m in range(1, self.N + 1):
            idx = self.levels[m]
            for i in range(self.N):
                sel = idx[pick[idx] == i]
                V[sel] = cost[sel, i] + (self.q[sel, i] * V[self.succ(sel, i)]).sum(1)
        return V

    def value_random(self, cost=None):
        cost = self.eps if cost is None else cost; V = np.zeros(self.B)
        for m in range(1, self.N + 1):
            idx = self.levels[m]; acc = np.zeros(len(idx))
            for i in range(self.N):
                w = self.masked[idx, i]; sel = idx[w]
                acc[w] += cost[sel, i] + (self.q[sel, i] * V[self.succ(sel, i)]).sum(1)
            V[idx] = acc / m
        return V

    def static_value(self, perm, cost=None):
        """Exact expected total cost of the fixed order `perm` (only the states it reaches are visited)."""
        cost = self.eps if cost is None else cost; codes = np.array([self.e0]); w = np.ones(1); tot = 0.0
        for i in perm:
            tot += float((w * cost[codes, i]).sum())
            w = (w[:, None] * self.q[codes, i]).ravel(); codes = self.succ(codes, i).ravel(); keep = (w > 0) & self.alive[codes]; codes, w = codes[keep], w[keep]     # the proposal leaves ~1e-17 on dead states
        return tot

    def best_static(self, rng, restarts=12, cost=None):
        """Local search over permutations (swaps and insertions, first improvement) from several starts."""
        N = self.N; best_p, best_v = None, np.inf
        starts = [list(range(N))] + [list(rng.permutation(N)) for _ in range(restarts)]
        for p in starts:
            v = self.static_value(p, cost); improved = True
            while improved:
                improved = False
                for a in range(N):
                    for b in range(N):
                        if a == b:
                            continue
                        for move in ("swap", "insert"):
                            c = list(p)
                            if move == "swap":
                                if b < a:
                                    continue
                                c[a], c[b] = c[b], c[a]
                            else:
                                c.insert(b, c.pop(a))
                            vc = self.static_value(c, cost)
                            if vc < v - 1e-12:
                                p, v, improved = c, vc, True
            if v < best_v:
                best_p, best_v = list(p), v
        return [int(i) for i in best_p], best_v

    def static_pick(self, perm):
        p = np.full(self.B, -1)
        for j in reversed(perm):
            p[self.masked[:, j]] = j
        p[(self.nm == 0) | ~self.alive] = -1
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10); ap.add_argument("--model", default="hidden_0.85"); ap.add_argument("--constraint", default="counting")
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--surrogates", nargs="*", default=["markov1", "markov2"])
    ap.add_argument("--proposal", choices=("product", "chain"), default="product", help="propose from the product twist's constrained marginals, or the chain twist's")
    ap.add_argument("--restarts", type=int, default=12); ap.add_argument("--out", default="surrogate_plan.json")
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time(); N = a.N; S = n_symbols(a.constraint)
    den = make_model(a.model, N, S); W = World(den, a.constraint, N, proposal=a.proposal, pilot=a.pilot); e0 = W.e0; rng = np.random.default_rng(3)
    Vopt, pick_opt = W.solve(W.eps); Vr = W.value_random()
    res = dict(args=vars(a), provenance=provenance(), log_pC=W.log_pC, kl={}, checks={}, static={})

    def record(name, pick):
        v = float(W.value(pick)[e0]); res["kl"][name] = v
        print(f"  {name:28s} {v:.4f}   ({v / Vopt[e0]:.2f} x optimal)", flush=True)

    print(f"{a.constraint} {a.model} N={N}, {a.proposal} proposal: exact KL of the sampler from the constrained law (nats, lower is better); log p(C) = {W.log_pC:.3f}; tables {time.time() - t0:.0f} s")
    res["kl"]["random"] = float(Vr[e0]); print(f"  {'random':28s} {Vr[e0]:.4f}   ({Vr[e0] / Vopt[e0]:.2f} x optimal)")
    record("optimal", pick_opt); g = W.argmin_pick(W.eps); record("greedy", g)
    record("rollout_greedy", W.argmin_pick(W.lookahead(W.eps, W.value(g)))); record("rollout_random", W.argmin_pick(W.lookahead(W.eps, Vr)))
    record("least_tilted", W.argmin_pick(W.tilt))
    # identities: total tilt + log p(C) = KL, for the optimal rule and for random order; and the tilt form gives the same best plan
    Tt, pick_tilt = W.solve(W.tilt)
    res["checks"] = dict(optimal_kl=float(Vopt[e0]), optimal_total_tilt_plus_log_pC=float(W.value(pick_opt, W.tilt)[e0] + W.log_pC),
                         random_kl=float(Vr[e0]), random_total_tilt_plus_log_pC=float(W.value_random(W.tilt)[e0] + W.log_pC),
                         best_plan_for_tilt_kl=float(W.value(pick_tilt)[e0]))
    c = res["checks"]
    print(f"  identity: total tilt + log p(C) = {c['optimal_total_tilt_plus_log_pC']:.6f} against KL {c['optimal_kl']:.6f} (optimal); "
          f"{c['random_total_tilt_plus_log_pC']:.6f} against {c['random_kl']:.6f} (random); the best plan for total tilt has KL {c['best_plan_for_tilt_kl']:.6f}")
    perm, v = W.best_static(rng, a.restarts); res["static"]["truth"] = dict(order=perm, kl=v); res["kl"]["static"] = v
    print(f"  {'static (best fixed order)':28s} {v:.4f}   ({v / Vopt[e0]:.2f} x optimal)   order {perm}", flush=True)
    x = den.sample(a.pilot, torch.Generator().manual_seed(11)).numpy()
    for sname in a.surrogates:
        ds = fit_markov(x, S, 1 if sname == "markov1" else 2); Ws = World(ds, a.constraint, N, proposal=a.proposal, pilot=a.pilot)
        assert (Ws.alive == W.alive).all()
        Vs, ps = Ws.solve(Ws.eps); Vsr = Ws.value_random()
        record(f"{sname}:optimal", ps); record(f"{sname}:rollout_random", Ws.argmin_pick(Ws.lookahead(Ws.eps, Vsr)))
        perm_s, _ = Ws.best_static(rng, a.restarts); vs = W.static_value(perm_s); res["static"][sname] = dict(order=perm_s, kl=vs); res["kl"][f"{sname}:static"] = vs
        print(f"  {sname + ':static':28s} {vs:.4f}   ({vs / Vopt[e0]:.2f} x optimal)   order {perm_s}", flush=True)
        # true tilt now + the surrogate's expected total tilt afterwards (expectation under the true proposal)
        T_opt = np.where(Ws.alive, Vs - np.where(Ws.alive, Ws.lt, 0.0), 0.0); T_rand = np.where(Ws.alive, Vsr - np.where(Ws.alive, Ws.lt, 0.0), 0.0)
        record(f"mix-{sname}:optimal", W.argmin_pick(W.lookahead(W.tilt, T_opt))); record(f"mix-{sname}:rollout_random", W.argmin_pick(W.lookahead(W.tilt, T_rand)))
        res["kl"][f"{sname}:own_optimal_value"] = float(Vs[e0]); res["kl"][f"{sname}:own_random_value"] = float(Vsr[e0])
        print(f"  (inside {sname}: optimal {Vs[e0]:.4f}, random {Vsr[e0]:.4f}, log p(C) {Ws.log_pC:.3f})", flush=True)
        del Ws
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(a.out, "w"), indent=1); print(f"{res['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
