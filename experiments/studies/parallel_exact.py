"""Parallel reveals, exactly: the best trade of steps against error at a length where every state can be tabulated.

A step reveals a set A of masked positions. Their classes are drawn jointly from the proposal (the product twist's
constrained law at A, or the chain twist's), using the model's marginals at the current state: the dependence between
the positions of A is ignored. Three laws over finished sequences follow from a rule (state -> set):

    s          the sampler's law (one particle, no weights)
    P_par      the law of the same reveals drawn from the reference law with no constraint (the parallel process): the
               product of the model's marginals at the set, or with --reference chain the chain law's joint at the
               set; the weights correct s towards P_par given C, and Z_hat estimates P_par(C)
    p(. | C)   the target: the model's law given C (what one token per step samples)

    one particle   KL(s || p(. | C)) = sum over steps of E_s[ KL(q_A || exact conditional of A given C) ]   (additive)
    bias           KL(P_par(. | C) || p(. | C)); log P_par(C) / p(C). No number of particles removes it.
    spread         chi-square of P_par(. | C) against s: the noise that particles do remove, about spread / (2 K) in the
                   KL of a returned sample for K particles (large-K expansion; quoted, not derived here)

Rules compared, all scored exactly (every finished sequence is enumerated):
    plan mu, lam  minimises [steps x mu + one-particle KL + lam x ignored dependence] by a backward pass over states and
                  sets. mu is the price of a step in nats; the ignored dependence of a step is KL(the model's joint of
                  the set || the reference law of the set), which is what the weights cannot see (zero for one
                  position). Both are swept; lam = 0 ignores the bias, large lam co-reveals only independent positions.
    sequential    the best one-token-per-step order (the plan restricted to single positions)
    greedy tau    at each state the largest set whose step error is at most tau (ties: the least error); a single
                  position of least error if none qualifies
    policy        the repository's budgeted schedules (`lex`, `sum` at budgets B)
For a tolerance eps on the KL of a returned sample, the particles are the smallest K whose exact output law meets it (class
OutputLaw; `compute`). The rule used before section 36ai is kept as `compute_old_rule`: particles K = spread / (2 (eps - bias)) (1 if one particle already
meets eps; impossible if bias >= eps), compute = expected steps x K.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, n_symbols                                                      # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402
from surrogate_plan import constraint_for                                                          # noqa: E402
from twistroute.decode import CouplingPolicy, fit_distance_kernel, COUNT_EDGES                     # noqa: E402
from twistroute.decode.policy import RevealState                                                   # noqa: E402
from twistroute.decode.schedule import clamp_revealed                                              # noqa: E402


class PWorld:
    def __init__(self, den, cname, N, proposal="product", pilot=2000, reference="product"):
        self.N, self.S, self.den = N, den.S, den; S = den.S
        self.cl, self.route, truth = constraint_for(cname, den, N); cl, route = self.cl, self.route
        self.log_pC = float(truth.log_pC); act = getattr(truth, "active", None)
        ref = te.Twists(den, truth.wdfa, act if act is not None else [True] * N)
        chain = None
        if proposal == "chain":
            from twistroute.decode import ChainTwist, fit_pair_kernel
            chain = ChainTwist(truth.wdfa, fit_pair_kernel(den, cl, pilot, torch.Generator().manual_seed(5)), active=act)
        free = None
        if reference == "chain":                            # the chain law with no constraint: the reference for sets of 2 or more
            from twistroute import compile as C
            from twistroute.decode import ChainTwist, fit_pair_kernel
            free = ChainTwist(C.dfa_at_least_k(0, 0, S), fit_pair_kernel(den, cl, pilot, torch.Generator().manual_seed(5)))
        B = self.B = (S + 1) ** N; pw = self.pw = (S + 1) ** np.arange(N); codes = self.codes = np.arange(B)
        dig = ((codes[:, None] // pw[None]) % (S + 1)).astype(np.int64); self.masked = dig == S; self.nm = self.masked.sum(1)
        self.pattern = (self.masked * (1 << np.arange(N))[None]).sum(1)
        self.x_all = x = torch.from_numpy(dig); mk = x == S
        ycls = cl.token_class[torch.where(mk, torch.zeros_like(x), x)]
        lc = den.log_cond(x).double()                                                         # tokens are the classes here
        lwc = clamp_revealed(lc, x, ycls, S)                                                  # (B, N, S), double throughout
        pm = torch.exp(lc).numpy(); lt = ref.true(x).numpy(); logp = den.log_joint(x).numpy()
        self.lt, self.logp = lt, logp; self.alive = alive = np.isfinite(lt) & (lt > -1e200)
        z0 = (route.logZ(lwc) if chain is None else chain.log_chain(lwc)).double().numpy()
        self.combos = {}                                                               # (pattern, set) -> tables
        self.by_pattern = {}
        onehot = torch.log(torch.eye(S, dtype=lwc.dtype))                               # 0 on the class, -inf elsewhere
        with np.errstate(divide="ignore", invalid="ignore"):
            for M in range(1, 1 << N):
                sel = codes[(self.pattern == M) & alive]
                if len(sel) == 0:
                    continue
                Mpos = [i for i in range(N) if M >> i & 1]; self.by_pattern[M] = []
                for r in range(1, len(Mpos) + 1):
                    for A in itertools.combinations(Mpos, r):
                        a = np.array(list(itertools.product(range(S), repeat=r)))       # (S^r, r)
                        off = ((a - S) * pw[list(A)][None]).sum(1); succ = sel[:, None] + off[None]
                        lognat = sum(np.log(np.clip(pm[sel][:, i, :][:, a[:, j]], 1e-300, None)) for j, i in enumerate(A))   # (n, S^r)
                        n, R = len(sel), len(a)
                        rows = lwc[torch.from_numpy(sel)].repeat_interleave(R, 0)       # (n R, N, S)
                        at = torch.from_numpy(a).repeat(n, 1)
                        if chain is None:
                            for j, i in enumerate(A):
                                rows[:, i] = onehot[at[:, j]]
                            zc = route.logZ(rows).double().numpy().reshape(n, R)
                            logq = lognat + zc - z0[sel][:, None]
                        else:
                            rr = torch.tensor(A)[None].expand(n * R, r)
                            logq = chain.log_clamped(rows, rr, at).numpy().reshape(n, R) - z0[sel][:, None]
                        if free is not None and r > 1:      # reference law of the set: the chain's joint instead of the product of marginals
                            rr = torch.tensor(A)[None].expand(n * R, r)
                            rows0 = lwc[torch.from_numpy(sel)].repeat_interleave(R, 0)
                            lognat = free.log_clamped(rows0, rr, at).numpy().reshape(n, R)
                            assert np.abs(np.exp(lognat).sum(1) - 1).max() < 1e-7
                        q = np.exp(logq); q[~alive[succ]] = 0.0
                        assert np.abs(q.sum(1) - 1).max() < 1e-7, (M, A, np.abs(q.sum(1) - 1).max())
                        q = q / q.sum(1, keepdims=True); pos = q > 0
                        logq = np.where(pos, np.log(np.where(pos, q, 1.0)), 0.0)
                        model = logp[succ] - logp[sel][:, None]                         # model's joint conditional of the set
                        target = model + np.where(alive[succ], lt[succ], 0.0) - lt[sel][:, None]
                        eps = (q * np.where(pos, logq - target, 0.0)).sum(1)
                        dep = (np.exp(model) * (model - lognat)).sum(1)     # KL(model's joint of the set || reference law of the set): what the weights cannot see
                        tilt = (q * np.where(pos, logq - lognat, 0.0)).sum(1); co = (q * np.where(pos, lognat - model, 0.0)).sum(1)
                        assert eps.min() > -1e-9
                        Abits = sum(1 << i for i in A)
                        self.combos[(M, Abits)] = dict(sel=sel, succ=succ, q=q, nat=np.exp(lognat), eps=np.clip(eps, 0, None), tilt=tilt, co=co, dep=np.clip(dep, 0, None), size=r)
                        self.by_pattern[M].append(Abits)
        self.order_up = sorted(self.by_pattern, key=lambda M: bin(M).count("1"))        # fewest masked first
        self.e0 = B - 1
        self.final = codes[self.nm == 0]

    def plan(self, mu, max_size=None, lam=0.0):
        """Rule minimising expected [mu per step + step error + lam x ignored dependence]; returns (value, pick (B,) set bits)."""
        V = np.zeros(self.B); pick = np.zeros(self.B, dtype=np.int64)
        for M in self.order_up:
            best = None
            for Ab in self.by_pattern[M]:
                c = self.combos[(M, Ab)]
                if max_size is not None and c["size"] > max_size:
                    continue
                tot = mu + c["eps"] + lam * c["dep"] + (c["q"] * V[c["succ"]]).sum(1)
                if best is None:
                    best = tot.copy(); bp = np.full(len(tot), Ab)
                else:
                    w = tot < best - 1e-11; best[w] = tot[w]; bp[w] = Ab     # ties go to the smaller set (enumerated first)
            V[c["sel"]] = best; pick[c["sel"]] = bp
        return V, pick

    def greedy(self, tau):
        pick = np.zeros(self.B, dtype=np.int64)
        for M in self.order_up:
            key = None
            for Ab in self.by_pattern[M]:
                c = self.combos[(M, Ab)]; e = c["eps"]
                k = np.where((e <= tau) | (c["size"] == 1), c["size"] * (e <= tau) - 1e-3 * np.minimum(e, 1e2), -np.inf)   # larger sets first, then less error
                if key is None:
                    key = k.copy(); bp = np.full(len(k), Ab)
                else:
                    w = k > key; key[w] = k[w]; bp[w] = Ab
            pick[c["sel"]] = bp
        return pick

    def score(self, pick, laws=False):
        """Exact laws of a rule. Returns a dict of the quantities in the module docstring."""
        rs = np.zeros(self.B); rn = np.zeros(self.B); rs[self.e0] = 1.0; rn[self.e0] = 1.0
        steps = 0.0; kl1 = 0.0; sizes = 0.0; worst = np.zeros(self.B, dtype=np.int64)
        for M in reversed(self.order_up):
            for Ab in self.by_pattern[M]:
                c = self.combos[(M, Ab)]; use = pick[c["sel"]] == Ab
                if not use.any():
                    continue
                sel = c["sel"][use]; w = rs[sel]
                steps += w.sum(); kl1 += (w * c["eps"][use]).sum(); sizes += (w * c["size"]).sum()
                np.add.at(rs, c["succ"][use], w[:, None] * c["q"][use]); np.add.at(rn, c["succ"][use], rn[sel][:, None] * c["nat"][use])
                np.maximum.at(worst, c["succ"][use].ravel(), np.repeat(worst[sel] + 1, c["succ"].shape[1]))
        f = self.final; inC = self.alive[f]; s = rs[f]; n = rn[f]
        tgt = np.where(inC, np.exp(self.logp[f] - self.log_pC), 0.0)
        # the parallel process is followed through states consistent with C only, which is all of its mass on C
        assert abs(s.sum() - 1) < 1e-7 and n.sum() < 1 + 1e-7 and s[~inC].sum() < 1e-9 and abs(tgt.sum() - 1) < 1e-7
        pC = n[inC].sum(); npar = np.where(inC, n / pC, 0.0)
        kl = lambda a, b: float((a[a > 0] * (np.log(a[a > 0]) - np.log(b[a > 0]))).sum())
        ok = npar > 0
        out = dict(steps=float(steps), max_steps=int(worst[f][s > 1e-15].max()), mean_set=float(sizes / steps), kl_one=float(kl1), kl_one_check=kl(s, tgt),
                   bias=kl(npar, tgt), log_z_bias=float(np.log(pC) - self.log_pC), spread=float((npar[ok] ** 2 / s[ok]).sum() - 1), kl_s_to_par=kl(s, npar))
        if laws:
            out["laws"] = (s, npar, tgt)
        return out

    def repo_policy(self, name, budget, kern):
        N, S = self.N, self.S
        pol = CouplingPolicy(kern, 0.0, priority=name, rate=None if budget is None else budget / N, **({"max_set": 1} if budget is None else {}))
        g = torch.Generator().manual_seed(7); pick = np.zeros(self.B, dtype=np.int64)
        for m in range(1, N + 1):
            idx = self.codes[(self.nm == m) & self.alive]
            for lo in range(0, len(idx), 20000):
                sl = idx[lo:lo + 20000]; x = self.x_all[sl]; mk = x == S
                ycls = self.cl.token_class[torch.where(mk, torch.zeros_like(x), x)]
                st = RevealState(x, ycls, self.cl.class_logw(self.den.logits(x)), S, self.cl, self.route)
                r, valid = pol.reveal(st, N - m, N, g)
                valid = torch.ones_like(r, dtype=torch.bool) if valid is None else valid
                bits = ((1 << r) * valid).sum(1).numpy()
                assert (bits > 0).all() and ((bits & ~self.pattern[sl]) == 0).all()
                pick[sl] = bits
        return pick


def compute_for(sc, eps_tol):
    """(particles, compute) for a tolerance on the KL of a returned sample; (inf, inf) if the bias alone exceeds it."""
    if sc["kl_one"] <= eps_tol:
        return 1.0, sc["steps"]
    if sc["bias"] >= eps_tol:
        return float("inf"), float("inf")
    K = max(1.0, sc["spread"] / (2 * (eps_tol - sc["bias"])))
    return K, sc["steps"] * K


class OutputLaw:
    """Exact error of the returned sample for K particles (no resampling): the rule is deterministic, so a finished sequence has
    one path and one weight w = P_par(x | C) / s(x), and the output law is q_K(x) = K s(x) w(x) E[1 / (w(x) + S_{K-1})] with S_{K-1}
    the sum of K - 1 independent weights (section 36ai). The expectation uses R draws; KL(q_K || target) is cached per K."""

    def __init__(self, s, npar, tgt, R=4000, seed=0):
        self.s, self.tgt, self.R = s, tgt, R; self.sup = s > 0; self.w = np.where(self.sup, npar / np.where(self.sup, s, 1.0), 0.0)
        self.rng = np.random.default_rng(seed); self.cache = {}; self.live = np.nonzero(self.sup & (self.w > 0))[0]; self.p = s / s.sum()

    def kl(self, K):
        K = int(K)
        if K not in self.cache:
            if K == 1:
                qK = self.s.copy()
            else:
                Ssum = np.zeros(self.R)
                for lo in range(0, K - 1, 256):                                            # draws in chunks: R x (K - 1) weights
                    Ssum += self.w[self.rng.choice(len(self.s), size=(self.R, min(256, K - 1 - lo)), p=self.p)].sum(1)
                qK = np.zeros(len(self.s)); wl = self.w[self.live]
                for lo in range(0, len(self.live), 1000):
                    sl = self.live[lo:lo + 1000]; qK[sl] = K * self.s[sl] * self.w[sl] * (1.0 / (self.w[sl][:, None] + Ssum[None])).mean(1)
                qK = qK / qK.sum()
            m = qK > 0
            self.cache[K] = float((qK[m] * (np.log(qK[m]) - np.log(self.tgt[m]))).sum()) if np.all(self.tgt[m] > 0) else float("inf")
        return self.cache[K]

    def particles(self, eps_tol, bias, kmax=1 << 16):
        """Smallest K with KL(q_K || target) <= eps_tol (inf if the bias alone exceeds it or kmax is not enough)."""
        if self.kl(1) <= eps_tol:
            return 1.0
        if bias >= eps_tol:
            return float("inf")
        hi = 2
        while self.kl(hi) > eps_tol:
            hi *= 2
            if hi > kmax:
                return float("inf")
        lo = hi // 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if self.kl(mid) <= eps_tol else (mid, hi)
        return float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--model", default="hidden_0.85"); ap.add_argument("--constraint", default="counting")
    ap.add_argument("--proposal", choices=("product", "chain"), default="product"); ap.add_argument("--pilot", type=int, default=2000)
    ap.add_argument("--reference", choices=("product", "chain"), default="product", help="law of a co-revealed set that the weights correct towards: product of marginals, or the chain law (sets of 2 or more)")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0, 3.0, 10.0, 100.0])
    ap.add_argument("--mus", type=float, nargs="+", default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--taus", type=float, nargs="+", default=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.03, 0.3, 1.0, 3.0]); ap.add_argument("--tolerances", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--out", default="parallel_exact.json"); ap.add_argument("--R", type=int, default=4000, help="draws for the exact K-particle output law")
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time(); N = a.N; S = n_symbols(a.constraint)
    den = make_model(a.model, N, S); W = PWorld(den, a.constraint, N, a.proposal, a.pilot, a.reference)
    print(f"{a.constraint} {a.model} N={N}, {a.proposal} proposal, {a.reference} reference: {W.B} states, {len(W.combos)} (pattern, set) tables, log p(C) = {W.log_pC:.3f}, built in {time.time() - t0:.0f} s", flush=True)
    rules = {}
    Vseq, pseq = W.plan(0.0, max_size=1); rules["sequential (best order)"] = pseq
    for lam in a.lams:
        for mu in a.mus:
            rules[f"plan mu={mu:g} lam={lam:g}"] = W.plan(mu, lam=lam)[1]
    for tau in a.taus:
        rules[f"greedy tau={tau:g}"] = W.greedy(tau)
    kern, _ = fit_distance_kernel(den, N, a.pilot, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
    for name in ("lex", "sum"):
        rules[f"policy {name} sequential"] = W.repo_policy(name, None, kern)
        for bud in a.budgets:
            rules[f"policy {name} B={bud:g}"] = W.repo_policy(name, bud, kern)
    res = dict(args=vars(a), provenance=provenance(), log_pC=W.log_pC, rows=[])
    print("rule | expected steps (max) | mean set | one-particle KL | bias KL | log Z bias | spread | " + " | ".join(f"compute at eps={e:g}" for e in a.tolerances))
    for name, pick in rules.items():
        sc = W.score(pick, laws=True); assert abs(sc["kl_one"] - sc["kl_one_check"]) < 1e-7, (name, sc)
        law = OutputLaw(*sc.pop("laws"), R=a.R); sc["compute_old_rule"] = {f"{e:g}": compute_for(sc, e) for e in a.tolerances}      # spread / (2 (eps - bias)): superseded
        sc["compute"] = {f"{e:g}": (lambda K: (K, sc["steps"] * K))(law.particles(e, sc["bias"])) for e in a.tolerances}
        sc["kl_by_particles"] = {str(K): v for K, v in sorted(law.cache.items())}
        sc["rule"] = name; res["rows"].append(sc)
        print(f"  {name:28s} | {sc['steps']:5.2f} ({sc['max_steps']}) | {sc['mean_set']:4.2f} | {sc['kl_one']:.4f} | {sc['bias']:.4f} | {sc['log_z_bias']:+.4f} | {sc['spread']:.4f} | "
              + " | ".join(f"{sc['compute'][f'{e:g}'][1]:7.1f}" for e in a.tolerances), flush=True)
    assert abs(res["rows"][0]["bias"]) < 1e-9 and abs(res["rows"][0]["log_z_bias"]) < 1e-9          # one token per step has no bias
    res["sequential_optimal_kl"] = float(Vseq[W.e0]); res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1); print(f"{res['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
