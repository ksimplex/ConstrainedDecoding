"""What does raising the order of the chain twist buy, and what does it cost?

Stand-ins for the model's joint over the masked positions, all with the constraint summed exactly:
    product        independent positions (order 0: the twist in use)
    chain 1        first-order chain: each position depends on the one before it
    chain 2        second-order chain: each position depends on the two before it
each chain in two versions: `exact` uses the model's own conditionals given the revealed tokens (the best a chain of
that order can do; oracle models only), `ratio` uses a table pooled over positions before the run (pairs K(a, b),
triples K2(a, b, c)) and the current marginals, with no denoiser call:
    Q_i(a, b)    proportional to  mu_{i+1}(b) K(a, b)     / sum_a'     mu_i(a') K(a', b)
    R_i(a, b, c) proportional to  mu_{i+1}(c) K2(a, b, c) / sum_a',b'  mu_{i-1}(a') mu_i(b') K2(a', b', c)

Measured on states drawn from the exact constrained law and revealed at random, per stand-in:
    sd_whole     spread over states of log psi_standin - log psi_true (nats): what moves the weights
    pos_ratio    per masked position, sqrt(KL(stand-in's constrained marginal || exact conditional)) relative to the
                 product twist's, median over the costlier half of positions (1 = no better than the product twist)
    mean_kl      mean over those positions of that KL (nats): the error a proposal built on the stand-in would add
    ms           milliseconds per state for one evaluation of the twist (one CPU thread)
Automaton constraints only.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twist_error as te                                                                           # noqa: E402
from pulib.cells import make_model, make_constraint, n_symbols, stream_seed                         # noqa: E402
from pulib.orders import kl, exact_conditionals                                                     # noqa: E402
from pulib.summary import provenance                                                                # noqa: E402

FRACS = (0.1, 0.25, 0.5, 0.75)


class Chains:
    """Chain twists of order 1 and 2 for an automaton, with per-position log-weights `theta` for the marginals."""

    def __init__(self, wdfa, active, S):
        e = torch.tensor(wdfa.edges, dtype=torch.long).reshape(-1, 3)
        self.src, self.sym, self.dst = e[:, 0], e[:, 1], e[:, 2]
        self.Q, self.q0, self.S = wdfa.n, wdfa.q0, S
        self.F = torch.zeros(self.Q, dtype=torch.float64); self.F[sorted(wdfa.F)] = 1.0
        self.active = np.asarray(active, dtype=bool)

    def read(self, g, i):
        """g[..., c, q]: class c at position i, automaton in state q before reading it -> after reading it."""
        if not self.active[i]:
            return g
        v = g[..., self.sym, self.src]
        out = torch.zeros(g.shape[:-2] + (self.S * self.Q,), dtype=g.dtype)
        return out.index_add(-1, self.sym * self.Q + self.dst, v).view(g.shape)

    def order1(self, mu, Q1, theta=None):
        B, N, S = mu.shape
        w = (lambda i: 1.0) if theta is None else (lambda i: torch.exp(theta[:, i]))
        start = torch.zeros(self.Q, dtype=torch.float64); start[self.q0] = 1.0
        f = self.read((mu[:, 0] * w(0))[:, :, None] * start, 0)
        logs = torch.zeros(B, dtype=torch.float64)
        for i in range(N - 1):
            s = f.sum((1, 2)); logs = logs + torch.log(s); f = f / s.clamp_min(1e-300)[:, None, None]
            g = torch.einsum("zaq,zac->zcq", f, Q1[:, i])
            f = self.read(g * (w(i + 1)[:, :, None] if theta is not None else 1.0), i + 1)
        return logs + torch.log((f.sum(1) * self.F).sum(-1))

    def order2(self, mu, Q1, R, theta=None):
        """R (B, N - 2, S, S, S): R[:, i - 1, a, b, c] = p(class c at i + 1 | a at i - 1, b at i)."""
        B, N, S = mu.shape
        w = (lambda i: 1.0) if theta is None else (lambda i: torch.exp(theta[:, i]))
        start = torch.zeros(self.Q, dtype=torch.float64); start[self.q0] = 1.0
        f0 = self.read((mu[:, 0] * w(0))[:, :, None] * start, 0)                               # (B, a, q)
        g = f0[:, :, None, :] * (Q1[:, 0] * (w(1)[:, None, :] if theta is not None else 1.0))[:, :, :, None]
        f = self.read(g, 1)                                                                     # (B, a, b, q)
        logs = torch.zeros(B, dtype=torch.float64)
        for i in range(1, N - 1):
            s = f.sum((1, 2, 3)); logs = logs + torch.log(s); f = f / s.clamp_min(1e-300)[:, None, None, None]
            g = torch.einsum("zabq,zabc->zbcq", f, R[:, i - 1])
            f = self.read(g * (w(i + 1)[:, None, :, None] if theta is not None else 1.0), i + 1)
        return logs + torch.log((f.sum((1, 2)) * self.F).sum(-1))

    def marginals(self, fn, mu, *tables):
        with torch.enable_grad():
            theta = torch.zeros(mu.shape, dtype=torch.float64, requires_grad=True)
            g, = torch.autograd.grad(fn(mu, *tables, theta).sum(), theta)
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def triple_conditionals(den, x):
    """(B, N - 2, S, S, S): the model's own p(x_{i+1} = c | x_{i-1} = a, x_i = b, revealed tokens elsewhere)."""
    e = den._evidence(x); ahat, _ = den._forward(e); bk = den._backward(e)
    rev = x != den.mask_id
    B, N = x.shape; S = den.S
    out = torch.empty(B, N - 2, S, S, S, dtype=torch.float64)
    for i in range(1, N - 1):
        left = ahat[:, i - 1][:, :, None] * den.emit[i - 1][None]                              # (B, h, a)
        right = den.emit[i + 1][None] * bk[:, i + 1][:, :, None]                               # (B, l, c)
        j = torch.einsum("zha,hk,kd,kl,zlc->zadc", left, den.trans[i - 1], den.emit[i], den.trans[i], right)
        r1 = rev[:, i + 1]
        if r1.any():
            oh = torch.nn.functional.one_hot(torch.where(r1, x[:, i + 1], torch.zeros_like(x[:, i + 1])), S).double()
            j = torch.where(r1[:, None, None, None], oh[:, None, None, :].expand(-1, S, S, -1), j)
        out[:, i - 1] = j / j.sum(-1, keepdim=True).clamp_min(1e-300)
    return out


def pooled_tables(den, n, seed):
    """K (S, S) and K2 (S, S, S): next class given the previous one / two, pooled over positions, from model samples."""
    S = den.S; xs = den.sample(n, torch.Generator().manual_seed(seed))
    J = torch.bincount((xs[:, :-1] * S + xs[:, 1:]).reshape(-1), minlength=S * S).double().view(S, S) + 1e-3
    J2 = torch.bincount((xs[:, :-2] * S * S + xs[:, 1:-1] * S + xs[:, 2:]).reshape(-1), minlength=S ** 3).double().view(S, S, S) + 1e-3
    return J / J.sum(-1, keepdim=True), J2 / J2.sum(-1, keepdim=True)


def ratio_tables(mu, K, K2):
    """Transitions of both orders from the current marginals and the pooled tables (no denoiser call)."""
    kbar = torch.einsum("zia,ac->zic", mu[:, :-1], K).clamp_min(1e-300)
    Q1 = mu[:, 1:, None, :] * K[None, None] / kbar[:, :, None, :]
    Q1 = Q1 / Q1.sum(-1, keepdim=True).clamp_min(1e-300)
    kbar2 = torch.einsum("zia,zib,abc->zic", mu[:, :-2], mu[:, 1:-1], K2).clamp_min(1e-300)
    R = mu[:, 2:, None, None, :] * K2[None, None] / kbar2[:, :, None, None, :]
    return Q1, R / R.sum(-1, keepdim=True).clamp_min(1e-300)


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    from twistroute.decode import constrained_marginals
    from twistroute.decode.schedule import clamp_revealed
    den = make_model(mname, N, n_symbols(cname))
    cl, route, truth = make_constraint(cname, den, N)
    ref = te.Twists(den, truth.wdfa, truth.active); ch = Chains(truth.wdfa, truth.active, den.S)
    seed = stream_seed(a["seed"], mname, cname, N, "twist_order")
    K, K2 = pooled_tables(den, a["pilot"], seed)
    g = torch.Generator().manual_seed(seed)
    y = torch.tensor(truth.sample(a["states"], np.random.default_rng(seed)))
    rows = []
    for f in FRACS:
        x = torch.where(torch.rand(y.shape, generator=g) < f, y, torch.full_like(y, den.mask_id))
        masked = x == den.mask_id
        mu, rev = ref._marg(x)
        lwc = clamp_revealed(cl.class_logw(den.logits(x)), x, cl.token_class[torch.where(masked, torch.zeros_like(x), x)], den.mask_id)
        Q1x, Rx = ref.pair_conditionals(x), triple_conditionals(den, x)
        Q1r, Rr = ratio_tables(mu, K, K2)
        true = ref.true(x)
        pi = exact_conditionals(ref, den, x, torch.exp(den.log_cond(x)))
        q0 = constrained_marginals(route, lwc)
        e0 = kl(q0, pi)
        ok = masked & torch.isfinite(e0)
        stake = ok & (kl(q0, torch.exp(den.log_cond(x))) > 1e-6)
        use = stake.clone(); use[:, 1:] |= stake[:, :-1] & ok[:, 1:]; use[:, :-1] |= stake[:, 1:] & ok[:, :-1]
        big = use & (e0 > e0[use].median())
        t0 = time.time(); p0 = route.logZ(lwc); t_prod = time.time() - t0
        row = dict(model=mname, constraint=cname, N=N, frac=f, states=int(a["states"]), positions=int(big.sum()))
        def record(tag, val, marg, secs):
            err = val - true; fin = torch.isfinite(err)
            row[f"sd_whole_{tag}"] = float(err[fin].std()); row[f"ms_{tag}"] = 1e3 * secs / len(x)
            if marg is not None:
                e = kl(marg, pi); m = big & torch.isfinite(e)
                row[f"pos_ratio_{tag}"] = float(torch.sqrt(e[m] / e0[m]).median()); row[f"mean_kl_{tag}"] = float(e[m].mean())
        record("product", p0, q0, t_prod)
        for tag, fn, tabs in (("chain1_ratio", ch.order1, (Q1r,)), ("chain1_exact", ch.order1, (Q1x,)),
                              ("chain2_ratio", ch.order2, (Q1r, Rr)), ("chain2_exact", ch.order2, (Q1x, Rx))):
            t0 = time.time(); val = fn(mu, *tabs); secs = time.time() - t0
            record(tag, val, ch.marginals(fn, mu, *tabs), secs)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--models", nargs="+", default=["hidden_0.85", "hidden_0.85_n0.15", "hidden_0.85_n0.5", "chain_0.85"])
    ap.add_argument("--constraints", nargs="+", default=["counting", "dyck", "forbidden", "window", "two_window"])
    ap.add_argument("--pilot", type=int, default=20000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default="twist_order.json")
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
