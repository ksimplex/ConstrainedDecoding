"""How wrong is the twist, and does a first-order chain stand-in fix it?

The sampler's twist is the constraint's probability under a stand-in for the model given the revealed tokens. This
script compares, on states drawn from the constrained law and partly revealed at random, the exact value
psi_true = P(C | revealed) with three stand-ins:

  product       independent positions with the model's current marginals (the twist in use)
  chain_exact   a first-order chain whose adjacent-pair conditionals p(x_{i+1} | x_i, revealed) are the model's own:
                the best any first-order chain can do (exact on a Markov chain)
  chain_pilot   the same chain built without extra denoiser calls: current marginals times a lift table
                L(a, b) = p(a, b) / (p(a) p(b)) for adjacent tokens, estimated once (here from the unconditional
                model), in four versions by whether the outer neighbours of the pair are revealed

A constant error in log psi is harmless; what moves the weights is its spread across states. Reported per reveal
fraction: the standard deviation over states of log psi_standin - log psi_true (nats), and its mean.
DFA constraints only (optionally restricted to a set of active positions).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as cp

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)


def dfa_of(cname, den, N):
    """(WDFA, active mask) for the DFA constraints of compare.make_constraint."""
    from twistroute import compile as C
    cl, route, truth = cp.make_constraint(cname, den, N)
    act = np.ones(N, dtype=bool)
    if cname in ("window", "two_window"):
        act = np.zeros(N, dtype=bool)
        w = max(8, N // 10)
        if cname == "window":
            lo = N // 2 - w // 2; act[lo:lo + w] = True
        else:
            w = 2 * (w // 2); lo = N // 4 - w // 4
            act[lo:lo + w // 2] = True; act[lo + N // 2:lo + N // 2 + w // 2] = True
        d = C.dfa_at_least_k(0, truth.info["k"], den.S)
    else:
        d = {"forbidden": lambda: C.dfa_forbidden([(0, 0), (1, 2)], 3), "counting": lambda: C.dfa_exactly_k(0, N // 3, 3),
             "dyck": lambda: C.dfa_dyck1(N)}[cname]()
    return d, act, truth


class Twists:
    def __init__(self, den, d, act):
        self.den, self.N, self.H, self.S, self.Q = den, den.N, den.H, den.S, d.n
        self.act = act
        self.delta = torch.full((d.n, den.S), -1, dtype=torch.long)
        for (s, y), t in d.delta.items():
            self.delta[s, y] = t
        self.acc = torch.zeros(d.n, dtype=torch.float64); self.acc[list(d.F)] = 1.0
        self.q0 = d.q0

    def _move(self, f, b, i):
        """Apply the DFA move for token b at position i to the last axis (Q) of f; dead moves drop the mass."""
        if not self.act[i]:
            return f
        dst = self.delta[:, b]; ok = dst >= 0
        out = torch.zeros_like(f)
        out.index_add_(-1, dst[ok], f[..., ok])
        return out

    def true(self, x):
        """log P(C | revealed tokens of x), exactly: forward pass over (hidden state, DFA state)."""
        den, B = self.den, x.shape[0]
        f = torch.zeros(B, self.H, self.Q, dtype=torch.float64); f[:, :, self.q0] = den.pi[None]
        logs = torch.zeros(B, dtype=torch.float64)
        for i in range(self.N):
            g = torch.zeros_like(f)
            for b in range(self.S):
                keep = ((x[:, i] == b) | (x[:, i] == den.mask_id)).double()[:, None, None]
                g = g + self._move(f * den.emit[i, :, b][None, :, None], b, i) * keep
            s = g.sum((1, 2)); logs = logs + torch.log(s); g = g / s[:, None, None].clamp_min(1e-300)
            f = torch.einsum("bhq,hk->bkq", g, den.trans[i]) if i < self.N - 1 else g
        num = logs + torch.log((f.sum(1) * self.acc[None]).sum(-1))
        return num - den.log_joint(x)

    def _marg(self, x):
        mu = torch.exp(self.den.log_cond(x))
        rev = x != self.den.mask_id
        oh = torch.nn.functional.one_hot(torch.where(rev, x, torch.zeros_like(x)), self.S).double()
        return torch.where(rev[..., None], oh, mu), rev

    def product(self, x):
        mu, _ = self._marg(x)
        f = torch.zeros(x.shape[0], self.Q, dtype=torch.float64); f[:, self.q0] = 1.0
        for i in range(self.N):
            f = sum(self._move(f * mu[:, i, b][:, None], b, i) for b in range(self.S))
        return torch.log((f * self.acc[None]).sum(-1))

    def pair_conditionals(self, x):
        """(B, N-1, S, S): p(x_{i+1} = b | x_i = a, revealed), the model's own adjacent-pair conditionals."""
        den = self.den
        e = den._evidence(x); ahat, _ = den._forward(e); bk = den._backward(e)
        rev = x != den.mask_id
        out = torch.empty(x.shape[0], self.N - 1, self.S, self.S, dtype=torch.float64)
        for i in range(self.N - 1):
            left = ahat[:, i][:, :, None] * den.emit[i][None]                               # (B, H, a)
            right = den.emit[i + 1][None] * bk[:, i + 1][:, :, None]                        # (B, H', b)
            j = torch.einsum("bha,hk,bkc->bac", left, den.trans[i], right)
            r1 = rev[:, i + 1]
            if r1.any():                                                                     # next token known
                oh = torch.nn.functional.one_hot(torch.where(r1, x[:, i + 1], torch.zeros_like(x[:, i + 1])), self.S).double()
                j = torch.where(r1[:, None, None], oh[:, None, :].expand(-1, self.S, -1), j)
            out[:, i] = j / j.sum(-1, keepdim=True).clamp_min(1e-300)
        return out

    def lifts(self, n=4000, seed=0):
        """Four lift tables L_c(a, b) by context c = 2 * [left outer neighbour revealed] + [right outer neighbour
        revealed], estimated on model samples as the mean over pairs of p(a, b | rev) / (p(a | rev) p(b | rev))."""
        den = self.den; g = torch.Generator().manual_seed(seed)
        xs = den.sample(n, g)
        L = torch.zeros(4, self.S, self.S, dtype=torch.float64); cnt = torch.zeros(4)
        for c in range(4):
            x = xs.clone(); keep = torch.rand(x.shape, generator=g) < 0.5
            i = self.N // 2
            keep[:, i] = False; keep[:, i + 1] = False; keep[:, i - 1] = bool(c & 2); keep[:, i + 2] = bool(c & 1)
            x[~keep] = den.mask_id
            Qc = self.pair_conditionals(x)[:, i]                                             # p(b | a, rev)
            mu = torch.exp(den.log_cond(x))                                                  # p(. | rev) at every position
            L[c] = (Qc / mu[:, i + 1][:, None, :].clamp_min(1e-300)).mean(0)
        return L

    def chain(self, x, Qc):
        """log P(C | revealed) under a first-order chain with transitions Qc (B, N-1, S, S) and first marginal mu_0."""
        mu, _ = self._marg(x)
        B = x.shape[0]
        f = torch.zeros(B, self.S, self.Q, dtype=torch.float64)
        base = torch.zeros(B, self.Q, dtype=torch.float64); base[:, self.q0] = 1.0
        for a in range(self.S):
            f[:, a] = self._move(base * mu[:, 0, a][:, None], a, 0)
        for i in range(self.N - 1):
            g = torch.einsum("baq,bac->bcq", f, Qc[:, i])
            f = torch.stack([self._move(g[:, b], b, i + 1) for b in range(self.S)], 1)
        return torch.log((f.sum(1) * self.acc[None]).sum(-1))

    def pilot_transitions(self, x, L):
        mu, rev = self._marg(x)
        out = torch.empty(x.shape[0], self.N - 1, self.S, self.S, dtype=torch.float64)
        for i in range(self.N - 1):
            lrev = rev[:, i - 1] if i > 0 else torch.ones_like(rev[:, 0])
            rrev = rev[:, i + 2] if i + 2 < self.N else torch.ones_like(rev[:, 0])
            Lc = L[(2 * lrev.long() + rrev.long())]                                         # (B, S, S)
            q = mu[:, i + 1][:, None, :] * torch.where((rev[:, i] | rev[:, i + 1])[:, None, None], torch.ones_like(Lc), Lc)
            out[:, i] = q / q.sum(-1, keepdim=True).clamp_min(1e-300)
        return out


def run(mname, cname, N, n_states, seed):
    S = 2 if cname == "dyck" else 3
    den = cp.make_model(mname, N, S); d, act, truth = dfa_of(cname, den, N)
    tw = Twists(den, d, act); L = tw.lifts(seed=seed)
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    y = torch.tensor(truth.sample(n_states, rng))
    rows = []
    for f in FRACS:
        x = y.clone(); x[torch.rand(x.shape, generator=g) >= f] = den.mask_id
        lt = tw.true(x)
        est = dict(product=tw.product(x), chain_exact=tw.chain(x, tw.pair_conditionals(x)), chain_pilot=tw.chain(x, tw.pilot_transitions(x, L)))
        row = dict(model=mname, constraint=cname, N=N, frac=f, n=n_states, log_psi_true_mean=float(lt.mean()), log_pC=truth.log_pC)
        for k, v in est.items():
            e = (v - lt); ok = torch.isfinite(e)
            row[f"sd_{k}"] = float(e[ok].std()) if ok.sum() > 1 else float("nan"); row[f"mean_{k}"] = float(e[ok].mean()); row[f"dead_{k}"] = int((~ok).sum())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["hidden_0.85", "hidden_0.85_n0.15", "hidden_0.85_n0.5", "chain_0.85"])
    ap.add_argument("--constraints", nargs="+", default=["counting", "window", "two_window", "forbidden", "dyck"])
    ap.add_argument("--out", default="twist_error.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    rows, t0 = [], time.time()
    for m in a.models:
        for c in a.constraints:
            rows += run(m, c, a.N, a.n, a.seed)
            json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"), indent=1)
            print(f"[{time.time() - t0:6.0f}s] {m} {c}: sd of log-twist error at 25% revealed: " +
                  ", ".join(f"{k} {next(r for r in rows[-len(FRACS):] if r['frac'] == 0.25)['sd_' + k]:.3f}" for k in ("product", "chain_exact", "chain_pilot")), flush=True)


if __name__ == "__main__":
    main()
