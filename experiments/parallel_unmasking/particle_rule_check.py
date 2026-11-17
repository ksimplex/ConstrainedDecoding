"""How many particles for a tolerance on the returned sample's law?  Check of the rule of thumb, 8 positions, no resampling.

One token per step in a fixed order: the sampler's law q over complete sequences and the exact constrained law pi are both
known, so the importance weight w = pi / q is known for every sequence.  With K particles and one output drawn in proportion
to the weights, the output law is  q_K(x) = K q(x) w(x) E[1 / (w(x) + S_{K-1})],  S_{K-1} the sum of K - 1 independent weights.
The expectation is estimated with R draws (shared across x).  Compared:
    KL(q_K || pi)                     the error of the returned sample's law (what the tolerance is on)
    chi2 / (2K)                       the rule quoted earlier in this project, chi2 = E_q[w^2] - 1   (WRONG for this quantity)
    Var_pi(w) / (2 K^2)               the correct large-K expansion: the self-normalised output law is
                                      q_K(x) = pi(x) [1 + (E_pi w - w(x)) / K + O(1/K^2)], and KL is second order in the deviation
    exp(4 kl) (exp(2 kl) - 1) / (2 K^2)   the same with Var_pi(w) replaced by its value for normal log-weights, kl = KL(q || pi)
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surrogate_plan import World                                                                   # noqa: E402
from pulib.cells import make_model, n_symbols                                                      # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--model", default="hidden_0.85"); ap.add_argument("--constraint", default="forbidden")
    ap.add_argument("--proposal", default="product"); ap.add_argument("--match", type=int, default=0); ap.add_argument("--R", type=int, default=4000)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]); ap.add_argument("--out", default="particle_rule_check.json")
    a = ap.parse_args(); N = a.N; S = n_symbols(a.constraint); rng = np.random.default_rng(0)
    den = make_model(a.model, N, S); W = World(den, a.constraint, N, proposal=a.proposal, match=a.match)
    xs = np.array(np.meshgrid(*[np.arange(S)] * N, indexing="ij")).reshape(N, -1).T                 # all complete sequences
    code = (xs * W.pw[None]).sum(1); ok = W.alive[code]
    lp = den.log_joint(torch.from_numpy(xs)).numpy(); pi = np.where(ok, np.exp(lp), 0.0); pi = pi / pi.sum()
    res = dict(args=vars(a), provenance=provenance(), log_pC=W.log_pC, orders={})
    for oname, perm in (("left to right", np.arange(N)), ("random order A", rng.permutation(N)), ("random order B", rng.permutation(N))):
        q = np.ones(len(xs)); st = np.full(len(xs), W.e0)
        for i in perm:
            q = q * W.q[st, i, xs[:, i]]; st = st + (xs[:, i] - S) * W.pw[i]
        assert abs(q.sum() - 1) < 1e-9 and np.all(q[~ok] < 1e-12)
        sup = ok & (q > 0); w = np.where(sup, pi / np.where(sup, q, 1.0), 0.0)
        kl = float((q[sup] * np.log(q[sup] / pi[sup])).sum()); chi2 = float((q * w ** 2).sum() - 1); varpi = float((pi * w ** 2).sum() - (pi * w).sum() ** 2)
        assert abs(kl - float(W.static_value(perm))) < 1e-8                                   # the same number as the exact tables
        rows = []
        for K in a.Ks:
            if K == 1:
                qK = q.copy()
            else:
                idx = rng.choice(len(xs), size=(a.R, K - 1), p=q); Ssum = w[idx].sum(1)             # R draws of the sum of K - 1 weights
                qK = np.zeros(len(xs))
                for lo in range(0, len(xs), 2000):
                    sl = slice(lo, lo + 2000); qK[sl] = K * q[sl] * w[sl] * (1.0 / (w[sl][:, None] + Ssum[None])).mean(1)
                qK = qK / qK.sum()
            m = sup & (qK > 0); klK = float((qK[m] * np.log(qK[m] / pi[m])).sum())
            rows.append(dict(K=K, kl_output=klK, chi2_rule=chi2 / (2 * K), second_order=varpi / (2 * K * K), second_order_lognormal=float(np.exp(4 * kl) * (np.exp(2 * kl) - 1) / (2 * K * K))))
        res["orders"][oname] = dict(perm=[int(v) for v in perm], kl_one_particle=kl, chi2=chi2, var_pi_w=varpi, var_pi_w_if_lognormal=float(np.exp(4 * kl) * (np.exp(2 * kl) - 1)), rows=rows)
        print(f"{a.constraint} {a.model} N={N} {a.proposal}, {oname}: KL of one particle {kl:.4f}, chi2 {chi2:.4f}, Var_pi(w) {varpi:.4f} (normal log-weights would give {np.exp(4 * kl) * (np.exp(2 * kl) - 1):.4f})")
        print("   K:                         " + "  ".join(f"{r['K']:>7d}" for r in rows)); print("   KL of the output law:      " + "  ".join(f"{r['kl_output']:.5f}" for r in rows))
        print("   Var_pi(w) / (2 K^2):       " + "  ".join(f"{r['second_order']:.5f}" for r in rows)); print("   same, normal log-weights:  " + "  ".join(f"{r['second_order_lognormal']:.5f}" for r in rows))
        print("   chi2 / (2K) (old rule):    " + "  ".join(f"{r['chi2_rule']:.5f}" for r in rows))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
