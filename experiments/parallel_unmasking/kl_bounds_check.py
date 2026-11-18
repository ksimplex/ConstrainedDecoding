"""Bounds on the error of the returned sample's law, checked where the law itself is known (8 positions, one token per step, no resampling).

q_K = law of the output when K particles are drawn from the sampler q and one is returned in proportion to its weight w = pi / q.
    exact          KL(q_K || pi), by enumeration (the quantity the tolerance is on)
    upper          -E log(Z_hat_K / Z): the KL between the sampler and the target on the extended space of all K particles and the
                   index returned; an upper bound by data processing. Needs Z = p(C).
    upper, est.    log(mean Z_hat) - mean(log Z_hat) over n runs: the same without knowing Z. Biased low when the mean of Z_hat is
                   itself underestimated (heavy right tail); mean and sd over repeated sets of n runs are reported.
    moment         Var_pi(w) / (2 K^2), the leading term for large K; `moment, est.` is its plug-in estimate from the n K weights of n runs,
                   (m3 - m2^2) / (2 K^2) with m_k = mean(w^k) / mean(w)^k.
    lower, 1 / 2   max over positions (pairs of positions) of the KL between the marginals of q_K and of pi: lower bounds by data processing.
                   Exact here; at scale they need reference marginals of pi.
"""
import argparse, itertools, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surrogate_plan import World                                                                   # noqa: E402
from pulib.cells import make_model, n_symbols                                                      # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--model", default="hidden_0.85"); ap.add_argument("--constraints", nargs="+", default=["counting", "dyck", "forbidden"])
    ap.add_argument("--proposals", nargs="+", default=["product", "chain"]); ap.add_argument("--match", type=int, default=5); ap.add_argument("--R", type=int, default=20000)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 4, 16]); ap.add_argument("--n", type=int, default=200, help="runs per estimate of the quantities available at scale")
    ap.add_argument("--repeats", type=int, default=200); ap.add_argument("--out", default="kl_bounds_check.json")
    a = ap.parse_args(); N = a.N; rng = np.random.default_rng(0); res = dict(args=vars(a), provenance=provenance(), cases=[])
    for cname in a.constraints:
        S = n_symbols(cname); den = make_model(a.model, N, S)
        xs = np.array(np.meshgrid(*[np.arange(S)] * N, indexing="ij")).reshape(N, -1).T
        for prop in a.proposals:
            W = World(den, cname, N, proposal=prop, match=a.match if prop == "chain" else 0)
            code = (xs * W.pw[None]).sum(1); ok = W.alive[code]
            lp = den.log_joint(torch.from_numpy(xs)).numpy(); pi = np.where(ok, np.exp(lp), 0.0); pi = pi / pi.sum()
            pairs = list(itertools.combinations(range(N), 2))
            def lowers(qK):
                one = max(kl(np.bincount(xs[:, i], weights=qK, minlength=S), np.bincount(xs[:, i], weights=pi, minlength=S)) for i in range(N))
                two = max(kl(np.bincount(xs[:, i] * S + xs[:, j], weights=qK, minlength=S * S), np.bincount(xs[:, i] * S + xs[:, j], weights=pi, minlength=S * S)) for i, j in pairs)
                return one, two
            for oname, perm in (("left to right", np.arange(N)), ("random order A", rng.permutation(N)), ("random order B", rng.permutation(N))):
                q = np.ones(len(xs)); st = np.full(len(xs), W.e0)
                for i in perm:
                    q = q * W.q[st, i, xs[:, i]]; st = st + (xs[:, i] - S) * W.pw[i]
                assert abs(q.sum() - 1) < 1e-9
                sup = ok & (q > 0); w = np.where(sup, pi / np.where(sup, q, 1.0), 0.0); varpi = float((pi * w ** 2).sum() - (pi * w).sum() ** 2)
                for K in a.Ks:
                    draws = w[rng.choice(len(xs), size=(a.R, K), p=q)]                                  # R runs of K weights
                    upper = float(-np.log(draws.mean(1)).mean())
                    if K == 1:
                        qK = q.copy()
                    else:
                        Ssum = draws[:, 1:].sum(1); qK = np.zeros(len(xs))
                        for lo in range(0, len(xs), 2000):
                            sl = slice(lo, lo + 2000); qK[sl] = K * q[sl] * w[sl] * (1.0 / (w[sl][:, None] + Ssum[None])).mean(1)
                        qK = qK / qK.sum()
                    qK = np.where(sup, qK, 0.0); qK = qK / qK.sum()                                      # the sampler puts < 1e-12 outside the constraint: drop it
                    exact = kl(qK, np.where(sup, pi, 1.0)); l1, l2 = lowers(qK)
                    ue, me = [], []
                    for _ in range(a.repeats):
                        d = w[rng.choice(len(xs), size=(a.n, K), p=q)]; z = d.mean(1)
                        ue.append(np.log(z.mean()) - np.log(z).mean())
                        f = d.ravel() / d.mean(); m2, m3 = (f ** 2).mean(), (f ** 3).mean(); me.append((m3 - m2 * m2) / (2 * K * K))
                    r = dict(constraint=cname, proposal=prop if prop == "product" else "matched chain", order=oname, K=K, exact=exact, upper=upper, upper_est_mean=float(np.mean(ue)), upper_est_sd=float(np.std(ue)),
                             moment=varpi / (2 * K * K), moment_est_median=float(np.median(me)), moment_est_iqr=[float(np.percentile(me, 25)), float(np.percentile(me, 75))], lower_one=l1, lower_two=l2)
                    res["cases"].append(r)
                    print(f"{cname:9s} {r['proposal']:13s} {oname:14s} K {K:2d}  exact {exact:.5f}  upper {upper:.5f} (est. {r['upper_est_mean']:.5f} ± {r['upper_est_sd']:.5f})  moment {r['moment']:.5f} "
                          f"(est. median {r['moment_est_median']:.5f})  lower 1 / 2 {l1:.5f} / {l2:.5f}", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
