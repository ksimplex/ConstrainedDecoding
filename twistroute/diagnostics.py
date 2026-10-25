"""
Statistical diagnostics for SMC outputs.

psis_khat           Pareto shape k of the upper tail of importance weights, fitted as in Pareto-smoothed
                    importance sampling (Vehtari, Simpson, Gelman, Yao & Gabry, arXiv:1507.02646) with the
                    Zhang & Stephens (Technometrics 51:316, 2009) estimator. k > 0.5: infinite variance;
                    k > 0.7: the plain mean is unreliable and typically low; k > 1: infinite mean.
ratio_ci            bootstrap CI for mean(Z_hat) / h0 including binomial uncertainty in h0.
subsample_medians   median of the mean over random subsets of n runs: rises with n under heavy tails,
                    flat under a constant bias.
pairwise_tv         TWISTER Sec. 4.3 metric: TV between empirical joint class distributions of every position
                    pair, averaged over pairs (our reading; the paper does not spell out the averaging).
tv_with_floor       pairwise_tv of a decoder sample against a reference, with a matched noise floor.
first_trigger,      hidden-mass decomposition of resampling-off weight traces (notes Sec. 7): the share of
hidden_mass         E[Z_hat] carried by particles whose normalised weight is below eps at the step where adaptive
                    resampling would first fire.
"""
import itertools
import numpy as np
from scipy.special import logsumexp


def gpdfit(x, prior_bs=3, prior_k=10):
    """Generalised Pareto shape for exceedances x > 0 (Zhang & Stephens 2009, PSIS prior)."""
    x = np.sort(np.asarray(x, float)); n = len(x); m = 30 + int(np.sqrt(n))
    bs = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs = bs / (prior_bs * x[int(n / 4 + 0.5) - 1]) + 1 / x[-1]
    ks = np.log1p(-bs[:, None] * x).mean(1)
    L = n * (np.log(-(bs / ks)) - ks - 1)
    w = np.exp(L - logsumexp(L)); w /= w.sum()
    b = np.sum(bs * w)
    k = np.log1p(-b * x).mean()
    return (n * k + prior_k * 0.5) / (n + prior_k)


def psis_khat(logw):
    """Tail shape of exp(logw) from the top M = min(0.2 n, 3 sqrt(n)) weights."""
    w = np.sort(np.exp(np.asarray(logw) - np.max(logw))); n = len(w)
    M = int(min(0.2 * n, 3 * np.sqrt(n)))
    return gpdfit(w[-M:] - w[-M - 1])


def ratio_ci(z, h0, n0, B=4000, rng=None, q=(2.5, 97.5)):
    """Bootstrap percentile CI for mean(z) / h0, with h0 = hits / n0 resampled binomially."""
    rng = rng or np.random.default_rng(0)
    z = np.asarray(z)
    bm = np.array([z[rng.integers(0, len(z), len(z))].mean() for _ in range(B)])
    bh = rng.binomial(n0, h0, B) / n0
    return tuple(np.percentile(bm / bh, q))


def subsample_medians(z, sizes, reps=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    z = np.asarray(z)
    return {n: float(np.median([z[rng.choice(len(z), n, replace=False)].mean() for _ in range(reps)]))
            for n in sizes if n <= len(z)}


def tv_dict(p, q):
    ks = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in ks)


def pairwise_tv(S1, S2, A, pairs=None):
    """Mean over position pairs (i < j) of TV between the empirical joint (A x A) class distributions of two
    samples S1 (n1, N) and S2 (n2, N)."""
    S1, S2 = np.asarray(S1), np.asarray(S2)
    N = S1.shape[1]
    pairs = pairs or list(itertools.combinations(range(N), 2))
    tvs = []
    for i, j in pairs:
        h1 = np.bincount(S1[:, i] * A + S1[:, j], minlength=A * A) / len(S1)
        h2 = np.bincount(S2[:, i] * A + S2[:, j], minlength=A * A) / len(S2)
        tvs.append(0.5 * np.abs(h1 - h2).sum())
    return float(np.mean(tvs))


def tv_with_floor(S, R, A, n=None, n_floor=5, seed0=0):
    """(tv, floor): pairwise TV of the first n rows of sample S against reference R, and the mean pairwise TV
    between a random size-n subset of R and the rest of R over n_floor splits, so both are at the same sample
    size. Default n = min(len(S), len(R) // 2)."""
    S, R = np.asarray(S), np.asarray(R)
    n = min(len(S), len(R) // 2) if n is None else n
    tv = pairwise_tv(S[:n], R, A)
    floor = np.mean([pairwise_tv(R[q[:n]], R[q[n:]], A)
                     for q in (np.random.default_rng(seed0 + s).permutation(len(R)) for s in range(n_floor))])
    return float(tv), float(floor)


def first_trigger(trace, ess_min=0.5):
    """trace (T, R, K): per-step log-weights of resampling-off runs. Returns (s, wn): the first step s < T - 1
    with ESS < ess_min * K for each run (-1 if none), and the normalised weights wn (T, R, K)."""
    T, R, K = trace.shape
    m = trace.max(-1, keepdims=True)
    w = np.nan_to_num(np.exp(trace - np.where(np.isfinite(m), m, 0)))
    wn = w / np.clip(w.sum(-1, keepdims=True), 1e-300, None)
    ess = 1.0 / np.clip((wn ** 2).sum(-1), 1e-300, None)
    trig = (ess < ess_min * K)[: T - 1]
    return np.where(trig.any(0), trig.argmax(0), -1), wn


def hidden_mass(trace, h0, eps_grid=(1e-6, 1e-4, 1e-2, 0.1), ess_min=0.5):
    """Share of E[Z_hat] / h0 carried by particles with normalised weight < eps at the first trigger step.

    Under multinomial resampling such a particle survives with probability <= K * eps, so with R runs the
    mass below eps ~ 1 / (K R) is rarely realised (notes, hidden-mass proposition). Returns a dict with
    'Z_over_h0' (resampling-off estimate), 'frac_runs_triggered' and 'mass_below' {eps: share}."""
    T, R, K = trace.shape
    s, wn = first_trigger(trace, ess_min)
    trig = s >= 0
    wn_s = np.where(trig[:, None], wn[np.maximum(s, 0), np.arange(R)], np.nan)
    contrib = np.exp(trace[-1]) / K / (R * h0)
    return dict(Z_over_h0=float(contrib.sum()), frac_runs_triggered=float(trig.mean()),
                mass_below={e: float(contrib[trig[:, None] & (wn_s < e)].sum()) for e in eps_grid})
