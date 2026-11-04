"""Shared helpers. Tests are statistical where sampling is involved: thresholds are set at 3-4 standard
errors (or 2x the multinomial TV floor) with fixed seeds, so they are deterministic and fail only on real
regressions."""
import itertools
import numpy as np
import pytest


def enumerate_law(logW, weight):
    """Exact target law over A^L: p(y) ∝ weight(y) * prod_i exp(logW[i, y_i]). Returns (law dict, Z)."""
    L, A = logW.shape
    p = {}
    for y in itertools.product(range(A), repeat=L):
        d = weight(y)
        if d:
            v = d * np.exp(sum(logW[i, y[i]] for i in range(L)))
            if v > 0:
                p[y] = v
    Z = sum(p.values())
    return {k: v / Z for k, v in p.items()}, Z


def tv_vs_law(draws, law):
    n = len(draws); cnt = {}
    for y in draws:
        cnt[y] = cnt.get(y, 0) + 1
    tv = 0.5 * sum(abs(law.get(k, 0) - cnt.get(k, 0) / n) for k in set(law) | set(cnt))
    invalid = sum(v for k, v in cnt.items() if k not in law)
    return tv, invalid


def tv_floor(law, n, rng, reps=20):
    pr = np.array(list(law.values()))
    return float(np.mean([0.5 * np.abs(rng.multinomial(n, pr) / n - pr).sum() for _ in range(reps)]))


def tv_threshold(law, n, rng, level=1e-3, reps=4000):
    """Level-`level` acceptance threshold for the TV of n exact draws: the (1 - level) quantile of the TV of
    n multinomial draws from the law itself, and never below 2x its mean. Laws with few atoms have a widely
    spread TV, so a fixed multiple of the mean is not a fixed significance level."""
    pr = np.array(list(law.values()))
    tvs = 0.5 * np.abs(rng.multinomial(n, pr, size=reps) / n - pr).sum(-1)
    return float(max(np.quantile(tvs, 1 - level), 2.0 * tvs.mean()))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
