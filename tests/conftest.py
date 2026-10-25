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


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
