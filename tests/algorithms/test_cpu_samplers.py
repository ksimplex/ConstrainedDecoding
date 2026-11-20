"""Exact samplers of every CPU route: empirical law vs enumeration, within 2x the multinomial TV floor."""
import numpy as np
import pytest
from twistroute import compile as C
from conftest import enumerate_law, tv_vs_law, tv_floor
from test_compile import ROUTES, weight

N_DRAWS = 6000


@pytest.mark.parametrize("name", list(ROUTES))
def test_sampler_matches_law(name, rng):
    route, L, A, g = ROUTES[name]
    logW = rng.normal(0, 1, (L, A))
    law, _ = enumerate_law(logW, weight(route, g))
    draws = [route.sample(logW, rng) for _ in range(N_DRAWS)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0
    assert tv < 2.0 * tv_floor(law, N_DRAWS, rng), f"{name}: TV {tv:.4f}"


def test_wilson_fallback_with_weak_root_edges(rng):
    route = C.Arborescence(4)
    logW = rng.normal(0, 1, (4, 5)); logW[:, 0] -= 6.0           # long random walks before hitting the root
    law, _ = enumerate_law(logW, lambda y: float(route.accepts(y)))
    draws = [route.sample(logW, rng) for _ in range(N_DRAWS)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0 and tv < 2.0 * tv_floor(law, N_DRAWS, rng)
