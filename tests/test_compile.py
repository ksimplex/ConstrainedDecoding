"""CPU reference routes: logZ and accepts against brute-force enumeration, and the refusal boundary."""
import itertools
import numpy as np
import pytest
from twistroute import compile as C
from conftest import enumerate_law


def ring_fg(L, A):
    neq = 1 - np.eye(A)
    return C.FactorGraph(L, A, [((i, (i + 1) % L), neq) for i in range(L)], name="ring")


ROUTES = {   # name: (route, L, A, derivation weight d(y))
    "forbidden motifs": (C.dfa_forbidden([(1, 1), (0, 0, 0)], 2), 7, 2, None),
    "contains motif": (C.dfa_contains([(0, 1, 0)], 2), 7, 2, None),
    "exactly k": (C.dfa_exactly_k(0, 3, 3), 6, 3, None),
    "fraction band": (C.dfa_fraction_band(1, 2, 4, 2), 7, 2, None),
    "prefix a.*": (C.dfa_prefix_class(0, 3), 5, 3, None),
    "window all-different w=2": (C.dfa_window_alldiff(2, 4), 6, 4, None),
    "Dyck-2 depth-bounded automaton": (C.dfa_dyck2_depth(3), 6, 4, None),
    "copy automaton": (C.dfa_copy_positional(3, 2), 6, 2, None),
    "conjunction (product)": (C.product_dfa(C.dfa_forbidden([(1, 1)], 2), C.dfa_fraction_band(0, 3, 4, 2), 2), 7, 2, None),
    "copy factor graph": (C.CopyFG(3, 2), 6, 2, None),
    "ring 3-colouring factor graph": (ring_fg(5, 3), 5, 3, None),
    "Dyck-2 chart": (C.Inside(*C.DYCK2), 6, 4, C.DYCK2),
    "ambiguous Dyck-1 chart": (C.Inside(*C.DYCK1_AMBIGUOUS), 8, 2, C.DYCK1_AMBIGUOUS),
    "RNA pairing chart": (C.Inside(*C.rna_pair_grammar()), 6, 4, C.rna_pair_grammar()),
    "arborescence": (C.Arborescence(4), 4, 5, None),
}


def weight(route, g):
    if g is not None:
        return lambda y: C.count_parses(g[0], g[1], g[2], y)
    return lambda y: float(route.accepts(tuple(y)))


@pytest.mark.parametrize("name", list(ROUTES))
def test_logZ_matches_enumeration(name, rng):
    route, L, A, g = ROUTES[name]
    logW = rng.normal(0, 1, (L, A))
    law, Z = enumerate_law(logW, weight(route, g))
    assert law, f"{name}: empty language, test would be vacuous"
    assert abs(route.logZ(logW) - np.log(Z)) < 1e-9


@pytest.mark.parametrize("name", [n for n in ROUTES if ROUTES[n][3] is not None])
def test_chart_accepts_iff_parses(name):
    route, L, A, g = ROUTES[name]
    for y in itertools.product(range(A), repeat=L):
        assert route.accepts(y) == (C.count_parses(g[0], g[1], g[2], y) > 0)


def test_ambiguous_and_unambiguous_dyck1_same_language():
    a, u = C.Inside(*C.DYCK1_AMBIGUOUS), C.Inside(*C.DYCK1_UNAMBIGUOUS)
    for y in itertools.product(range(2), repeat=8):
        assert a.accepts(y) == u.accepts(y)
        if u.accepts(y):
            assert C.count_parses(*C.DYCK1_UNAMBIGUOUS[:3], y) == 1


def test_max_semiring_is_constrained_map(rng):
    g = C.DYCK2; route = C.Inside(*g)
    logW = rng.normal(0, 1, (6, 4))
    best = max(sum(logW[i, y[i]] for i in range(6)) for y in itertools.product(range(4), repeat=6) if route.accepts(y))
    assert abs(route.logZ(logW, "max") - best) < 1e-9


def test_compiler_refuses_past_budget():
    with pytest.raises(C.CompilerRefuse):
        C.dfa_dyck2_depth(12, max_states=1000)


def test_copy_automaton_size_is_k_plus_1_times_2_to_k():
    for k in range(2, 8):
        assert C.dfa_copy_positional(k, 2).n_states == (k + 1) * 2 ** k
