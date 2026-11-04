"""CPU reference routes: logZ and accepts against brute-force enumeration, and the refusal boundary."""
import itertools
import numpy as np
import pytest
from twistroute import compile as C
from conftest import enumerate_law

# every pair (i, j) of the RNA pairing grammar is produced through some Q_xy covering (i+1 .. j), so bounding
# all Q_xy by W bounds every pair span by W + 1
PAIR_SPAN_3 = {n: 3 for n in {b for _, _, b in C.rna_pair_grammar()[1] if b.startswith("Q")}}


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
    "span-bounded ambiguous Dyck-1 chart": (C.Inside(*C.DYCK1_AMBIGUOUS, span_limits={"C": 3}), 8, 2,
                                             C.DYCK1_AMBIGUOUS + ({"C": 3},)),
    "span-bounded RNA pairing chart": (C.Inside(*C.rna_pair_grammar(), span_limits=PAIR_SPAN_3), 8, 4,
                                        C.rna_pair_grammar() + (PAIR_SPAN_3,)),
    "arborescence": (C.Arborescence(4), 4, 5, None),
}


def _and_dfa_cfg():
    d, g = C.dfa_forbidden([(0, 0, 0)], 2), C.DYCK1_AMBIGUOUS
    return C.intersect_dfa_cfg(d, C.Inside(*g)), (lambda y: C.count_parses(*g[:3], y) * d.accepts(y))


def _and_dfa_fg():
    d, fg = C.dfa_fraction_band(0, 1, 3, 3), ring_fg(5, 3)
    return C.DFAFactorProduct(d, fg), (lambda y: float(d.accepts(y) and fg.accepts(y)))


PRODUCTS = {"automaton AND ambiguous grammar (Bar-Hillel)": (_and_dfa_cfg, 8, 2),
            "automaton AND factor graph (augmented chain)": (_and_dfa_fg, 5, 3)}


@pytest.mark.parametrize("name", list(PRODUCTS))
def test_product_routes_match_enumeration(name, rng):
    """logZ, membership and the exact sampler of each heterogeneous product against brute force."""
    from conftest import tv_vs_law, tv_floor
    build, L, A = PRODUCTS[name]
    route, w = build()
    logW = rng.normal(0, 1, (L, A))
    law, Z = enumerate_law(logW, w)
    assert law, "empty intersection: test would be vacuous"
    assert abs(route.logZ(logW) - np.log(Z)) < 1e-9
    for y in itertools.product(range(A), repeat=L):
        assert route.accepts(y) == (w(y) > 0)
    draws = [route.sample(logW, rng) for _ in range(6000)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0 and tv < 2.0 * tv_floor(law, 6000, rng)


def test_product_keeps_span_limits_and_refuses_large_tables():
    d = C.dfa_forbidden([(1, 1, 1, 1)], 2)          # does not exclude the test strings itself
    prod = C.intersect_dfa_cfg(d, C.Inside(*C.DYCK1_AMBIGUOUS, span_limits={"C": 3}))
    deep, flat = (0, 0, 0, 1, 1, 1), (0, 0, 1, 1, 0, 1)
    assert not prod.accepts(deep) and prod.accepts(flat)
    with pytest.raises(C.CompilerRefuse):
        C.DFAFactorProduct(C.dfa_exactly_k(0, 6, 3), ring_fg(12, 3), max_cells=1000)


def weight(route, g):
    if g is not None:
        lim = g[4] if len(g) > 4 else None
        return lambda y: C.count_parses(g[0], g[1], g[2], y, lim)
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
        assert route.accepts(y) == (C.count_parses(g[0], g[1], g[2], y, g[4] if len(g) > 4 else None) > 0)


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


def test_span_limits_restrict_the_language():
    """A span bound removes exactly the derivations with an over-long bounded subtree: for Dyck-1, pairs of
    span <= 4 under the ambiguous grammar's C (= 'S )') leave '(())()' but not '((()))'."""
    full, bnd = C.Inside(*C.DYCK1_AMBIGUOUS), C.Inside(*C.DYCK1_AMBIGUOUS, span_limits={"C": 3})
    deep, flat = (0, 0, 0, 1, 1, 1), (0, 0, 1, 1, 0, 1)
    assert full.accepts(deep) and not bnd.accepts(deep)
    assert full.accepts(flat) and bnd.accepts(flat)
    with pytest.raises(ValueError):
        C.Inside(*C.DYCK1_AMBIGUOUS, span_limits={"nope": 3})
