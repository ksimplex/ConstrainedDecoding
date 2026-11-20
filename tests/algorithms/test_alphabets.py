"""Alphabet changes (inverse homomorphism, minimisation, symbol classes) and per-position restrictions,
against brute force on abstract alphabets."""
import itertools
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.torch_routes import TorchChain, TorchMasked
from conftest import enumerate_law, tv_vs_law, tv_threshold

# new alphabet of 6 symbols over base {0, 1, 2}: words of length 1, 2 and 3, mixed (like single tokens and triples)
H = [(0,), (1,), (2, 2), (0, 1, 2), (1, 1, 0), (2, 0, 2)]
BASES = {
    "forbidden motifs": C.dfa_forbidden([(1, 2), (0, 0, 0)], 3),
    "contains motif": C.dfa_contains([(2, 1, 0)], 3),
    "count band": C.dfa_fraction_band(2, 2, 4, 3),
}


def flat(seq):
    return tuple(y for a in seq for y in H[a])


@pytest.mark.parametrize("name", list(BASES))
def test_lift_accepts_iff_base_accepts_the_image(name):
    base = BASES[name]
    lifted = C.lift_dfa(base, H)
    for n in range(0, 4):
        for seq in itertools.product(range(len(H)), repeat=n):
            assert lifted.accepts(seq) == base.accepts(flat(seq)), (seq, flat(seq))


@pytest.mark.parametrize("name", list(BASES))
def test_minimised_and_class_quotient_preserve_logZ(name, rng):
    lifted = C.lift_dfa(BASES[name], H)
    mini = C.minimize_dfa(lifted)
    assert mini.n <= lifted.n
    cls, quo = C.symbol_classes(mini)
    L = 4
    logW = rng.normal(0, 1, (L, len(H)))
    law, Z = enumerate_law(logW, lambda y: float(lifted.accepts(y)))
    assert abs(lifted.logZ(logW) - np.log(Z)) < 1e-9
    assert abs(mini.logZ(logW) - np.log(Z)) < 1e-9
    k = max(cls) + 1                                            # class weights = sums of member weights
    cw = np.stack([np.logaddexp.reduce(np.where(np.array(cls)[None] == c, logW, -np.inf), axis=1) for c in range(k)], 1)
    assert abs(quo.logZ(cw) - np.log(Z)) < 1e-9


def test_symbol_classes_merge_and_refine():
    """Under 'count symbol 2' only the number of 2s matters: words with equal counts merge, unless refined."""
    base = C.dfa_fraction_band(2, 0, 6, 3)
    lifted = C.lift_dfa(base, H)
    cls, _ = C.symbol_classes(lifted)
    n2 = [w.count(2) for w in H]
    assert all((cls[a] == cls[b]) == (n2[a] == n2[b]) for a in range(len(H)) for b in range(len(H)))
    cls_r, _ = C.symbol_classes(lifted, refine=[len(w) for w in H])
    assert max(cls_r) + 1 > max(cls) + 1


def test_masked_route_cpu_and_batched(rng):
    base = C.dfa_forbidden([(1, 1)], 3)
    L = 6
    allowed = rng.random((L, 3)) < 0.7
    allowed[:, 0] = True                                        # keep the language non-empty
    m = C.Masked(base, allowed)
    logW = rng.normal(0, 1, (L, 3))
    law, Z = enumerate_law(logW, lambda y: float(base.accepts(y) and all(allowed[i, v] for i, v in enumerate(y))))
    assert abs(m.logZ(logW) - np.log(Z)) < 1e-9
    draws = [m.sample(logW, rng) for _ in range(6000)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0 and tv < tv_threshold(law, 6000, rng)
    g = TorchMasked(TorchChain(base), allowed)
    t = torch.tensor(logW, dtype=torch.float32)[None]
    assert abs(float(g.logZ(t)[0]) - np.log(Z)) < 1e-4
    ys = g.sample(t.expand(20000, L, 3).contiguous(), torch.Generator().manual_seed(0))
    tv, invalid = tv_vs_law(list(map(tuple, ys.tolist())), law)
    assert invalid == 0 and tv < tv_threshold(law, 20000, rng)
    Y = torch.tensor(list(itertools.product(range(3), repeat=L)))
    ref = np.array([m.accepts(tuple(y)) for y in Y.tolist()])
    assert (g.accepts(Y).numpy() == ref).all()
