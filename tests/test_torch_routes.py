"""Batched torch routes against the CPU compiler and enumeration: logZ (with revealed positions), membership,
and the law of one batched sampling call."""
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.torch_routes import TorchChain, TorchChart, TorchCopy, det_logZ
from conftest import enumerate_law, tv_vs_law, tv_floor


class PairCopy:   # CPU reference for a partial copy (pairs (i, j), other positions free)
    def __init__(self, pairs, L): self.pairs, self.L = pairs, L
    def accepts(self, y): return all(y[i] == y[j] for i, j in self.pairs)


CASES = {
    "chain: forbidden": (lambda: C.dfa_forbidden([(3, 3), (1, 1, 1)], 4), TorchChain, 7, 4),
    "chain: exactly k": (lambda: C.dfa_exactly_k(0, 3, 4), TorchChain, 7, 4),
    "chain: prefix a.*": (lambda: C.dfa_prefix_class(2, 5), TorchChain, 5, 5),
    "chart: Dyck-1 unambiguous": (lambda: C.Inside(*C.DYCK1_UNAMBIGUOUS), TorchChart, 10, 2),
    "chart: Dyck-1 ambiguous": (lambda: C.Inside(*C.DYCK1_AMBIGUOUS), TorchChart, 10, 2),
    "chart: Dyck-2": (lambda: C.Inside(*C.DYCK2), TorchChart, 6, 4),
    "chart: RNA pairing": (lambda: C.Inside(*C.rna_pair_grammar()), TorchChart, 6, 4),
}


def _weight(cpu):
    if isinstance(cpu, C.Inside):
        return lambda y: C.count_parses(cpu.start, cpu.binary, cpu.lexical, y)
    return lambda y: float(cpu.accepts(tuple(y)))


def _logW_with_reveals(rng, L, A, law_fn, reveal=2):
    for _ in range(50):
        lw = rng.normal(0, 1, (L, A))
        for i in rng.choice(L, reveal, replace=False):
            lw[i] = -np.inf; lw[i, rng.integers(A)] = 0.0
        law, Z = enumerate_law(lw, law_fn)
        if law:
            return lw, law, Z
    raise RuntimeError("no satisfiable reveal pattern")


def _check(gpu, cpu_accepts, law_fn, L, A, rng, n=20000):
    lw, law, Z = _logW_with_reveals(rng, L, A, law_fn)
    t = torch.tensor(lw, dtype=torch.float32)[None]
    assert abs(float(gpu.logZ(t)[0]) - np.log(Z)) < 1e-4
    ys = gpu.sample(t.expand(n, L, A).contiguous(), torch.Generator().manual_seed(1))
    tv, invalid = tv_vs_law(list(map(tuple, ys.tolist())), law)
    assert invalid == 0 and tv < 2.0 * tv_floor(law, n, rng)
    Y = torch.tensor(rng.integers(0, A, (1500, L)))
    assert (gpu.accepts(Y).numpy() == np.array([bool(cpu_accepts(tuple(y))) for y in Y.tolist()])).all()


@pytest.mark.parametrize("name", list(CASES))
def test_route(name, rng):
    build, cls, L, A = CASES[name]
    cpu = build()
    _check(cls(cpu), cpu.accepts, _weight(cpu), L, A, rng)


def test_copy_partial(rng):
    pairs = [(0, 4), (1, 5), (2, 6)]
    ref = PairCopy(pairs, 8)
    _check(TorchCopy(pairs, 8, 3), ref.accepts, lambda y: float(ref.accepts(y)), 8, 3, rng)


def test_det_logZ_matches_cpu(rng):
    tree = C.Arborescence(5)
    lw = rng.normal(0, 1, (3, 5, 6))
    got = det_logZ(torch.tensor(lw)).numpy()
    assert np.allclose(got, [tree.logZ(w) for w in lw], atol=1e-8)
