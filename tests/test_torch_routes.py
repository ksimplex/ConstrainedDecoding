"""Batched torch routes against the CPU compiler and enumeration: logZ (with revealed positions), membership,
and the law of one batched sampling call."""
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute.torch_routes import TorchChain, TorchChart, TorchCopy, det_logZ
from conftest import enumerate_law, tv_vs_law, tv_floor

# every pair (i, j) of the RNA pairing grammar is produced through some Q_xy covering (i+1 .. j), so bounding
# all Q_xy by W bounds every pair span by W + 1
PAIR_SPAN_3 = {n: 3 for n in {b for _, _, b in C.rna_pair_grammar()[1] if b.startswith("Q")}}


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
    "chart: automaton AND grammar (Bar-Hillel)": (lambda: C.intersect_dfa_cfg(C.dfa_forbidden([(0, 0, 0)], 2),
                                                                         C.Inside(*C.DYCK1_AMBIGUOUS)),
                                                   TorchChart, 10, 2),
    "chart: span-bounded Dyck-1 ambiguous": (lambda: C.Inside(*C.DYCK1_AMBIGUOUS, span_limits={"C": 3}),
                                             TorchChart, 10, 2),
    "chart: span-bounded RNA pairing": (lambda: C.Inside(*C.rna_pair_grammar(), span_limits=PAIR_SPAN_3),
                                        TorchChart, 8, 4),
}


def _weight(cpu):
    if isinstance(cpu, C.Inside):
        return lambda y: C.count_parses(cpu.start, cpu.binary, cpu.lexical, y, cpu.span_limits)
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


def test_span_limits_skip_work_and_match_unbounded_when_loose(rng):
    """A limit at least L changes nothing; a tight limit evaluates fewer split points than the full chart."""
    g = C.rna_pair_grammar()
    lw = torch.tensor(rng.normal(0, 1, (4, 9, 4)), dtype=torch.float32)
    a = TorchChart(C.Inside(*g)).logZ(lw)
    b = TorchChart(C.Inside(*g, span_limits={k: 9 for k in PAIR_SPAN_3})).logZ(lw)
    assert torch.allclose(a, b)
    tight = TorchChart(C.Inside(*g, span_limits={k: 2 for k in PAIR_SPAN_3}))
    ids, _, lo, hi = tight.g.plan(8)
    assert ids is not None and hi - lo + 1 < 7          # fewer than the span - 1 = 7 unrestricted splits
    names = tight.g.names
    assert tight.g.limit[names.index("P")] == 3 and tight.g.limit[names.index("S")] == float("inf")


def test_build_route_conjunctions():
    """Spec-level conjunctions compile to one route of the right kind and accept exactly the intersection."""
    from twistroute.torch_routes import build_route
    import itertools
    fb = dict(kind="forbidden", A=2, N=8, patterns=[[0, 0, 0]])
    ek = dict(kind="exactly_k", A=2, N=8, cls=0, count=4)
    cfg = dict(kind="cfg", A=2, N=8, grammar="dyck1_amb", span_limits={"C": 5})
    r1 = build_route(dict(kind="and", A=2, N=8, parts=[fb, ek]))
    r2 = build_route(dict(kind="and", A=2, N=8, parts=[fb, cfg]))
    assert isinstance(r1, TorchChain) and isinstance(r2, TorchChart)
    parts = [build_route(fb), build_route(cfg)]
    Y = torch.tensor(list(itertools.product(range(2), repeat=8)))
    assert (r2.accepts(Y) == (parts[0].accepts(Y) & parts[1].accepts(Y))).all()
    with pytest.raises(C.CompilerRefuse):
        build_route(dict(kind="and", A=2, N=8, parts=[cfg, cfg]))
