"""Layer 2: CDS constraint builders against independent nucleotide-level checks and brute-force enumeration."""
import itertools
import numpy as np
import pytest
import torch
from twistroute import compile as C
from twistroute_bio import CODONS, codon_index, synonymous_mask
from twistroute_bio.constraints import (Motifs, GCBand, TargetStructure, compile_cds, expand_iupac,
                                        parse_dot_bracket)
from conftest import enumerate_law, tv_vs_law, tv_threshold


def test_iupac_and_dot_bracket():
    assert sorted(expand_iupac("GTRAGT")) == ["GTAAGT", "GTGAGT"]
    assert len(expand_iupac("NNN")) == 64
    with pytest.raises(C.CompilerRefuse):
        expand_iupac("N" * 7)
    assert parse_dot_bracket("((..))..(.)") == [(0, 5), (1, 4), (8, 10)]
    with pytest.raises(ValueError):
        parse_dot_bracket("(()")


def _all_valid(protein):
    m = synonymous_mask(protein)
    return [tuple(c) for c in itertools.product(*[np.flatnonzero(r) for r in m])]


CASES = {
    "forbidden EcoRI + homopolymer": ("LRS", [Motifs(forbidden=("GAATTC", "AAAA"))]),
    "forbidden IUPAC splice-like": ("SR", [Motifs(forbidden=("GTRAGT",))]),
    "required motif": ("LS", [Motifs(required=("CTGAG",))]),
    "GC band": ("LRG", [GCBand(5, 7)]),
    "target structure": ("LRS", [TargetStructure.from_dot_bracket("((((....))))")]),
    "motifs AND GC AND structure": ("LRS", [Motifs(forbidden=("CGCG",)), GCBand(4, 9),
                                            TargetStructure.from_dot_bracket("(((......)))")]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_compiled_route_matches_independent_check(name, rng):
    protein, cons = CASES[name]
    r = compile_cds(protein, cons)
    valid = _all_valid(protein)                                # every synonymous coding of protein + stop
    acc = [r.route.accepts(y) for y in valid]
    chk = [r.check(y) for y in valid]
    assert acc == chk and any(chk), f"{name}: route and nucleotide check disagree or language empty"
    # off-protein sequences are rejected
    for y in rng.integers(0, 64, (300, r.n_codons)):
        assert r.route.accepts(tuple(y)) == r.check(tuple(y))


@pytest.mark.parametrize("name", ["forbidden EcoRI + homopolymer", "target structure", "motifs AND GC AND structure"])
def test_compiled_route_law_and_logZ(name, rng):
    """Exact sampler and logZ against enumeration over the synonymous codings (the only support)."""
    protein, cons = CASES[name]
    r = compile_cds(protein, cons)
    logW = rng.normal(0, 1, (r.n_codons, 64))
    valid = _all_valid(protein)
    p = {y: np.exp(sum(logW[i, c] for i, c in enumerate(y))) for y in valid if r.check(y)}
    Z = sum(p.values()); law = {k: v / Z for k, v in p.items()}
    assert abs(r.route.logZ(logW) - np.log(Z)) < 1e-9
    draws = [r.route.sample(logW, rng) for _ in range(4000)]
    tv, invalid = tv_vs_law(draws, law)
    assert invalid == 0 and tv < tv_threshold(law, 4000, rng)


def test_structure_factors_merge_pairs_between_the_same_codons():
    ts = TargetStructure(((0, 5), (1, 4), (2, 3)))             # all within codons 0 and 1
    fs = ts.factors(2)
    assert [k for k, _ in fs] == [(0, 1)]
    t = fs[0][1]
    a, b = codon_index("GCA")[0], codon_index("TGC")[0]         # GCA|TGC: G-C, C-G, A-T
    assert t[a, b] == 1.0 and t[a, codon_index("TGA")[0]] == 0.0


def test_batched_routes_match_cpu(rng):
    from twistroute.torch_routes import torch_route, TorchMasked
    protein, cons = CASES["motifs AND GC AND structure"]
    r = compile_cds(protein, cons)
    inner = r.route.route
    g = TorchMasked(torch_route(inner), r.route.allowed)
    logW = rng.normal(0, 1, (r.n_codons, 64))
    t = torch.tensor(logW, dtype=torch.float32)[None]
    assert abs(float(g.logZ(t)[0]) - r.route.logZ(logW)) < 1e-4
    ys = g.sample(t.expand(500, r.n_codons, 64).contiguous(), torch.Generator().manual_seed(0))
    assert all(r.check(tuple(y)) for y in ys.tolist())
