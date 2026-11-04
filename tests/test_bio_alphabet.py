"""Layer 2: codon alphabet, genetic code, codon lifting of nucleotide constraints, synonymous masks."""
import itertools
import numpy as np
import pytest
from twistroute import compile as C
from twistroute_bio import (CODONS, GENETIC_CODE, AMINO_ACIDS, codon_index, nuc_index, translate, synonymous_mask,
                            lift_to_codons)
from conftest import enumerate_law


def test_genetic_code_is_table_1():
    assert len(CODONS) == 64 and len(set(CODONS)) == 64
    counts = {a: sum(v == a for v in GENETIC_CODE.values()) for a in AMINO_ACIDS + ["*"]}
    assert len(AMINO_ACIDS) == 20
    assert counts == {**{a: 2 for a in "FYCHQNKDE"}, **{a: 4 for a in "AGPTV"}, **{a: 6 for a in "LSR"},
                      "I": 3, "*": 3, "M": 1, "W": 1}
    spot = {"ATG": "M", "TGG": "W", "TAA": "*", "TAG": "*", "TGA": "*", "ATA": "I", "AGA": "R", "AGC": "S",
            "TTA": "L", "CTG": "L", "GGG": "G", "TGC": "C", "GAT": "D", "AAA": "K"}
    assert all(GENETIC_CODE[c] == a for c, a in spot.items())
    assert translate(codon_index("ATGGCATGGTAA")) == "MAW*"


def test_synonymous_mask():
    m = synonymous_mask("MW")
    assert m.shape == (3, 64)
    assert m[0].sum() == 1 and m[1].sum() == 1 and m[2].sum() == 3
    assert m[0, codon_index("ATG")[0]] and m[2, codon_index("TAA")[0]]
    with pytest.raises(ValueError):
        synonymous_mask("MXZ")


MOTIFS = {"EcoRI site": ["GAATTC"], "two motifs": ["GGATCC", "AAAA"], "out-of-frame start": ["ATG"]}


@pytest.mark.parametrize("name", list(MOTIFS))
def test_lifted_motif_automaton_matches_nucleotide_check(name, rng):
    """A codon sequence is accepted iff its nucleotide string avoids the motifs, including motifs that
    straddle codon boundaries (exhaustive over 2 codons, random over 6)."""
    pats = [tuple(nuc_index(p)) for p in MOTIFS[name]]
    nuc = C.dfa_forbidden(pats, 4)
    d, cls, q = lift_to_codons(nuc)
    def ok(codons):
        s = "".join(CODONS[c] for c in codons)
        return not any(p in s for p in MOTIFS[name])
    for seq in itertools.product(range(64), repeat=2):
        assert d.accepts(seq) == ok(seq)
    for seq in rng.integers(0, 64, (3000, 6)):
        assert d.accepts(tuple(seq)) == ok(seq)
    assert d.accepts(tuple(rng.integers(0, 64, 0)))                  # empty sequence avoids every motif
    # the class quotient computes the same mass as the codon automaton
    logW = rng.normal(0, 1, (3, 64))
    k = max(cls) + 1
    cw = np.stack([np.logaddexp.reduce(np.where(np.array(cls)[None] == c, logW, -np.inf), axis=1) for c in range(k)], 1)
    assert abs(q.logZ(cw) - d.logZ(logW)) < 1e-9


def test_straddling_motif_is_caught():
    pats = [tuple(nuc_index("GAATTC"))]
    d, _, _ = lift_to_codons(C.dfa_forbidden(pats, 4), minimise_symbols=False)
    assert not d.accepts(tuple(codon_index("AGAATTCAA")))               # A|GAA|TTC|AA: frame offset 1
    assert not d.accepts(tuple(codon_index("GAATTC")))                  # in frame
    assert d.accepts(tuple(codon_index("GAAATTAAA")))       # GAAATT CAAA... no site


def test_masked_lifted_route_against_enumeration(rng):
    """Protein identity AND motif avoidance, as Masked(lifted automaton), equals brute force on codons."""
    pats = [tuple(nuc_index("GCGC"))]
    d, _, _ = lift_to_codons(C.dfa_forbidden(pats, 4), minimise_symbols=False)
    mask = synonymous_mask("AR", stop=False)                            # A: GCN (4), R: CGN + AGA/AGG (6)
    route = C.Masked(d, mask)
    logW = rng.normal(0, 1, (2, 64))
    def w(y):
        s = "".join(CODONS[c] for c in y)
        return float(mask[0, y[0]] and mask[1, y[1]] and "GCGC" not in s)
    law, Z = enumerate_law(logW, w)
    assert law and abs(route.logZ(logW) - np.log(Z)) < 1e-9
    assert all(("GCGC" not in "".join(CODONS[c] for c in route.sample(logW, rng))) for _ in range(200))
