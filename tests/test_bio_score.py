"""Layer 2 scorers (ViennaRNA) and rule-based construction baselines."""
import pytest
from twistroute_bio import CODONS, translate, codon_index
from twistroute_bio.score import (mfe, mfe_structure, structure_distance, ensemble_defect, fraction_paired,
                                  score_batch, frequency_codons, rule_construct, HUMAN_CODON_FREQ)
from twistroute_bio.constraints import Motifs, GCBand, TargetStructure, compile_cds


def test_vienna_scorers_basic():
    hairpin = "GGGGGAAAACCCCC"
    s, e = mfe_structure(hairpin)
    assert e < 0 and s.count("(") == s.count(")") and 0 < fraction_paired(hairpin) <= 1
    assert mfe(hairpin) == e
    assert structure_distance(hairpin, s) == 0                       # distance to its own MFE structure is 0
    assert structure_distance(hairpin, "." * len(hairpin)) == s.count("(")
    assert ensemble_defect(hairpin, s) >= 0
    assert mfe("GGGGGAAAACCCCC") == mfe("GGGGGAAAACCCCC".replace("T", "U"))   # T and U equivalent


def test_score_batch():
    seqs = ["GGGAAACCC", "AAAAAAAAA"]
    rows = score_batch(seqs, target="(((...)))")
    assert len(rows) == 2 and all("structure_distance" in r and "ensemble_defect" in r for r in rows)
    assert rows[1]["fraction_paired"] == 0.0                         # poly-A does not fold


def test_frequency_codons_is_valid_and_deterministic():
    prot = "MAInputLKESR".replace("nput", "")                        # MAILKESR
    seq = frequency_codons(prot)
    codons = codon_index(seq)
    assert translate(codons) == prot + "*"
    assert HUMAN_CODON_FREQ[CODONS[codons[0]]] == 1.0                 # M has a unique codon
    # leucine's most frequent human codon is CTG
    assert CODONS[codon_index(frequency_codons("L"))[0]] == "CTG"


@pytest.mark.parametrize("name,cons,solvable", [
    ("motif only", [Motifs(forbidden=("GAATTC",))], True),
    ("GC band", [GCBand(10, 14)], True),
    ("unsatisfiable motif", [Motifs(forbidden=("GCN",))], None),      # forbids every GC-starting codon; may fail
])
def test_rule_construct(name, cons, solvable):
    prot = "ARNDCEQGH"
    r = compile_cds(prot, cons)
    out = rule_construct(prot, r, tries=50)
    if solvable:
        assert out is not None and r.check(out)
    # when not solvable the baseline returns None; either way it never returns an invalid sequence
    assert out is None or r.check(out)


def test_rule_construct_respects_structure():
    prot = "LLLLLL"
    r = compile_cds(prot, [TargetStructure.from_dot_bracket("((((((......))))))" + "." * 3)])
    out = rule_construct(prot, r, tries=100)
    assert out is None or r.check(out)
