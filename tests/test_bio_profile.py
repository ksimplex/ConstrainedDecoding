"""Positional and distributional sequence metrics.

Each metric is checked against a property its definition forces, not against a remembered number: %MinMax
must hit +/-100 at the extreme synonymous encodings, CAI must hit 1 at the max-frequency encoding, and both
must be invariant to things their definitions say they ignore.
"""
import math
import random

import pytest

from twistroute_bio.alphabet import CODONS, GENETIC_CODE
from twistroute_bio.score import profile as prof


def _corpus(seed=0, n=40, length=90):
    """Random coding sequences over a fixed protein alphabet, as a frequency-table corpus."""
    rng = random.Random(seed)
    aa = [a for a in prof._SYN if a != "*"]
    out = []
    for _ in range(n):
        prot = [rng.choice(aa) for _ in range(length)]
        out.append("".join(rng.choice(prof._SYN[a]) for a in prot))
    return out


@pytest.fixture(scope="module")
def table():
    return prof.codon_table(_corpus())


def _reencode(seq, pick):
    return "".join(pick(prof._SYN[GENETIC_CODE[c]]) for c in prof._codons(seq))


def test_table_sums_and_positivity(table):
    assert abs(sum(table["abs"].values()) - 1000.0) < 1e-6
    for aa, fam in prof._SYN.items():
        assert abs(sum(table["rel"][c] for c in fam) - 1.0) < 1e-9, aa
    assert all(v > 0 for v in table["rel"].values())          # pseudocount keeps every codon drawable


def test_minmax_hits_its_bounds(table):
    """The extreme synonymous encodings are what +/-100 are defined to mean."""
    seq = _corpus(seed=7, n=1, length=60)[0]
    rel = table["rel"]
    hi = _reencode(seq, lambda f: max(f, key=lambda c: rel[c]))
    lo = _reencode(seq, lambda f: min(f, key=lambda c: rel[c]))
    for s, want in [(hi, 100.0), (lo, -100.0)]:
        vals = [v for _, v in prof.minmax_profile(s, table)]
        assert vals and all(abs(v - want) < 1e-6 for v in vals), (want, vals[:3])


def test_minmax_is_bounded_and_windowed(table):
    seq = _corpus(seed=3, n=1, length=60)[0]
    p = prof.minmax_profile(seq, table, window=18)
    assert len(p) == 60 - 18 + 1
    assert [i for i, _ in p] == list(range(9, 9 + len(p)))     # reported at the window midpoint
    assert all(-100.0 - 1e-9 <= v <= 100.0 + 1e-9 for _, v in p)


def test_cai_is_one_at_the_max_frequency_encoding(table):
    seq = _corpus(seed=5, n=1, length=60)[0]
    rel = table["rel"]
    hi = _reencode(seq, lambda f: max(f, key=lambda c: rel[c]))
    assert abs(prof.cai(hi, table) - 1.0) < 1e-12
    assert prof.cai(seq, table) <= 1.0 + 1e-12


def test_cai_excludes_single_codon_families(table):
    """Met and Trp have w = 1 by definition and carry no information; including them would dilute CAI."""
    base = "ATG" * 5                                            # all Met
    with pytest.raises(ValueError):
        prof.cai(base, table)                                   # nothing scorable
    mixed = base + "AAA" * 5
    assert prof.cai(mixed, table) == pytest.approx(prof.cai("AAA" * 5, table))


def test_profile_distance_is_zero_on_itself_and_positive_otherwise(table):
    seq = _corpus(seed=11, n=1, length=60)[0]
    rel = table["rel"]
    a = prof.minmax_profile(seq, table)
    b = prof.minmax_profile(_reencode(seq, lambda f: max(f, key=lambda c: rel[c])), table)
    assert prof.profile_distance(a, a) == {"aligned": 0.0, "dtw": 0.0}
    d = prof.profile_distance(a, b)
    assert d["aligned"] > 0 and d["dtw"] >= 0


def test_gc3_sees_only_third_positions():
    """Synonymous change at the wobble base must move gc3; the protein and gc of first two bases do not."""
    assert prof.gc3("GCA") == 0.0 and prof.gc3("GCG") == 1.0    # Ala: both encode A
    assert prof.gc3("GCAGCG") == 0.5


def test_metrics_reject_nothing_a_real_sequence_contains(table):
    """U is read as T, and a trailing partial codon is ignored rather than crashing."""
    assert prof.gc3("GCU") == prof.gc3("GCT")
    assert len(prof._codons("ATGAAAT")) == 2
