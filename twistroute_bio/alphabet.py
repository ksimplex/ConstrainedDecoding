"""
Nucleotide and codon alphabets, the standard genetic code, and the codon -> nucleotide homomorphism.

Index conventions (model-independent; a model adapter maps its own token ids onto these):
    NUCLEOTIDES = "ACGT"                      U is read as T
    CODONS      = all 64 triples in lexicographic ACGT order: AAA, AAC, ..., TTT
The genetic code is NCBI translation table 1 (standard code). Stop codons translate to "*".
"""
import itertools
import numpy as np
from twistroute.compile import lift_dfa, symbol_classes

NUCLEOTIDES = "ACGT"
CODONS = ["".join(c) for c in itertools.product(NUCLEOTIDES, repeat=3)]

# NCBI table 1, written in the conventional TCAG order of the first, second and third base
_TCAG = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
GENETIC_CODE = {a + b + c: _AAS[16 * i + 4 * j + k]
                for i, a in enumerate(_TCAG) for j, b in enumerate(_TCAG) for k, c in enumerate(_TCAG)}
AMINO_ACIDS = sorted(set(GENETIC_CODE.values()) - {"*"})


def nuc_index(seq):
    """Nucleotide string -> list of indices into NUCLEOTIDES (U read as T)."""
    return [NUCLEOTIDES.index(ch) for ch in seq.upper().replace("U", "T")]


def codon_index(seq):
    """Coding sequence (length a multiple of 3) -> list of indices into CODONS."""
    s = seq.upper().replace("U", "T")
    if len(s) % 3:
        raise ValueError("coding sequence length must be a multiple of 3")
    return [CODONS.index(s[i:i + 3]) for i in range(0, len(s), 3)]


def codon_words():
    """The homomorphism h for lift_dfa: codon index -> its three nucleotide indices."""
    return [tuple(nuc_index(c)) for c in CODONS]


def amino_acid_of():
    """Per codon index, its amino acid letter ('*' for stop)."""
    return [GENETIC_CODE[c] for c in CODONS]


def translate(codons):
    """Codon indices -> protein string."""
    aa = amino_acid_of()
    return "".join(aa[c] for c in codons)


def synonymous_mask(protein, stop=True):
    """(L, 64) bool: at each position, the codons that encode that residue. With stop=True a final stop codon
    position is appended (any of TAA, TAG, TGA); stop codons are never allowed inside the reading frame."""
    aa = np.array(amino_acid_of())
    rows = [aa == r for r in protein.upper()]
    if any(not r.any() for r in rows):
        raise ValueError("protein contains a letter that is not an amino acid")
    if stop:
        rows.append(aa == "*")
    return np.stack(rows)


def lift_to_codons(nuc_dfa, minimise_symbols=True):
    """Lift a nucleotide automaton to codon tokens. With minimise_symbols, also return the codon classes
    (codons with the same transitions and the same amino acid share a class) and the quotient automaton over
    them; the amino-acid refinement keeps synonymous_mask exact at class level.
    Returns (codon_dfa, cls, class_dfa) or (codon_dfa, None, None)."""
    d = lift_dfa(nuc_dfa, codon_words(), name=f"codon({nuc_dfa.name})")
    if not minimise_symbols:
        return d, None, None
    cls, q = symbol_classes(d, refine=amino_acid_of())
    return d, cls, q
