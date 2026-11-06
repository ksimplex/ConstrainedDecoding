"""
Coding-sequence constraint builders. Each builder states a constraint at nucleotide level; compile_cds turns a
set of them into one route over codon tokens (indices into twistroute_bio.CODONS), with the protein fixed.

    Motifs           forbidden / required nucleotide motifs, anywhere in the sequence and in any reading frame,
                     including across codon boundaries. Degenerate positions use the IUPAC nucleotide codes
                     (Cornish-Bowden, Nucleic Acids Res. 13(9):3021-3030, 1985, doi:10.1093/nar/13.9.3021).
    GCBand           total G+C count within [lo, hi]
    TargetStructure  base pairs (dot-bracket or a pair list) whose nucleotides must be complementary:
                     Watson-Crick, optionally plus the G-U wobble pair (Crick, J. Mol. Biol. 19(2):548-555,
                     1966, doi:10.1016/s0022-2836(66)80022-0), which is a real RNA pairing and so is on by
                     default.

Automaton constraints are combined by the product construction and lifted to codons by inverse homomorphism;
this is why a motif straddling a codon boundary needs no special case. A target structure becomes a pairwise
factor graph over the codons that contain the paired nucleotides, with a unary factor when both ends of a pair
fall in the same codon.

Protein identity is always imposed through the route, as per-position synonymous-codon masks, never by masking
the model's predictions. Masking predictions step by step is the step-local projection that biases the
trajectory law (notes, Sec. 1); a mask applied inside the route is a constraint like any other and the twisted
SMC weights account for it exactly.
"""
from dataclasses import dataclass, field
import itertools
import numpy as np
from twistroute import compile as C
from ..alphabet import NUCLEOTIDES, CODONS, nuc_index, synonymous_mask, codon_words

IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "U": "T", "R": "AG", "Y": "CT", "S": "CG", "W": "AT",
         "K": "GT", "M": "AC", "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT"}
WATSON_CRICK = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
WOBBLE = {("G", "T"), ("T", "G")}


def expand_iupac(motif, max_expansions=4096):
    """All concrete nucleotide strings matching an IUPAC motif."""
    opts = [IUPAC[ch] for ch in motif.upper()]
    n = int(np.prod([len(o) for o in opts]))
    if n > max_expansions:
        raise C.CompilerRefuse(f"motif {motif} expands to {n} strings")
    return ["".join(p) for p in itertools.product(*opts)]


@dataclass
class Motifs:
    forbidden: tuple = ()
    required: tuple = ()

    def dfas(self):
        out = []
        if self.forbidden:
            pats = [tuple(nuc_index(s)) for m in self.forbidden for s in expand_iupac(m)]
            out.append(C.dfa_forbidden(pats, 4, name="forbid(" + ",".join(self.forbidden) + ")"))
        for m in self.required:
            pats = [tuple(nuc_index(s)) for s in expand_iupac(m)]
            out.append(C.dfa_contains(pats, 4, name=f"require({m})"))
        return out

    def check(self, nuc):
        import re
        rx = lambda m: "".join("[" + IUPAC[ch] + "]" for ch in m.upper())
        return (not any(re.search("(?=" + rx(m) + ")", nuc) for m in self.forbidden)
                and all(re.search("(?=" + rx(m) + ")", nuc) for m in self.required))


@dataclass
class GCBand:
    lo: int
    hi: int

    def dfas(self):
        gc = {NUCLEOTIDES.index("C"), NUCLEOTIDES.index("G")}
        step = lambda c, y: None if c + (y in gc) > self.hi else c + (y in gc)
        return [C.build_dfa(0, step, lambda c: c >= self.lo, 4, name=f"gc[{self.lo},{self.hi}]")]

    def check(self, nuc):
        return self.lo <= nuc.count("G") + nuc.count("C") <= self.hi


def parse_dot_bracket(db):
    stack, pairs = [], []
    for i, ch in enumerate(db):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket")
            pairs.append((stack.pop(), i))
        elif ch != ".":
            raise ValueError(f"unsupported dot-bracket symbol {ch!r}")
    if stack:
        raise ValueError("unbalanced dot-bracket")
    return sorted(pairs)


@dataclass
class TargetStructure:
    """Nucleotide pairs (i, j), 0-based over the coding sequence, that must be complementary."""
    pairs: tuple
    wobble: bool = True

    @classmethod
    def from_dot_bracket(cls, db, wobble=True):
        return cls(tuple(parse_dot_bracket(db)), wobble)

    def allowed_pairs(self):
        return WATSON_CRICK | (WOBBLE if self.wobble else set())

    def factors(self, n_codons):
        """Pairwise factors over codon positions (and unary ones for pairs inside one codon)."""
        ok = self.allowed_pairs()
        tabs = {}
        for i, j in self.pairs:
            ci, oi, cj, oj = i // 3, i % 3, j // 3, j % 3
            if max(ci, cj) >= n_codons:
                raise ValueError("pair outside the coding sequence")
            if ci == cj:
                t = np.array([float((c[oi], c[oj]) in ok) for c in CODONS])
                key = (ci,)
            else:
                (ci, oi), (cj, oj) = sorted([(ci, oi), (cj, oj)])
                t = np.array([[float((a[oi], b[oj]) in ok) for b in CODONS] for a in CODONS])
                key = (ci, cj)
            tabs[key] = tabs[key] * t if key in tabs else t
        return [(k, v) for k, v in sorted(tabs.items())]

    def check(self, nuc):
        ok = self.allowed_pairs()
        return all((nuc[i], nuc[j]) in ok for i, j in self.pairs)


@dataclass
class CDSRoute:
    """A compiled coding-sequence constraint. route reads codon indices (0..63) at n_codons positions."""
    route: object
    n_codons: int
    protein: str
    parts: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def check(self, codons):
        """Independent nucleotide-level check of every constraint, for tests and audits."""
        from ..alphabet import translate
        nuc = "".join(CODONS[c] for c in codons)
        prot = self.protein + "*"
        return len(codons) == self.n_codons and translate(codons) == prot and all(p.check(nuc) for p in self.parts)


def compile_cds(protein, constraints=(), max_cells=2 * 10 ** 7):
    """One exact route for: codons translate to protein + stop, and every constraint holds."""
    from ..alphabet import lift_to_codons
    L = len(protein) + 1
    mask = synonymous_mask(protein, stop=True)
    dfas = [d for c in constraints if hasattr(c, "dfas") for d in c.dfas()]
    structs = [c for c in constraints if isinstance(c, TargetStructure)]
    stats = {}
    codon_dfa = None
    if dfas:
        nuc = dfas[0]
        for d in dfas[1:]:
            nuc = C.product_dfa(nuc, d, 4, name=f"{nuc.name}&{d.name}")
        codon_dfa = C.minimize_dfa(lift_to_codons(nuc, minimise_symbols=False)[0])
        stats["automaton_states"] = codon_dfa.n
    fg = None
    if structs:
        fs = [f for s in structs for f in s.factors(L)]
        fg = C.FactorGraph(L, 64, fs, order=C.min_degree_order(L, [k for k, _ in fs]), name="structure")
        stats["structure_width"] = C.induced_width(L, [(i,) for i in range(L)] + [k for k, _ in fs], fg.order)
    if codon_dfa is not None and fg is not None:
        inner = C.DFAFactorProduct(codon_dfa, fg, max_cells=max_cells, allowed=mask)
        stats["product_width"], stats["product_domain"] = inner.width, inner.domain
    else:
        inner = codon_dfa if codon_dfa is not None else fg
    if inner is None:
        inner = C.build_dfa(0, lambda s, y: 0, lambda s: True, 64, name="free")
    return CDSRoute(C.Masked(inner, mask, name="cds"), L, protein.upper(), list(constraints), stats)
