"""
Model-free construction baselines. For each constraint family, build a valid coding sequence without the model:
frequency-based codon choice (the standard codon-optimisation baseline, which LSCO also compares against) plus
local repair to satisfy hard constraints. A family where this baseline scores as well as the sampler is a
family where the model's distribution does no work; it is reported, not hidden. The baseline is a reference
point rather than a competitor, since it cannot offer distributional fidelity or diversity at all.

Scope note. A frequency table is a product of independent per-position distributions, so it reproduces a
model's per-position mode while being structurally unable to reproduce a joint that couples positions. On
mRNAutilus these two facts are both true at once (docs/mrna_model_validation.md): top-1 synonymous-codon
agreement is within noise of this baseline, while a synonymous shuffle preserving the protein moves the
model's predicted distribution by TV 0.261. Comparisons against this baseline should therefore use a
positional or distributional metric, not per-position accuracy.

HUMAN_CODON_FREQ: relative synonymous-codon usage in the human genome (Kazusa codon usage database), given
per amino acid and renormalised so each synonymous set sums to 1.
"""
import random
from ..alphabet import CODONS, GENETIC_CODE, nuc_index

# relative usage within each amino acid's synonymous set (human; rounded, renormalised per amino acid)
_RAW = {
 "TTT":0.46,"TTC":0.54,"TTA":0.08,"TTG":0.13,"CTT":0.13,"CTC":0.20,"CTA":0.07,"CTG":0.40,
 "ATT":0.36,"ATC":0.47,"ATA":0.17,"ATG":1.00,"GTT":0.18,"GTC":0.24,"GTA":0.12,"GTG":0.46,
 "TCT":0.19,"TCC":0.22,"TCA":0.15,"TCG":0.05,"AGT":0.15,"AGC":0.24,"CCT":0.29,"CCC":0.32,
 "CCA":0.28,"CCG":0.11,"ACT":0.25,"ACC":0.36,"ACA":0.28,"ACG":0.11,"GCT":0.27,"GCC":0.40,
 "GCA":0.23,"GCG":0.11,"TAT":0.44,"TAC":0.56,"CAT":0.42,"CAC":0.58,"CAA":0.27,"CAG":0.73,
 "AAT":0.47,"AAC":0.53,"AAA":0.43,"AAG":0.57,"GAT":0.46,"GAC":0.54,"GAA":0.42,"GAG":0.58,
 "TGT":0.45,"TGC":0.55,"TGG":1.00,"CGT":0.08,"CGC":0.18,"CGA":0.11,"CGG":0.20,"AGA":0.21,
 "AGG":0.21,"GGT":0.16,"GGC":0.34,"GGA":0.25,"GGG":0.25,"TAA":0.30,"TAG":0.24,"TGA":0.47,
}
HUMAN_CODON_FREQ = dict(_RAW)

_SYN = {}
for _c, _a in GENETIC_CODE.items():
    _SYN.setdefault(_a, []).append(_c)


def frequency_codons(protein, stop=True):
    """Most frequent synonymous codon for each residue (the plain codon-optimisation baseline)."""
    seq = [max(_SYN[a], key=lambda c: HUMAN_CODON_FREQ[c]) for a in protein.upper()]
    if stop:
        seq.append(max(_SYN["*"], key=lambda c: HUMAN_CODON_FREQ[c]))
    return "".join(seq)


def rule_construct(protein, cds_route, tries=200, seed=0):
    """Frequency-based start, then randomised local repair against a compiled CDSRoute (constraints.CDSRoute):
    repeatedly replace a codon with a random synonymous one and keep changes that do not increase the number of
    violated constraints, restarting from the frequency optimum. Returns a valid codon-index list, or None if no
    try satisfies every constraint. A model-free heuristic: success means the family is rule-solvable."""
    rng = random.Random(seed)
    aa = (protein.upper() + "*")
    syn_idx = [[CODONS.index(c) for c in _SYN[a]] for a in aa]
    base = [CODONS.index(c) for c in [max(_SYN[a], key=lambda c: HUMAN_CODON_FREQ[c]) for a in aa]]

    def violations(codons):
        return 0 if cds_route.check(codons) else sum(1 for p in cds_route.parts
                                                      if not p.check("".join(CODONS[c] for c in codons)))
    for t in range(tries):
        cur = list(base)
        v = violations(cur)
        for _ in range(60 * len(aa)):
            if v == 0 and cds_route.check(cur):
                return cur
            i = rng.randrange(len(aa))
            if len(syn_idx[i]) == 1:
                continue
            old = cur[i]; cur[i] = rng.choice(syn_idx[i])
            nv = violations(cur)
            if nv <= v:
                v = nv
            else:
                cur[i] = old
        if cds_route.check(cur):
            return cur
    return None
