"""Physical (thermodynamic) scoring of designed transcripts. ViennaRNA only -- nothing learned, nothing
fitted, so no instrument here can be accused of favouring a learned sampler.

The metric decomposition is the point of this module, so it is worth stating plainly.

Global minimum free energy is LinearDesign's OWN objective. Scoring designs on it is a home-turf match that
the exact optimiser wins by construction, and winning or losing it would tell us nothing about whether
sampling beats optimising. But minimising global folding energy is not unambiguously good: the ribosome has
to load at the 5' end, and an over-structured start region impedes initiation. A design can therefore be
"better" by total folding energy while being worse at the thing the folding energy is a proxy for.

So we split it:

    accessibility_5p   probability the start window is UNPAIRED        <- primary metric
    body_energy        ensemble free energy of the remainder            <- held matched
    mfe                whole-transcript MFE                             <- reference only, flagged

Both numbers come from the same partition function, so there is no extra cost in computing both.
"""
import math

import RNA


def _fc(seq):
    fc = RNA.fold_compound(seq.upper().replace("T", "U"))
    fc.pf()
    return fc


def unpaired_probability(seq, start, end, fc=None):
    """Mean probability that positions [start, end) are unpaired, from the base-pair probability matrix.

    `RNA.fold_compound.bpp()` is 1-indexed and upper-triangular: bpp[i][j] for i < j. A position's unpaired
    probability is 1 minus its total pairing probability, so both triangle directions must be summed --
    missing one silently halves the pairing mass and inflates accessibility.
    """
    fc = fc or _fc(seq)
    bpp = fc.bpp()
    n = len(seq)
    tot = 0.0
    for p in range(start + 1, min(end, n) + 1):           # 1-indexed
        paired = 0.0
        for q in range(1, n + 1):
            if q == p:
                continue
            i, j = (p, q) if p < q else (q, p)
            paired += bpp[i][j]
        tot += max(0.0, 1.0 - paired)
    return tot / max(1, min(end, n) - start)


def score(seq, start_window=(0, 45), fc=None):
    """All physical metrics for one transcript, from a single partition function.

    `start_window` is in nucleotides from the transcript 5' end and should span the region a ribosome must
    access -- by default the 5'UTR end plus the first codons.
    """
    s = seq.upper().replace("T", "U")
    fc = fc or _fc(s)
    _, mfe = fc.mfe()
    ens = fc.pf()[1]
    a, b = start_window
    acc = unpaired_probability(s, a, b, fc=fc)
    # body energy: the same ensemble energy with the start window excluded, so the two metrics are
    # separable rather than two views of one number
    body = _fc(s[b:])
    _, body_mfe = body.mfe()
    body_ens = body.pf()[1]
    return dict(accessibility_5p=acc, body_ensemble_energy=body_ens, body_mfe=body_mfe,
                mfe=mfe, ensemble_energy=ens, length=len(s),
                mfe_per_nt=mfe / len(s), body_energy_per_nt=body_ens / max(1, len(s) - b))


def ensemble_defect(seq, structure):
    """Expected number of incorrectly-paired positions relative to a target structure, normalised."""
    fc = _fc(seq)
    return fc.ensemble_defect(structure) / len(seq)
