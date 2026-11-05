"""
ViennaRNA scorers. These are deterministic functions of a sequence, used to RANK valid samples (E6b), never
inside a route. An external, algorithmic score cannot be gamed by an optimiser exploiting predictor error,
which is why the headline comparison uses it rather than a learned expression predictor.

Each scorer takes a coding sequence as a nucleotide string (T or U both accepted). mfe is in kcal/mol.
"""
import RNA


def _rna(seq):
    return seq.upper().replace("T", "U")


def mfe_structure(seq):
    """(dot-bracket, minimum free energy in kcal/mol) of the MFE structure."""
    s, e = RNA.fold_compound(_rna(seq)).mfe()
    return s, float(e)


def mfe(seq):
    return mfe_structure(seq)[1]


def structure_distance(seq, target):
    """Base-pair distance between the sequence's MFE structure and a target dot-bracket."""
    s, _ = mfe_structure(seq)
    if len(s) != len(target):
        raise ValueError("target length does not match the sequence")
    return int(RNA.bp_distance(s, target))


def ensemble_defect(seq, target):
    """Expected number of incorrectly paired positions relative to the target, under the Boltzmann ensemble."""
    fc = RNA.fold_compound(_rna(seq))
    fc.pf()
    return float(fc.ensemble_defect(target))


def fraction_paired(seq):
    """Fraction of positions paired in the MFE structure."""
    s, _ = mfe_structure(seq)
    return sum(c != "." for c in s) / len(s) if s else 0.0


def score_batch(seqs, target=None):
    """List of per-sequence score dicts: mfe, fraction_paired, and (with a target) structure_distance and
    ensemble_defect. One fold_compound per sequence; for large pools call in a worker pool."""
    out = []
    for q in seqs:
        s, e = mfe_structure(q)
        row = {"mfe": e, "fraction_paired": sum(c != "." for c in s) / len(s) if s else 0.0, "mfe_structure": s}
        if target is not None:
            row["structure_distance"] = int(RNA.bp_distance(s, target))
            row["ensemble_defect"] = ensemble_defect(q, target)
        out.append(row)
    return out
