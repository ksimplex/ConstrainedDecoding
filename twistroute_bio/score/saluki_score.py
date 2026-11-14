"""Score designed transcripts with Saluki (mRNA half-life).

Saluki reads a 6-channel encoding: four one-hot nucleotide channels, a coding-frame track, and a splice-site
track, zero-padded to 12,288 positions. Loader verified against the model's own shipped test predictions at
Pearson 1.00000 (max abs diff 5e-4), so the encoding below reproduces inference exactly.

For designed constructs the splice track is all zero: a synthetic transcript delivered as mRNA has no introns
and therefore no exon-junction marks. That is a real difference from the training distribution, where the
track carries information, and it is the reason `sensitivity()` exists -- if Saluki's output barely moves
when only synonymous codons change, it is reading signal we are holding fixed and is useless as an
instrument for ranking CDS designs.
"""
import numpy as np

SEQLEN = 12288
NT = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def encode(utr5, cds, utr3, splice=None):
    """(utr5, cds, utr3) -> (12288, 6) float32. Coding track marks CDS positions."""
    seq = (utr5 + cds + utr3).upper()
    x = np.zeros((SEQLEN, 6), dtype=np.float32)
    n = min(len(seq), SEQLEN)
    for i, ch in enumerate(seq[:n]):
        j = NT.get(ch)
        if j is not None:
            x[i, j] = 1.0
    a, b = len(utr5), len(utr5) + len(cds)
    x[a:min(b, SEQLEN), 4] = 1.0
    if splice is not None:
        for p in splice:
            if p < SEQLEN:
                x[p, 5] = 1.0
    return x


def score(model, rows, batch_size=8):
    """rows: list of (utr5, cds, utr3) -> predicted half-life, one per row."""
    X = np.stack([encode(*r) for r in rows])
    return model.predict(X, batch_size=batch_size, verbose=0).ravel()


# --- ensemble -------------------------------------------------------------------------------------
# The 50 checkpoints under train_gru are f{0..9}_c{0..4}: 10 cross-validation FOLDS x 5 separate half-life
# DATASETS. The `c` index is NOT a replicate -- each is its own measurement with its own scale (empirically
# c0 centres near 0.2, c2 near 1.0, c3 near 1.4 on the same sequences). Averaging raw predictions across `c`
# therefore mixes incommensurable units and manufactures variance: doing so gave a between-model sd of 1.159
# against a between-gene sd of 0.098, i.e. the "signal" was almost entirely scale disagreement.
#
# The sound protocol, and the one this implements: score every design with every checkpoint, then compare
# designs WITHIN a checkpoint (paired), never across. Absolute values are meaningless here anyway -- the
# splice track is all-zero for synthetic constructs, which is off-distribution -- so every reported quantity
# is a within-checkpoint contrast aggregated over checkpoints.
import glob
import re


def ensemble_paths(root="datasets/deeplearning/train_gru", head="model0"):
    out = {}
    for p in sorted(glob.glob(f"{root}/*/train/{head}_best.h5")):
        m = re.search(r"/f(\d+)_c(\d+)/", p)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def score_ensemble(loader, rows, paths, batch_size=8):
    """-> (n_models, n_rows) matrix. Rows are designs; compare DOWN columns within a row of the matrix.

    `loader` is saluki_load.load. Returns the raw matrix rather than an average, because the caller must do
    the within-checkpoint pairing itself; collapsing here would invite the cross-dataset averaging bug above.
    """
    import numpy as np
    X = np.stack([encode(*r) for r in rows])
    return np.array([loader(p).predict(X, batch_size=batch_size, verbose=0).ravel()
                     for p in paths]), list(paths)


def paired_contrast(P, i, j):
    """Mean within-checkpoint difference between design i and design j, with a paired CI over checkpoints."""
    import numpy as np
    d = P[:, i] - P[:, j]
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se),
                n_models=len(d), frac_positive=float((d > 0).mean()))
