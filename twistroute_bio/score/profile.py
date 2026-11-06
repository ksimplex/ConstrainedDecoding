"""
Distributional and positional sequence metrics: the panel the codon-optimisation literature reports.

Per-position reconstruction accuracy is not a metric this field uses (no published learned method reports
it against a frequency baseline as a primary result), and it cannot separate a model from a frequency table
anyway: a product model reproduces the per-position mode by construction. These metrics can.

    codon_table(cds_list)      absolute (per-thousand) and relative synonymous codon frequencies, estimated
                               from a corpus of real coding sequences
    cai(seq, table)            codon adaptation index (Sharp & Li, Nucleic Acids Res. 15(3):1281-1295, 1987,
                               doi:10.1093/nar/15.3.1281): the geometric mean of w = f / f_max within each
                               synonymous family. One number per sequence; a *distributional* measure that a
                               frequency table maximises by definition, so it is a control, not a target.
    minmax_profile(seq, table) %MinMax (Clarke & Clark, "Rare Codons Cluster", PLoS ONE 3(10):e3412, 2008,
                               doi:10.1371/journal.pone.0003412): per-window relative rareness, +100 for a
                               window using only the most common synonymous codons and -100 for one using
                               only the rarest. This is the *positional* measure, and the discriminating one:
                               it asks where rare codons sit, not which codon is modal. A product model
                               cannot place them, because placement is a property of the joint.
    profile_distance(a, b)     distance between two %MinMax profiles, aligned and by dynamic time warping
    gc_content, gc3            G+C overall and at third positions

Convention note. Clarke & Clark compute frequencies from the organism's own ORFeome and use absolute codon
frequencies, with X_avg the arithmetic mean over the synonymous family. `codon_table` follows that: it
estimates the table from whatever corpus it is given rather than hardcoding one. Windows are reported at
their midpoint, and the default window is the paper's 18 codons.
"""
import collections
import itertools
import math

from ..alphabet import CODONS, GENETIC_CODE

_SYN = collections.defaultdict(list)
for _c in CODONS:
    _SYN[GENETIC_CODE[_c]].append(_c)


def codon_table(cds_list, pseudocount=1.0):
    """Codon frequencies from a corpus of coding-sequence strings.

    Returns {"abs": per-thousand over all codons, "rel": fraction within the synonymous family,
    "n_codons": corpus size}. A pseudocount keeps every codon positive so CAI weights and %MinMax
    denominators are finite on a small corpus.
    """
    cnt = collections.Counter()
    for s in cds_list:
        s = s.upper().replace("U", "T")
        cnt.update(s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3))
    counts = {c: cnt.get(c, 0) + pseudocount for c in CODONS}
    total = sum(counts.values())
    rel = {c: counts[c] / sum(counts[d] for d in _SYN[GENETIC_CODE[c]]) for c in CODONS}
    return {"abs": {c: 1000.0 * counts[c] / total for c in CODONS}, "rel": rel,
            "n_codons": int(sum(cnt.values()))}


def _codons(seq):
    s = seq.upper().replace("U", "T")
    return [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]


def cai_weights(table):
    """w(c) = f(c) / max over the synonymous family, on the relative table (Sharp & Li 1987)."""
    rel = table["rel"]
    return {c: rel[c] / max(rel[d] for d in _SYN[GENETIC_CODE[c]]) for c in CODONS}


def cai(seq, table, skip_stop=True, skip_single=True):
    """Geometric mean of the CAI weights over the sequence.

    Single-codon families (Met, Trp) carry w = 1 by definition and are conventionally excluded, as are
    stop codons; both exclusions are the standard ones and are options here so the choice is visible.
    """
    w = cai_weights(table)
    vals = []
    for c in _codons(seq):
        aa = GENETIC_CODE[c]
        if (skip_stop and aa == "*") or (skip_single and len(_SYN[aa]) == 1):
            continue
        vals.append(math.log(w[c]))
    if not vals:
        raise ValueError("no scorable codons")
    return math.exp(sum(vals) / len(vals))


def minmax_profile(seq, table, window=18, use_abs=True):
    """%MinMax over sliding windows (Clarke & Clark 2008).

    Returns [(midpoint_codon_index, value), ...] with value in [-100, 100]: positive where the window
    uses commoner codons than the synonymous average, negative where it uses rarer ones. Stop codons and
    single-codon families contribute 0 to every term, which is what the windowed ratio implies for them.
    """
    f = table["abs"] if use_abs else table["rel"]
    cods = _codons(seq)
    act, avg, mx, mn = [], [], [], []
    for c in cods:
        fam = _SYN[GENETIC_CODE[c]]
        vals = [f[d] for d in fam]
        act.append(f[c]); avg.append(sum(vals) / len(vals)); mx.append(max(vals)); mn.append(min(vals))
    out = []
    for i in range(len(cods) - window + 1):
        sl = slice(i, i + window)
        a, v = sum(act[sl]), sum(avg[sl])
        if a >= v:
            den = sum(mx[sl]) - v
            out.append((i + window // 2, 100.0 * (a - v) / den if den > 0 else 0.0))
        else:
            den = v - sum(mn[sl])
            out.append((i + window // 2, -100.0 * (v - a) / den if den > 0 else 0.0))
    return out


def profile_distance(a, b, warp=True):
    """Distance between two %MinMax profiles, given as lists of (index, value).

    Sequences encoding the same protein have profiles of identical length, so `aligned` (mean absolute
    difference position by position) is well defined and is the primary number. `dtw` additionally allows
    the two profiles to shift relative to one another, which is the comparison published work uses; it
    rewards putting rare-codon clusters in roughly the right place rather than exactly.
    """
    x = [v for _, v in a]
    y = [v for _, v in b]
    aligned = (sum(abs(p - q) for p, q in zip(x, y)) / min(len(x), len(y))) if x and y else float("nan")
    if not warp:
        return {"aligned": aligned}
    n, m = len(x), len(y)
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = abs(x[i - 1] - y[j - 1]) + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return {"aligned": aligned, "dtw": prev[m] / (n + m)}


def gc_content(seq):
    s = seq.upper().replace("U", "T")
    return (s.count("G") + s.count("C")) / len(s) if s else float("nan")


def gc3(seq):
    """G+C at third codon positions, where synonymous choice mostly acts."""
    third = [c[2] for c in _codons(seq)]
    return sum(b in "GC" for b in third) / len(third) if third else float("nan")
