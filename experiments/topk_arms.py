"""Candidate generation for the top-k experiment: one function per arm, all returning CDS strings.

Each arm must emit k candidates satisfying the same constraint set C, and validity is CHECKED here rather
than trusted -- every arm's output goes through the compiled route's independent nucleotide-level check
before it is returned, because an arm that silently emits invalid sequences would flatter itself on every
downstream metric.

Design notes per arm are in docs/topk_experiment_plan.md Sec. 3. Two that bear on the code:

- LinearDesign cannot express C at all (the shipped binary takes only a protein, a codon table and
  --lambda), so its arm sweeps lambda and reports what FRACTION of outputs happen to satisfy C. Its lambda
  also saturates: past a low threshold CAI pins to 1.000 and the sequence stops changing, so the sweep is
  deduplicated and the distinct count recorded.
- max-CAI greedy and max-frequency greedy are the SAME arm: w = f/f_max is monotone in f within each
  synonymous family, so their argmaxes coincide. Running both would double-count one baseline.
"""
import os
import random
import subprocess
import tempfile

LD_BIN = os.environ.get("LINEARDESIGN_BIN", "")
LD_TABLE = os.environ.get("LINEARDESIGN_TABLE", "")
# The compiled binary dlopen()s src/Utils/libraries/LinearDesign_Mac_M1.so by RELATIVE path, so it aborts
# (SIGABRT, empty stdout) unless the process cwd is the LinearDesign repo root. Not documented anywhere.
LD_ROOT = os.environ.get("LINEARDESIGN_ROOT", os.path.dirname(os.path.dirname(LD_BIN)) if LD_BIN else "")


def _valid(cds, route_check, protein):
    from twistroute_bio.alphabet import CODONS
    try:
        idx = [CODONS.index(cds[i:i + 3]) for i in range(0, len(cds), 3)]
    except ValueError:
        return False
    return route_check(idx)


def arm_lineardesign(protein, k=32, lam_max=4.0):
    """Lambda sweep. Returns (candidates, n_distinct, n_runs) -- candidates may VIOLATE C by design."""
    if not (LD_BIN and os.path.exists(LD_BIN)):
        return [], 0, 0
    lams = [round(lam_max * i / max(1, k - 1), 4) for i in range(k)]
    seen, out = set(), []
    for lam in lams:
        p = subprocess.run([LD_BIN, str(lam), "0", LD_TABLE], input=protein + "\n",
                           capture_output=True, text=True, timeout=600, cwd=LD_ROOT or None)
        if p.returncode != 0:
            raise RuntimeError(f"LinearDesign exited {p.returncode}: {p.stderr[-200:]}")
        seq = ""
        for line in p.stdout.replace("\r", "\n").splitlines():
            if "mRNA sequence:" in line:
                seq = line.split(":", 1)[1].strip().replace("U", "T")
        if seq and seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out, len(seen), len(lams)


def arm_dnachisel(protein, natural_cds, sites, gc_band, k=32, seed=0, structure=None):
    """k independent DNA Chisel runs from different seeds. Constraints hard, MaximizeCAI soft."""
    import numpy as np
    from dnachisel import (DnaOptimizationProblem, EnforceTranslation, AvoidPattern,
                           EnforceGCContent, EnforceRegionsCompatibility, Location, MaximizeCAI)
    PAIR = {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G"), ("G", "T"), ("T", "G")}
    n = len(natural_cds)
    out = []
    for s in range(k):
        np.random.seed(seed + s)
        random.seed(seed + s)
        cons = [EnforceTranslation(location=Location(0, n))] + [AvoidPattern(x) for x in sites]
        if gc_band:
            cons.append(EnforceGCContent(mini=gc_band[0], maxi=gc_band[1], location=Location(0, n)))
        if structure:
            i, j, stem = structure
            def compat(l1, l2, problem):
                a = l1.extract_sequence(problem.sequence); b = l2.extract_sequence(problem.sequence)
                return all((x, y) in PAIR for x, y in zip(a, b[::-1]))
            cons.append(EnforceRegionsCompatibility(
                locations=[(i, i + stem, 1), (j - stem + 1, j + 1, 1)],
                compatibility_condition=compat, condition_label=f"stem{stem}"))
        # a different random starting point per seed, else every run returns the same local solution
        start = _shuffle_synonymous(natural_cds, random.Random(seed + s))
        try:
            prob = DnaOptimizationProblem(sequence=start, constraints=cons,
                                          objectives=[MaximizeCAI(species="h_sapiens")], logger=None)
            prob.resolve_constraints()
            prob.optimize()
            if prob.all_constraints_pass():
                out.append(prob.sequence)
        except Exception:
            continue
    return out


def _shuffle_synonymous(cds, rng):
    from twistroute_bio.alphabet import CODONS, GENETIC_CODE
    syn = {}
    for c in CODONS:
        syn.setdefault(GENETIC_CODE[c], []).append(c)
    return "".join(rng.choice(syn[GENETIC_CODE[cds[i:i + 3]]]) for i in range(0, len(cds), 3))


def arm_greedy_cai(protein, table):
    """Deterministic max-CAI floor. Identical to max-frequency greedy; one arm, not two."""
    from twistroute_bio.alphabet import CODONS, GENETIC_CODE
    syn = {}
    for c in CODONS:
        syn.setdefault(GENETIC_CODE[c], []).append(c)
    return "".join(max(syn[aa], key=lambda c: table["abs"][c]) for aa in protein + "*")


def default_chunk_rows(n_codons, device="cpu", reserve_gb=3.0):
    """Rows per forward pass that fit in available memory, rather than a constant tuned for a 48 GB GPU.

    Attention is quadratic in sequence length, so the transient cost of one layer is roughly
    rows * heads * L^2 * 4 bytes, and it recurs every layer. With L ~ 250 tokens and 20 heads, 256 rows is
    ~1.3 GB per attention matrix -- fine on an L40S, fatal on a 16 GB laptop, where it drove the machine
    into swap (load average 440 on 8 cores, ~14 MB free of 16 GB) and took the host down with it.

    This is deliberately conservative: the cost of an over-small batch is some wasted throughput, while the
    cost of an over-large one is thrashing that makes the whole machine unusable.
    """
    import torch
    L = n_codons + 160                                   # CDS plus the fixed UTR context
    per_row_gb = 20 * (L ** 2) * 4 * 3 / 1e9             # heads * L^2 * fp32 * a few live tensors
    if device == "cuda" and torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9 - reserve_gb
    else:
        import shutil
        try:                                             # psutil is not guaranteed; fall back to a floor
            import psutil
            free_gb = psutil.virtual_memory().available / 1e9 - reserve_gb
        except ImportError:
            free_gb = 2.0
    return int(max(2, min(512, free_gb / max(per_row_gb, 1e-6))))


def arm_sampler(denoiser, classes, route, n_codons, k=32, K=8, T=16, seed=0,
                step_exact=False, device="cpu", chunk_rows=None):
    """Our constrained sampler (`step_exact=False`) or the step-exact baseline (`step_exact=True`).

    Both share the same proposal and the same route, so the ONLY difference is whether trajectory weights
    are applied: step-exact runs K = 1 and ignores them, which is what a per-position-masking decoder does.
    That makes the pair a clean ablation of the trajectory correction rather than of the constraint handling.

    Resampling is OFF (`defensive=0, ess_min=0`). The mRNA routes here are factor graphs and therefore
    unambiguous, so the no-resampling rule is precaution rather than necessity -- but the stress-B analysis
    showed every biased Z-hat came from a resampling-on arm, and there is no reason to spend the risk.
    """
    import numpy as np
    import torch
    from twistroute.decode.smc import select_particles, twisted_smc
    from twistroute_bio.alphabet import CODONS

    if chunk_rows is None:
        chunk_rows = default_chunk_rows(n_codons, device)
    per = max(1, chunk_rows // max(1, K if not step_exact else 1))
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, k, per):
        o = twisted_smc(denoiser, classes, route, N=n_codons, T=T, nrun=min(per, k - i),
                        K=1 if step_exact else K,
                        gen=torch.Generator(device=device).manual_seed(seed + i),
                        schedule="independent", return_particles=True,
                        defensive=0.0, ess_min=0.0)
        out.append(select_particles(o, rng))
    draws = np.concatenate(out)[:k]
    return ["".join(CODONS[int(c)] for c in row) for row in draws]
