"""
Does the trajectory correction change what we sample from an mRNA model?

The question E1 answers for text cannot be answered the same way here. E1 measured total variation against
a REJECTION reference: draw native samples, keep the ones satisfying C. For a coding sequence the constraint
includes the protein, whose native probability is ~10^-40, so no rejection reference exists.

Instead we use a high-particle run of the corrected sampler as the reference. What this measures, stated
exactly: **the size of the trajectory correction** -- how far step-exact decoding's output distribution sits
from the corrected sampler's at higher K. It does NOT independently establish that the corrected sampler is
exact on this model; if K = 8 and K = 32 shared a bias, this design could not see it. The warrant for
treating the high-K run as a reference is the route-generic exactness theorem plus its enumeration check on
the toy families where the exact conditional IS computable (TV <= 4e-16 at L = 4, 6), not this experiment.

The control that makes the comparison readable is the corrected sampler at small K: if it sits at the
sampling floor while step-exact sits well above it, then the correction changes the sampled distribution and
has converged by K = 8.

    reference    corrected sampler, K = K_REF, resampled to unweighted draws
    corrected    corrected sampler, K = K_ARM            (expected: at the floor)
    step_exact   same proposal at K = 1, weights ignored (expected: above the floor)

Metric: pairwise total variation over a fixed random subsample of codon-position pairs, against a matched
sampling floor obtained by splitting the reference. Reported per gene, written after each gene so that a
run stopped by a scheduler time limit still returns what it finished.

    python experiments/mrna_bias_gpu.py <weights> <cds.json> <out.json> [n_genes]
"""
import json
import random
import sys
import time

import numpy as np
import torch

from twistroute.decode.smc import select_particles, twisted_smc
from twistroute.diagnostics import pairwise_tv
from twistroute.torch_routes.build import torch_route
from twistroute_bio.alphabet import CODONS, translate
from twistroute_bio.constraints import Motifs, compile_cds
from twistroute_bio.models.mrnautilus import MRNAutilus, codon_classes

SITES = ("GAATTC", "GGATCC", "AAGCTT", "CTCGAG", "CATATG", "TCTAGA", "GTCGAC", "GCGGCCGC")
T_STEPS, K_REF, K_ARM = 16, 32, 8
N_REF, N_ARM, N_PAIRS = 2048, 768, 400
CHUNK_ROWS = 512


def draws(den, classes, route, n_codons, nrun, K, seed, dev="cpu", chunk_rows=CHUNK_ROWS, **kw):
    """`nrun` runs of K particles, chunked by row budget; returns (nrun, n_codons) unweighted draws.

    `dev` must match the device the route and denoiser live on: torch.rand refuses a CPU generator when the
    tensors are on CUDA, which is not caught by any CPU test.
    """
    per = max(1, chunk_rows // K)
    out, rng = [], np.random.default_rng(seed)
    for i in range(0, nrun, per):
        o = twisted_smc(den, classes, route, N=n_codons, T=T_STEPS, nrun=min(per, nrun - i), K=K,
                        gen=torch.Generator(device=dev).manual_seed(seed + i),
                        schedule="independent", return_particles=True, **kw)
        out.append(select_particles(o, rng))
    return np.concatenate(out)


def run(weights, cds_path, out_path, n_genes=12):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.ones(1, device=dev)                                        # fail fast on a bad node
    hum = json.load(open(cds_path))
    # Weights in float32 with autocast doing the bf16 compute: this is the configuration the encoder was
    # validated in. Loading bf16 weights under autocast puts bf16 params against float32 layer norms, which
    # is the dtype mismatch that killed three earlier GPU jobs on this project. impl="torch" for the same
    # reason -- the non-flash path is the one checked bit-identical against upstream, and flash-attn version
    # skew has cost more time here than the speed is worth at these sequence lengths.
    model = MRNAutilus(weights, device=dev, dtype=torch.float32, impl="torch")
    print(f"model loaded on {dev}, impl=torch, "
          f"{sum(p.numel() for p in model.net.parameters())/1e6:.1f}M params", flush=True)
    classes = codon_classes(dev)
    results = []

    targets = sorted([h for h in hum if 90 <= h["n_codons"] <= 260], key=lambda h: h["n_codons"])[:int(n_genes)]
    for h in targets:
        nat = h["cds"]
        protein = translate([CODONS.index(nat[i:i + 3]) for i in range(0, len(nat), 3)])[:-1]
        cds = compile_cds(protein, [Motifs(forbidden=SITES)])
        route = torch_route(cds.route, dev)
        den = model.cds_window(cds.n_codons)
        N = cds.n_codons
        t0 = time.time()

        ref = draws(den, classes, route, N, N_REF, K_REF, seed=11, dev=dev, defensive=0.25, ess_min=0.5)
        cor = draws(den, classes, route, N, N_ARM, K_ARM, seed=101, dev=dev, defensive=0.25, ess_min=0.5)
        ste = draws(den, classes, route, N, N_ARM, 1, seed=201, dev=dev, defensive=0.0, ess_min=0.0)

        rp = random.Random(7)
        pairs = [tuple(sorted(rp.sample(range(N), 2))) for _ in range(N_PAIRS)]
        half = len(ref) // 2
        floor = float(np.mean([
            pairwise_tv(ref[q[:min(N_ARM, half)]], ref[q[half:]], 64, pairs=pairs)
            for q in (np.random.default_rng(1000 + s).permutation(len(ref)) for s in range(5))]))
        row = dict(gene=h["gene"], n_codons=N, seconds=round(time.time() - t0, 1), floor=floor,
                   tv_corrected=pairwise_tv(cor, ref, 64, pairs=pairs),
                   tv_step_exact=pairwise_tv(ste, ref, 64, pairs=pairs),
                   valid_corrected=float(np.mean([cds.check(list(map(int, s))) for s in cor[:200]])),
                   valid_step_exact=float(np.mean([cds.check(list(map(int, s))) for s in ste[:200]])))
        row["excess_corrected"] = row["tv_corrected"] - floor
        row["excess_step_exact"] = row["tv_step_exact"] - floor
        results.append(row)
        print(f"{h['gene']:<8} N={N:<4} {row['seconds']:>6.0f}s  floor {floor:.4f}  "
              f"corrected {row['excess_corrected']:+.4f}  step-exact {row['excess_step_exact']:+.4f}", flush=True)
        json.dump(dict(results=results, T=T_STEPS, K_ref=K_REF, K_arm=K_ARM,
                       n_ref=N_REF, n_arm=N_ARM, n_pairs=N_PAIRS), open(out_path, "w"))
    return results


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], *(sys.argv[4:5] or []))
