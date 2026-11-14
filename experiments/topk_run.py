"""Top-k experiment driver: generate k candidates per arm per gene, score them, write one row per candidate.

The design, the metric list and the falsification condition are fixed in docs/topk_experiment_plan.md and
were written BEFORE any results existed. This script only executes it.

    python experiments/topk_run.py <weights> <cds.json> <out.csv> [n_genes] [k]

Scoring is split by cost. Physical metrics (ViennaRNA) run on every candidate. The learned instruments are
expensive and are applied to the per-arm best and a fixed random subsample, recorded as such.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topk_arms import arm_dnachisel, arm_greedy_cai, arm_lineardesign, arm_sampler
from twistroute_bio.alphabet import CODONS, translate
from twistroute_bio.constraints import GCBand, Motifs, TargetStructure, compile_cds
from twistroute_bio.score.physical import score as phys_score
from twistroute_bio.score.profile import cai, gc_content, gc3, minmax_profile, profile_distance

SITES = ("GAATTC", "GGATCC", "AAGCTT", "CTCGAG", "CATATG", "TCTAGA", "GTCGAC", "GCGGCCGC")
GC_LO, GC_HI = 0.45, 0.60
UTR5 = "GGGAAATAAGAGAGAAAAGAAGAGTAAGAAGAAATATAAGAGCCACC"
UTR3 = ("GCTGGAGCCTCGGTGGCCATGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCC"
        "CCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGC")
START_WINDOW = (0, len(UTR5) + 30)       # 5'UTR plus the first 10 codons: what a ribosome must access


def satisfies_C(cds):
    gc = gc_content(cds)
    return (not any(s in cds for s in SITES)) and GC_LO <= gc <= GC_HI


def run(weights, cds_path, out_path, n_genes=4, k=16, with_structure=False):
    n_genes, k = int(n_genes), int(k)
    import torch
    from twistroute.torch_routes.build import torch_route
    from twistroute_bio.models.mrnautilus import MRNAutilus, codon_classes

    table = json.load(open("results/codon_table.json"))
    panel = json.load(open(cds_path))[:int(n_genes)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MRNAutilus(weights, device=dev, dtype=torch.float32, impl="torch")
    classes = codon_classes(dev)

    rows = []
    for h in panel:
        nat = h["cds"]
        protein = translate([CODONS.index(nat[i:i + 3]) for i in range(0, len(nat), 3)])[:-1]
        n_nt = len(nat)
        cons = [Motifs(forbidden=SITES), GCBand.fraction(GC_LO, GC_HI, n_nt)]
        cds_route = compile_cds(protein, cons, max_cells=int(1.2e9))
        route = torch_route(cds_route.route, dev)
        den = model.cds_window(cds_route.n_codons, utr5=UTR5, utr3=UTR3)
        t0 = time.time()

        cands = {}
        cands["ours"] = arm_sampler(den, classes, route, cds_route.n_codons, k=k, K=8, seed=11, device=dev)
        cands["step_exact"] = arm_sampler(den, classes, route, cds_route.n_codons, k=k, K=1, seed=23,
                                          step_exact=True, device=dev)
        cands["dnachisel"] = arm_dnachisel(protein, nat, SITES, (GC_LO, GC_HI), k=k, seed=7)
        ld, n_distinct, n_runs = arm_lineardesign(protein, k=k)
        cands["lineardesign"] = ld
        cands["greedy_cai"] = [arm_greedy_cai(protein, table)]
        cands["natural"] = [nat]

        nat_prof = minmax_profile(nat, table)
        for arm, seqs in cands.items():
            for idx, s in enumerate(seqs):
                ok = satisfies_C(s)
                p = phys_score(UTR5 + s + UTR3, start_window=START_WINDOW)
                rows.append(dict(
                    gene=h["gene"], n_codons=len(s) // 3, arm=arm, idx=idx,
                    satisfies_C=int(ok), gc=round(gc_content(s), 4), gc3=round(gc3(s), 4),
                    cai=round(cai(s, table), 4),
                    minmax_dist=round(profile_distance(minmax_profile(s, table), nat_prof)["dtw"], 3),
                    accessibility_5p=round(p["accessibility_5p"], 4),
                    body_energy=round(p["body_ensemble_energy"], 3),
                    body_energy_per_nt=round(p["body_energy_per_nt"], 5),
                    mfe=round(p["mfe"], 2),
                    n_distinct_arm=(n_distinct if arm == "lineardesign" else len(set(seqs))),
                    n_runs_arm=(n_runs if arm == "lineardesign" else len(seqs)),
                    cds=s))
        print(f"{h['gene']:<9} {len(rows):>5} rows  {time.time()-t0:>6.0f}s  "
              + "  ".join(f"{a}:{sum(satisfies_C(x) for x in v)}/{len(v)}" for a, v in cands.items()),
              flush=True)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], *(sys.argv[4:6] or []))
