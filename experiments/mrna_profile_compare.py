"""
One constraint family, end to end: restriction-site avoidance on real human coding sequences.

Compares four ways of producing a constrained coding sequence for a fixed protein, against the natural
sequence as the reference for what human codon structure looks like:

    corrected      twisted SMC on mRNAutilus with the compiled route (this method)
    step_exact     the same proposal at K = 1 with the weights ignored (the uncorrected baseline)
    freq_repair    most-frequent codon per residue, then randomised local repair until valid
    maxcai_repair  max-CAI greedy, then the same repair

The target metric is the distance between a design's %MinMax profile and the natural sequence's: it asks
whether rare codons are placed where the real gene places them. CAI, GC and GC3 are reported alongside as
distributional controls. A frequency table maximises CAI by construction and is pegged at +100 %MinMax, so
those columns are expected to look "better" for the rule arms and say nothing about positional fidelity.

    python experiments/mrna_profile_compare.py <checkpoint> <cds.json> <table.json> <out.json>
"""
import json
import random
import sys
import time

import numpy as np
import torch

from twistroute.decode.smc import twisted_smc
from twistroute.torch_routes.build import torch_route
from twistroute_bio.alphabet import CODONS, GENETIC_CODE, translate
from twistroute_bio.constraints import Motifs, compile_cds
from twistroute_bio.models.mrnautilus import MRNAutilus, codon_classes
from twistroute_bio.score import profile as prof

SITES = ("GAATTC", "GGATCC", "AAGCTT", "CTCGAG", "CATATG", "TCTAGA", "GTCGAC", "GCGGCCGC")
SYN = prof._SYN


def seq_of(idx):
    return "".join(CODONS[int(c)] for c in idx)


def repair(protein, cds, start_pick, tries=4000, seed=0):
    """Start from a per-residue codon choice and randomly re-pick synonymous codons until valid."""
    rng = random.Random(seed)
    aa = protein + "*"
    cur = [start_pick(SYN[a]) for a in aa]
    for _ in range(tries):
        idx = [CODONS.index(c) for c in cur]
        if cds.check(idx):
            return idx
        i = rng.randrange(len(cur))
        fam = SYN[aa[i]]
        if len(fam) > 1:
            cur[i] = rng.choice(fam)
    return None


def run(ckpt, cds_path, table_path, out_path, T=16, K=8, nrun=2, seed=0):
    hum = json.load(open(cds_path))
    table = json.load(open(table_path))
    table = {"abs": table["abs"], "rel": table["rel"], "n_codons": table["n_codons"]}
    rel = table["rel"]
    model = MRNAutilus(ckpt, device="cpu")
    classes = codon_classes()
    results = []

    targets = [h for h in hum if h["n_codons"] <= 200 and any(s in h["cds"] for s in SITES)]
    for h in targets:
        nat = h["cds"]
        protein = translate([CODONS.index(nat[i:i + 3]) for i in range(0, len(nat), 3)])[:-1]
        cds = compile_cds(protein, [Motifs(forbidden=SITES)])
        route = torch_route(cds.route, "cpu")
        nat_prof = prof.minmax_profile(nat, table)
        t0 = time.time()

        arms = {}
        # --- model arms -------------------------------------------------------------------------
        den = model.cds_window(cds.n_codons)
        for name, kk, defensive in [("corrected", K, 0.25), ("step_exact", 1, 0.0)]:
            gen = torch.Generator().manual_seed(seed)
            out = twisted_smc(den, classes, route, N=cds.n_codons, T=T,
                              nrun=nrun * (K // kk), K=kk, gen=gen, defensive=defensive,
                              ess_min=0.5 if kk > 1 else 0.0, schedule="independent",
                              return_particles=True)
            seqs = np.asarray(out["classes"]).reshape(-1, cds.n_codons)
            lw = np.asarray(out["logw"]).reshape(-1)
            arms[name] = [list(map(int, s)) for s, w in zip(seqs, lw) if np.isfinite(w)]

        # --- rule arms --------------------------------------------------------------------------
        arms["freq_repair"] = [r for s in range(len(arms["corrected"]))
                               for r in [repair(protein, cds, lambda f: max(f, key=lambda c: rel[c]), seed=s)]
                               if r is not None]
        w = prof.cai_weights(table)
        arms["maxcai_repair"] = [r for s in range(len(arms["corrected"]))
                                 for r in [repair(protein, cds, lambda f: max(f, key=lambda c: w[c]), seed=s)]
                                 if r is not None]

        for name, idxs in arms.items():
            for idx in idxs:
                s = seq_of(idx)
                d = prof.profile_distance(prof.minmax_profile(s, table), nat_prof)
                results.append(dict(gene=h["gene"], arm=name, valid=bool(cds.check(idx)),
                                    aligned=d["aligned"], dtw=d["dtw"], cai=prof.cai(s, table),
                                    gc=prof.gc_content(s), gc3=prof.gc3(s),
                                    identity=sum(a == b for a, b in zip(s, nat)) / len(nat)))
        # natural reference
        results.append(dict(gene=h["gene"], arm="natural", valid=bool(cds.check(
            [CODONS.index(nat[i:i + 3]) for i in range(0, len(nat), 3)])),
            aligned=0.0, dtw=0.0, cai=prof.cai(nat, table), gc=prof.gc_content(nat),
            gc3=prof.gc3(nat), identity=1.0))
        print(f"{h['gene']:<7} n={cds.n_codons:>4} states={cds.stats.get('automaton_states')} "
              f"{time.time()-t0:.0f}s  " + " ".join(f"{k}:{len(v)}" for k, v in arms.items()), flush=True)

    json.dump(dict(results=results, table_n_codons=table["n_codons"], T=T, K=K, sites=list(SITES)),
              open(out_path, "w"))
    return results


if __name__ == "__main__":
    run(*sys.argv[1:5])
