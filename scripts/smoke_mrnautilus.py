"""Smoke test: sample a short constrained coding sequence from mRNAutilus with twisted SMC.

Checks that layer 3 is wired to layers 1-2 end to end -- every sample must translate to the
requested protein and avoid the forbidden restriction sites. Runs on CPU in about a minute.

    python scripts/smoke_mrnautilus.py <checkpoint.ckpt>
"""
import sys
import torch

from twistroute.decode.smc import twisted_smc
from twistroute.torch_routes.build import torch_route
from twistroute_bio.alphabet import CODONS, translate
from twistroute_bio.constraints import Motifs, compile_cds
from twistroute_bio.models.mrnautilus import MRNAutilus, codon_classes

PROTEIN = "MKWVTFISLL"                      # 10 residues; the route adds a stop codon
UTR5 = "GGGAAATAAGAGAGAAAAGAAGAGTAAGAAGAAATATAAGAGCCACC"
SITES = ["GAATTC", "GGATCC", "AAGCTT"]      # EcoRI, BamHI, HindIII


def main(ckpt, T=8, K=4, nrun=4, seed=0):
    cds = compile_cds(PROTEIN, [Motifs(forbidden=tuple(SITES))])
    route = torch_route(cds.route, "cpu")
    model = MRNAutilus(ckpt, device="cpu")
    den = model.cds_window(cds.n_codons, utr5=UTR5)
    classes = codon_classes()
    print("route:", cds.stats, "| N:", cds.n_codons, "| denoiser:", den.info())

    gen = torch.Generator().manual_seed(seed)
    out = twisted_smc(den, classes, route, N=cds.n_codons, T=T, nrun=nrun, K=K, gen=gen,
                      defensive=0.25, schedule="independent", return_particles=True)

    seqs = torch.as_tensor(out["classes"]).reshape(-1, cds.n_codons)
    lw = torch.as_tensor(out["logw"]).reshape(-1)
    live = torch.isfinite(lw)
    ok = [cds.check(list(map(int, s))) for s in seqs[live]]
    print(f"live particles: {int(live.sum())}/{len(lw)}; all satisfy the constraint: {all(ok)}")
    print("log Zhat per run:", [round(float(v), 3) for v in out["logZ"]])
    uniq = {"".join(CODONS[int(c)] for c in s) for s in seqs[live]}
    print(f"distinct sequences: {len(uniq)}")
    for s in sorted(uniq)[:3]:
        print(" ", s, "->", translate([CODONS.index(s[i:i + 3]) for i in range(0, len(s), 3)]))
    assert all(ok) and live.any()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else open("../handoff_ckpt.txt").read().strip()))
