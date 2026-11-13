"""
Is the model a better predictor of codon choice than a frequency table, measured by a proper scoring rule?

The top-1 test said no: 39.8% against 38.7% for a lookup table, McNemar z = +0.53. But top-1 accuracy is a
poor instrument here. A natural coding sequence is ONE draw from a high-entropy conditional; even a perfect
model of that conditional would often rank the observed codon second. Near-baseline accuracy is therefore
weak evidence of a weak model, which is also why the field reports distributional agreement instead (see
docs/codon_benchmark_landscape.md).

This scores the log-probability assigned to the ACTUALLY OBSERVED codon, which is a proper scoring rule: it
is optimised uniquely by the true conditional, and it rewards a model for putting mass in the right places
even when its argmax is wrong. Two comparisons on identical positions:

    model       p(codon | all other codons revealed, protein fixed), renormalised over the synonymous set
    frequency   p(codon) from the corpus codon table, renormalised over the same synonymous set

Renormalising both over the synonymous set is what makes the comparison fair: the protein is given in the
design setting, so neither predictor should be credited for knowing which amino acid comes next. One
position is masked at a time with every other position revealed -- the most context the model can get, and
so the setting most favourable to it. If it cannot beat a context-free table here, it will not elsewhere.

What this is NOT: a joint likelihood. Masked diffusion models give an ELBO, not an exact joint, and summing
per-position conditionals is a pseudo-likelihood. That is the right object anyway for the design question,
which is per-position codon choice given context, not the probability of a whole transcript.

    python experiments/codon_scoring.py <weights> <cds.json> <codon_table.json> <out.json> [n_genes] [n_pos]
"""
import json
import math
import random
import sys

import numpy as np
import torch

from twistroute_bio.alphabet import CODONS, GENETIC_CODE, translate
from twistroute_bio.models.mrnautilus import MRNAutilus, CODON_TO_TOKEN

SYN = {}
for _c in CODONS:
    SYN.setdefault(GENETIC_CODE[_c], []).append(_c)


def score_gene(model, cds, table, positions, device="cpu", batch=8):
    """Per-position log p(observed codon) under the model and under the frequency table.

    Both are renormalised over the synonymous set of the observed amino acid. One row per scored position:
    that position masked, every other position revealed.
    """
    codons = [cds[i:i + 3] for i in range(0, len(cds), 3)]
    win = model.cds_window(len(codons))
    base = torch.tensor([CODON_TO_TOKEN[CODONS.index(c)] for c in codons],
                        dtype=torch.long, device=device)
    keep_pos = [p for p in positions if len(SYN[GENETIC_CODE[codons[p]]]) >= 2]   # Met/Trp have no choice
    out = []
    for i in range(0, len(keep_pos), batch):
        chunk = keep_pos[i:i + batch]
        x = base.unsqueeze(0).repeat(len(chunk), 1)
        for r, p in enumerate(chunk):
            x[r, p] = win.mask_id
        with torch.no_grad():
            lg = win.logits(x)
        for r, p in enumerate(chunk):
            aa = GENETIC_CODE[codons[p]]
            syn = SYN[aa]
            keep = torch.tensor([CODON_TO_TOKEN[CODONS.index(s)] for s in syn], device=device)
            m = torch.log_softmax(lg[r, p, keep].double(), dim=0).cpu()
            f = np.array([table["abs"][s] for s in syn], dtype=float)
            f = np.log(f / f.sum())
            j = syn.index(codons[p])
            out.append(dict(pos=p, aa=aa, n_syn=len(syn), codon=codons[p],
                            logp_model=float(m[j]), logp_freq=float(f[j]),
                            top1_model=int(int(torch.argmax(m)) == j),
                            top1_freq=int(int(np.argmax(f)) == j)))
    return out


def main(weights, cds_path, table_path, out_path, n_genes=6, n_pos=40):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MRNAutilus(weights, device=dev, dtype=torch.float32, impl="torch")
    table = json.load(open(table_path))
    genes = json.load(open(cds_path))[:int(n_genes)]
    rows = []
    for h in genes:
        n_cod = len(h["cds"]) // 3
        rp = random.Random(hash(h["gene"]) % 10 ** 6)
        pos = sorted(rp.sample(range(n_cod - 1), min(int(n_pos), n_cod - 1)))
        r = score_gene(model, h["cds"], table, pos, device=dev)
        for x in r:
            x["gene"] = h["gene"]
        rows += r
        d = np.mean([x["logp_model"] - x["logp_freq"] for x in r])
        print(f"{h['gene']:<9} n={len(r):<4} mean log-prob gain over frequency {d:+.4f}", flush=True)
        json.dump(rows, open(out_path, "w"))
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:5], *(sys.argv[5:7] or []))
