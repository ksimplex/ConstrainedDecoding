"""
Analysis of experiments/twister_fig1.py: pairwise TV to the rejection reference per (method, T), averaged with
equal weight over token-class partitions (TWISTER Sec. 4.3), plus

  split floor     pairwise TV between two random halves of the rejection samples (n_rej / 2 each);
  matched floor   the split floor rescaled to the comparison's sample sizes, using TV noise proportional
                  to sqrt(1/n1 + 1/n2): floor * sqrt((1/n_con + 1/n_rej) / (4 / n_rej));
  Z_hat / h0      mean SMC evidence estimate over the constrained runs vs the rejection acceptance rate
                  (exact identity in expectation; a cheap extra check on this configuration).

usage: python experiments/analysis/twister_fig1_analysis.py twister_fig1.npz twister_fig1.json [out_prefix]
"""
import json, re, sys
import numpy as np
from twistroute.diagnostics import pairwise_tv


def analyse(npz, meta, reps=5, seed=0):
    d = np.load(npz); m = json.load(open(meta))
    A = m["args"]["n_classes"] + 1
    rng = np.random.default_rng(seed)
    keys = [k for k in d.files if k.startswith("con_")]
    seeds = sorted({int(re.search(r"_s(\d+)_", k).group(1)) for k in keys})
    Ts = sorted({int(re.search(r"_T(\d+)", k).group(1)) for k in keys})
    Ks = sorted({int(re.search(r"_K(\d+)", k).group(1)) for k in keys})
    rows = []
    for T in Ts:
        for s in seeds:
            rej = d[f"rej_s{s}_T{T}"]; h0, n_nat = d[f"h0_s{s}_T{T}"]
            fl = []
            for _ in range(reps):
                p = rng.permutation(len(rej)); h = len(rej) // 2
                fl.append(pairwise_tv(rej[p[:h]], rej[p[h:2 * h]], A))
            for K in Ks:
                con = d[f"con_s{s}_T{T}_K{K}"]
                z = np.exp(d[f"logZ_s{s}_T{T}_K{K}"])
                rows.append(dict(T=T, seed=s, K=K, tv=pairwise_tv(con, rej, A), split_floor=float(np.mean(fl)),
                                 matched_floor=float(np.mean(fl) * np.sqrt((1 / len(con) + 1 / len(rej)) / (4 / len(rej)))),
                                 n_rej=len(rej), h0=float(h0), zhat_over_h0=float(z.mean() / h0),
                                 zhat_se_rel=float(z.std() / np.sqrt(len(z)) / h0)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    rows = analyse(sys.argv[1], sys.argv[2])
    df = pd.DataFrame(rows)
    out = sys.argv[3] if len(sys.argv) > 3 else "twister_fig1_summary"
    df.to_csv(out + "_per_partition.csv", index=False)
    g = df.groupby(["T", "K"]).mean(numeric_only=True).reset_index()
    g.to_csv(out + ".csv", index=False)
    pd.set_option("display.width", 200)
    print(g[["T", "K", "tv", "split_floor", "matched_floor", "n_rej", "h0", "zhat_over_h0"]].round(4).to_string(index=False))
