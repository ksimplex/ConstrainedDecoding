"""
E1 analysis: for each family and decoder, pairwise TV to the rejection reference (mean over position pairs,
twistroute.diagnostics.pairwise_tv), a matched noise floor (a reference subset of the same size as the decoder
sample against the remaining reference, averaged over 5 random splits), and Z_hat / h0 with a bootstrap CI that
includes the uncertainty in h0 (diagnostics.ratio_ci), plus the PSIS tail shape of Z_hat.

usage: python experiments/analysis/e1_analysis.py e1_A.npz,e1_B.npz e1_summary.csv
"""
import sys
import numpy as np, pandas as pd
from twistroute.diagnostics import tv_with_floor, ratio_ci, psis_khat

A_OF = {"regular": 4, "counting": 2, "copy": 2, "dyck1_unamb": 2, "dyck1_amb": 2, "rna_pair": 4}


def analyse(paths, n_floor=5):
    """One row per (family, decoder) across E1 output files."""
    rows = []
    for path in paths:
        Z = np.load(path)
        for fam in sorted(k[4:] for k in Z if k.startswith("ref_")):
            R = Z[f"ref_{fam}"].astype(np.int64)
            h0, n0 = Z[f"h0_{fam}"]
            for m in [k.split(f"con_{fam}_")[1] for k in Z if k.startswith(f"con_{fam}_")]:
                S, lz = Z[f"con_{fam}_{m}"].astype(np.int64), Z[f"logZ_{fam}_{m}"]
                z = np.exp(lz)
                tv, floor = tv_with_floor(S, R, A_OF[fam], n_floor=n_floor)
                lo, hi = np.ravel(ratio_ci(z, h0, int(n0), rng=np.random.default_rng(0)))
                rows.append(dict(family=fam, method=m, n_samples=len(S), n_ref=len(R), h0=h0, tv=tv, tv_floor=floor,
                                 tv_excess=tv - floor, z_ratio=z.mean() / h0, z_lo=lo, z_hi=hi,
                                 khat=float(psis_khat(lz)), mean_resamplings=float(Z[f"nres_{fam}_{m}"].mean())))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = analyse(sys.argv[1].split(","))
    df.to_csv(sys.argv[2], index=False)
    print(df.round(4).to_string(index=False))
