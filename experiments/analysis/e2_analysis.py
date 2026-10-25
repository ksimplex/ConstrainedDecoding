"""
E2 analysis: Z_hat / h0 per cell with bootstrap CIs (including h0 uncertainty), PSIS tail shape k, subsample
medians (rising = heavy tail, flat = constant bias), and, for two-sided results, the native-side identity
E_p[1_C / w], the tilt comparison and the Meng & Wong (Statistica Sinica 6:831, 1996) optimal bridge estimate.

usage: python experiments/analysis/e2_analysis.py e2_results.json e2_h0.json [twosided_results.json]
"""
import json, math, sys
import numpy as np, pandas as pd
from twistroute.diagnostics import psis_khat, ratio_ci, subsample_medians


def bridge(lq, lw_nC, iters=200):
    """Optimal bridge estimate of h0 from q-samples and p(.|C)-samples of log w (w = p 1_C / q)."""
    n1, n2 = len(lq), len(lw_nC); s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    wq, wp = np.exp(lq), np.exp(lw_nC); r = np.mean(wq)
    for _ in range(iters):
        r = np.mean(wq / (s1 * wq + s2 * r)) / np.mean(1.0 / (s1 * wp + s2 * r))
    return r


def smc_table(res, H0, n0=200_000):
    rows = []
    for c in res["cells"]:
        s = c["spec"]; lz = np.array(c["logZ"]); z = np.exp(lz); h = H0[f'{s["family"]}|{s["N"]}']["h0"]
        lo, hi = ratio_ci(z, h, n0)
        rows.append(dict(family=s["family"], N=s["N"], K=s["K"], ess_min=s.get("ess_min", 0.5), h0=h,
                         ratio=z.mean() / h, ci_lo=lo, ci_hi=hi, khat=psis_khat(lz),
                         **{f"med_n{n}": v / h for n, v in subsample_medians(z, [62, 250, 1000]).items()},
                         resamples=float(np.mean(c["n_res"])), frac_valid=c["frac_valid"]))
    return pd.DataFrame(rows)


def twosided_table(res, H0):
    rows = []
    for r in res:
        s = r["spec"]; lw = np.array(r["native_logw"]); v = np.array(r["native_valid"]).astype(bool)
        lq = np.array(r["q_logw"]); h = H0[f'{s["family"]}|{s["N"]}']["h0"]
        inv = np.where(v, np.exp(-np.where(v, lw, 0.0)), 0.0)
        w = np.exp(lq - lq.max()); o = np.argsort(lq); cw = np.cumsum(w[o]) / w.sum()
        rows.append(dict(family=s["family"], N=s["N"], h0_pool=h, K1_ratio=np.exp(lq).mean() / h,
                         K1_se=np.exp(lq).std() / math.sqrt(len(lq)) / h, I2=inv.mean(), I2_se=inv.std() / math.sqrt(len(inv)),
                         med_logw_native=np.median(lw[v]), med_logw_tilted_q=lq[o][np.searchsorted(cw, 0.5)],
                         frac_native_above_qmax=float((lw[v] > lq.max()).mean()), bridge_over_pool=bridge(lq, lw[v]) / h,
                         n_valid_native=int(v.sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    H0 = json.load(open(sys.argv[2]))
    print(smc_table(json.load(open(sys.argv[1])), H0).round(4).to_string(index=False))
    if len(sys.argv) > 3:
        print(twosided_table(json.load(open(sys.argv[3])), H0).round(4).to_string(index=False))
