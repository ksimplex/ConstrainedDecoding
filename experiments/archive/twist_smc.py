"""Does a better twist fix the weights? One token per step in uniformly random order (the base sampler), with the
twist computed under different stand-ins for the model (see twist_error.py):

  product       independent positions (the twist in use)
  chain_ratio   first-order chain from the current marginals and the stationary adjacent-token kernel; no extra
                denoiser call. Mixed with a share eps of the product twist so it is positive wherever that is.
  chain_exact   first-order chain with the model's own adjacent-pair conditionals (oracle only; a ceiling)

Any positive twist with the exact indicator at the end leaves Z_hat unbiased; the weight of a step is
p(a | revealed) psi(new) / (q(a) psi(old)) with q(a) proportional to the stand-in's P(C, x_i = a | revealed).
Reported: Z_hat / p(C), its spread over runs (sd / mean), the marginal error, and with --resample the same under
plain multinomial resampling at ESS < K / 2.
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "studies"))
import compare as cp
import twist_error as te


class Standin:
    def __init__(self, tw, kind, K=None, eps=0.05):
        self.tw, self.kind, self.K, self.eps = tw, kind, K, eps

    def prepare(self, x):
        mu, rev = self.tw._marg(x)
        Qc = None
        if self.kind == "chain_ratio":
            Qc = self.tw.ratio_from(mu, rev, self.K)
        elif self.kind == "chain_exact":
            Qc = self.tw.pair_conditionals(x)
        return mu, Qc

    def value(self, prep, clamp=None):
        mu, Qc = prep
        lp = self.tw.product_from(mu, clamp)
        if self.kind == "product":
            return lp
        lc = self.tw.chain_from(mu, Qc, clamp)
        return torch.logaddexp(lc + math.log1p(-self.eps), lp + math.log(self.eps))


def smc(den, tw, standin, nrun, K, gen, resample=False):
    N, S, P = den.N, den.S, nrun * K
    x = torch.full((P, N), den.mask_id); rows = torch.arange(P)
    prep = standin.prepare(x); logpsi = standin.value(prep)
    logw = torch.zeros(P, dtype=torch.float64); logZ = logpsi.view(nrun, K)[:, 0].clone()
    n_res = torch.zeros(nrun); trace = []
    for t in range(N):
        pos = torch.multinomial((x == den.mask_id).double(), 1, generator=gen).squeeze(-1)
        Psi = torch.stack([standin.value(prep, (pos, torch.full((P,), a))) for a in range(S)], -1)
        dead = torch.isneginf(Psi).all(-1)
        q = torch.softmax(torch.where(dead[:, None], torch.zeros_like(Psi), Psi), -1)
        a = torch.multinomial(q, 1, generator=gen).squeeze(-1)
        lmu = torch.log(prep[0][rows, pos, a])
        x[rows, pos] = a
        prep = standin.prepare(x); new = standin.value(prep)
        logw = torch.where(dead, torch.full_like(logw, -math.inf), logw + lmu + new - torch.log(q[rows, a]) - logpsi)
        logpsi = new
        trace.append(logw.clone())
        if resample and t < N - 1:
            lw = logw.view(nrun, K); wn = torch.nan_to_num(torch.softmax(lw, -1), nan=1.0 / K)
            fire = 1.0 / (wn ** 2).sum(-1) < 0.5 * K
            if fire.any():
                logZ = logZ + torch.where(fire, torch.logsumexp(lw, -1) - math.log(K), torch.zeros(nrun, dtype=torch.float64))
                idx = torch.multinomial(wn, K, replacement=True, generator=gen)
                idx = torch.where(fire[:, None], idx, torch.arange(K)[None]); flat = (idx + K * torch.arange(nrun)[:, None]).reshape(-1)
                x, logpsi = x[flat], logpsi[flat]; prep = tuple(None if p is None else p[flat] for p in prep)
                logw = torch.where(fire[:, None], torch.zeros(nrun, K, dtype=torch.float64), lw[torch.arange(nrun)[:, None], idx]).reshape(-1)
                n_res += fire.float()
    lw = logw.view(nrun, K)
    logZ = logZ + torch.logsumexp(lw, -1) - math.log(K)
    return dict(logZ=logZ.numpy(), logw=lw.numpy(), x=x.view(nrun, K, N).numpy(), n_res=float(n_res.mean()), trace=torch.stack(trace).numpy())


def summarise(out, truth, S):
    z = np.exp(out["logZ"] - truth.log_pC); nrun = len(z)
    lw = out["logw"]; w = np.exp(lw - lw.max(1, keepdims=True)); w = np.nan_to_num(w / w.sum(1, keepdims=True))
    rw = z[:, None] * w; rw = rw / rw.sum()
    mg = np.einsum("rk,rkna->na", rw, np.eye(S)[out["x"]])
    return dict(z_ratio=float(z.mean()), z_se=float(z.std(ddof=1) / math.sqrt(nrun)), spread=float(z.std(ddof=1) / z.mean()),
                marg_tv=float(0.5 * np.abs(mg - truth.marginals).sum(-1).mean()), n_res=out["n_res"])


def rho(trace, fracs=(0.1, 0.25, 0.5, 0.75)):
    T = trace.shape[0]; fin = trace[-1]; ok = np.isfinite(fin); wf2 = np.exp(2 * (fin[ok] - fin[ok].max())); o = {}
    for f in fracs:
        now = trace[max(int(round(f * T)) - 1, 0)][ok]; m = np.isfinite(now); wn = np.exp(now[m] - now[m].max())
        o[str(f)] = float((wf2[m] / wn).mean() * wn.mean() / wf2.mean())
    return o


def run_cell(job):
    mname, cname, N, a = job
    torch.set_num_threads(1)
    S = cp.n_symbols(cname)
    den = cp.make_model(mname, N, S); d, act, truth = te.dfa_of(cname, den, N)
    tw = te.Twists(den, d, act); Kst = tw.stationary_kernel(seed=a["seed"])
    rows = []
    for kind in a["kinds"]:
        sd = cp.stream_seed(a["seed"], mname, cname, N, kind)
        st = Standin(tw, kind, Kst, a["eps"])
        t0 = time.time()
        off = smc(den, tw, st, a["nrun"], a["K"], torch.Generator().manual_seed(sd))
        on = smc(den, tw, st, a["nrun"], a["K"], torch.Generator().manual_seed(sd + 1), resample=True)
        one = smc(den, tw, st, a["nrun"] * a["K"], 1, torch.Generator().manual_seed(sd + 2))       # single particles: weight trust
        fin = one["trace"][-1][:, None] if one["trace"].ndim == 1 else one["trace"][-1]
        rows.append(dict(model=mname, constraint=cname, N=N, twist=kind, nrun=a["nrun"], K=a["K"], log_pC=truth.log_pC, seconds=round(time.time() - t0, 1),
                         off=summarise(off, truth, S), on=summarise(on, truth, S), rho=rho(one["trace"]),
                         sd_log_w=float(np.std(one["trace"][-1][np.isfinite(one["trace"][-1])]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--nrun", type=int, default=500); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--models", nargs="+", default=["hidden_0.85", "hidden_0.85_n0.15", "hidden_0.85_n0.5"])
    ap.add_argument("--constraints", nargs="+", default=["counting", "window", "two_window", "dyck", "forbidden"])
    ap.add_argument("--kinds", nargs="+", default=["product", "chain_ratio", "chain_exact"])
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--out", default="twist_smc.json")
    a = vars(ap.parse_args())
    jobs = [(m, c, a["N"], a) for m in a["models"] for c in a["constraints"]]
    rows, t0 = [], time.time()
    with Pool(a["workers"]) as pool:
        for res in pool.imap_unordered(run_cell, jobs):
            rows += res
            json.dump(dict(args=a, rows=rows), open(a["out"], "w"), indent=1)
            print(f"[{time.time() - t0:6.0f}s] {res[0]['model']} {res[0]['constraint']}: " + "; ".join(
                f"{r['twist']} Z {r['off']['z_ratio']:.2f} spread off {r['off']['spread']:.2f} on {r['on']['spread']:.2f}" for r in res), flush=True)


if __name__ == "__main__":
    main()
