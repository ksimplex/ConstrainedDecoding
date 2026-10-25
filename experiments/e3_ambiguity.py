"""
E3 (CPU part): does twisted SMC stay exact and usable when the structure's ambiguity grows exponentially?

Grammars over the same denoiser, from d = 1 to exponential ambiguity:
    dyck1_unamb  Dyck-1, unambiguous grammar (d(y) = 1)                    A = 2
    dyck1_amb    Dyck-1 with S -> S S (d(y) = Catalan-type in #blocks)       A = 2
    rna_pair     perfect nested pairing, AU/GC/GU (d(y) = #structures)      A = 4

Part 1 (exact, no sampling noise): enumerate every single-reveal trajectory of decode.exact.ToyDenoiser at
L in {4, 6} and report TV(step-exact, Doob), TV(FK, Doob) and exact path ESS.
Part 2 (sweep): models.synthetic.SyntheticMDLM at L in {8, 12, 16, 24, 32}, T = L / 2. Per (grammar, L):
    h0 from native rejection sampling; Z_hat / h0 at K = 1 and K = 16 with resampling on/off; PSIS k-hat;
    single-particle ESS fraction E[w]^2 / E[w^2]; log d(y) over valid SMC outputs (counting semiring);
    pairwise TV of K = 16 samples vs rejection samples, with a split-half floor, where n_rej permits.

usage: python experiments/e3_ambiguity.py [--Ls 8,12,16,24,32] [--runs 4000] [--native 400000] [--out e3_cpu]
"""
import argparse, json, time
import numpy as np, torch
from twistroute import compile as C
from twistroute.decode import native_sample
from twistroute.decode.exact import Decoder, ToyDenoiser, tv
from twistroute.diagnostics import psis_khat, tv_with_floor
from twistroute.experiment import run_smc, z_summary
from twistroute.models import SyntheticMDLM
from twistroute.torch_routes import TorchChart, onehot_log

GRAMMARS = {"dyck1_unamb": (C.DYCK1_UNAMBIGUOUS, [0, 1, 0, 1]),
            "dyck1_amb": (C.DYCK1_AMBIGUOUS, [0, 1, 0, 1]),
            "rna_pair": (C.rna_pair_grammar(), [0, 1, 2, 3])}
ARMS = [("K1", 1, 0.5), ("K16_on", 16, 0.5), ("K16_off", 16, 0.0)]


def exact_cell(L, name):
    """Part 1: exact laws by enumeration for one grammar at length L."""
    g, _ = GRAMMARS[name]
    ins, A, t0 = C.Inside(*g), g[3], time.time()
    laws = Decoder(ins, ToyDenoiser(L, A, eps=1.0, seed=1), L, A).exact_laws()
    ds = [C.count_parses(g[0], g[1], g[2], y) for y in laws["doob"]]
    return dict(L=L, grammar=name, n_valid=len(laws["doob"]), max_parses=int(max(ds)), mean_parses=float(np.mean(ds)),
                tv_step=tv(laws["step"], laws["doob"]), tv_fk=tv(laws["fk"], laws["doob"]),
                zhat_over_h0=laws["zhat_over_h0"], path_ess=laws["path_ess"], seconds=time.time() - t0)


def sweep_cell(L, name, runs, native, scale):
    """Part 2: rejection reference and SMC arms for one grammar at length L on the synthetic denoiser."""
    T, (g, mp) = L // 2, GRAMMARS[name]
    den = SyntheticMDLM(L, scale=scale)
    cl = den.classes(4).merge(mp); A = cl.A
    route = TorchChart(C.Inside(*g))
    t0, gen = time.time(), torch.Generator().manual_seed(1000 + L)
    rej, hits = [], 0
    for i in range(0, native, 5000):
        S = native_sample(den, cl, L, T, min(5000, native - i), gen)
        acc = route.accepts(S); hits += int(acc.sum()); rej.append(S[acc])
    rej, h0 = torch.cat(rej), hits / native
    r = dict(L=L, T=T, grammar=name, h0=h0, n_rej=len(rej), s_native=time.time() - t0)
    if hits == 0:
        r["note"] = "h0 below 1/native; SMC metrics only"
    for tag, K, ess in ARMS:
        n_runs = runs if K == 1 else runs // 4
        out = run_smc(den, cl, route, L, T, n_runs, K, gen, chunk=n_runs * K, ess_min=ess,
                      select_seed=(lambda j, i: L) if K == 16 else None)
        lz = out["logZ"]; z = np.exp(lz)
        zs = z_summary(lz, h0) if h0 > 0 else dict(ratio=None, se_rel=None)
        r.update({f"{tag}_ratio": zs["ratio"], f"{tag}_se_rel": zs["se_rel"], f"{tag}_khat": float(psis_khat(lz)),
                  f"{tag}_ess_frac": float(z.mean() ** 2 / np.mean(z ** 2)),
                  f"{tag}_resamples": float(out["n_res"].mean()), f"{tag}_s": out["seconds"]})
        if K == 16:
            S = torch.tensor(out["samples"])
            logd = route.logZ(onehot_log(S, A)).numpy()                     # log #derivations of each output
            r[f"{tag}_logd_mean"], r[f"{tag}_logd_max"] = float(logd.mean()), float(logd.max())
            if len(rej) >= len(S) + 400 and L <= 16:
                r[f"{tag}_ptv"], r[f"{tag}_ptv_floor"] = tv_with_floor(S.numpy(), rej.numpy(), A, n=len(S))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="8,12,16,24,32")
    ap.add_argument("--exact-Ls", default="4,6")
    ap.add_argument("--runs", type=int, default=4000)
    ap.add_argument("--native", type=int, default=400_000)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--out", default="e3_cpu")
    a = ap.parse_args(argv)
    res = dict(args=vars(a), exact=[], sweep=[])

    def show(kind, r):
        print(kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        json.dump(res, open(a.out + ".json", "w"), indent=1)

    for L in map(int, a.exact_Ls.split(",")):
        for name in GRAMMARS:
            res["exact"].append(exact_cell(L, name)); show("exact", res["exact"][-1])
    for L in map(int, a.Ls.split(",")):
        for name in GRAMMARS:
            res["sweep"].append(sweep_cell(L, name, a.runs, a.native, a.scale)); show("sweep", res["sweep"][-1])


if __name__ == "__main__":
    main()
