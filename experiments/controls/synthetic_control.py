"""
CPU control for the E2 shortfall: the identical decoder on a synthetic denoiser where h0 is cheap to pin down.

    python experiments/controls/synthetic_control.py --scales 1.5,4 --runs 20000
    python experiments/controls/synthetic_control.py --scales 8,16 --gumbel float32,float64 --Ks 1,16

Separates (i) a heavy-tail sampling shortfall (vanishes with more runs), (ii) a resampling bug (K=1 unbiased,
K>1 biased), (iii) a shared-pipeline bias (all K biased), and (iv) float32 Gumbel resolution (float32 vs
float64 differ). Results of the 2026-10-14 runs: all cells within noise of h0 (control_*.json).
"""
import argparse, json, math
import numpy as np, torch
import twistroute.torch_routes.routes as routes
from twistroute.decode import twisted_smc, native_sample
from twistroute.models import SyntheticMDLM
from twistroute.torch_routes import build_route

ap = argparse.ArgumentParser()
ap.add_argument("--scales", default="1.5,4")
ap.add_argument("--Ks", default="1,4,16")
ap.add_argument("--runs", type=int, default=20000)
ap.add_argument("--gumbel", default="float32")
ap.add_argument("--out", default="synthetic_control.json")
a = ap.parse_args()
N, T = 8, 4
FAMS = {"copy": dict(kind="copy", A=2, map=[0, 0, 1, 1], pairs=4, N=N),
        "counting": dict(kind="exactly_k", A=2, map=[0, 1, 0, 1], cls=0, count=2, N=N),
        "dyck1_amb": dict(kind="cfg", grammar="dyck1_amb", A=2, map=[0, 1, 0, 1], N=N)}
out = []
for sc in map(float, a.scales.split(",")):
    for fam, spec in FAMS.items():
        den = SyntheticMDLM(N, scale=sc); cl = den.classes(4).merge(spec["map"]); route = build_route(spec)
        g = torch.Generator().manual_seed(77); hits = 0; n0 = 1_000_000
        for _ in range(n0 // 20000):
            hits += int(route.accepts(native_sample(den, cl, N, T, 20000, g)).sum())
        h0, h0se = hits / n0, math.sqrt(hits) / n0
        for dt in a.gumbel.split(","):
            routes.GUMBEL_DTYPE = getattr(torch, dt)
            for K in map(int, a.Ks.split(",")):
                gen = torch.Generator().manual_seed(100 + K); per = max(1, 40000 // K); lz = []
                for i in range(0, a.runs, per):
                    lz.append(twisted_smc(den, cl, route, N, T, min(per, a.runs - i), K, gen)["logZ"])
                lz = np.concatenate(lz); z = np.exp(lz); se = z.std(ddof=1) / math.sqrt(len(z))
                r = dict(scale=sc, family=fam, gumbel=dt, K=K, h0=h0, ratio=z.mean() / h0,
                         z=(z.mean() - h0) / math.sqrt(se ** 2 + h0se ** 2), sd_logZ=float(lz.std()))
                print(r, flush=True); out.append(r)
json.dump(out, open(a.out, "w"), indent=1)
