"""
E2: Z_hat unbiasedness of twisted SMC on OWT-130M across constraint routes.

For each cell (family, N, K, R[, ess_min]) run R independent SMC runs of K particles under the native kernel
(T = 16 steps, N/T uniform reveals per step). E[Z_hat] = h0 = P_native(C) exactly at every K (TWISTER Thm 4.2;
decode/smc.py); h0 comes from experiments/e2_native.py. Classes: the 4 linguistic classes of
models.classes.semantic_classes, merged by the cell's "map".

usage: python experiments/e2_smc.py configs/e2_cells.json e2_results.json
"""
import json, math, sys, time
import numpy as np, torch
from twistroute.decode import twisted_smc
from twistroute.models import semantic_classes
from twistroute.models.mdlm_owt import MDLM
from twistroute.torch_routes import build_route

T, CHUNK, dev = 16, 512, "cuda"


def main():
    cells, out = json.load(open(sys.argv[1])), sys.argv[2]
    t0 = time.time(); M = MDLM(dev)
    base = semantic_classes(M.tok, M.V, M.mask_id).to(dev)
    res = dict(load_s=time.time() - t0, cells=[])
    gen = torch.Generator(device=dev).manual_seed(1234)
    for spec in cells:
        route, cl = build_route(spec, dev), base.merge(spec["map"])
        K, R, ess_min = spec["K"], spec["R"], spec.get("ess_min", 0.5)
        per, ts, parts, done = max(1, CHUNK // K), time.time(), [], 0
        while done < R:
            n = min(per, R - done)
            parts.append(twisted_smc(M, cl, route, spec["N"], T, n, K, gen, ess_min=ess_min)); done += n
        lz = np.concatenate([p["logZ"] for p in parts])
        rec = dict(spec=spec, seconds=time.time() - ts, logZ=lz.tolist(),
                   ess_min=np.concatenate([p["ess_min"] for p in parts]).tolist(),
                   n_res=np.concatenate([p["n_res"] for p in parts]).tolist(),
                   frac_valid=float(np.mean([p["frac_valid_finite"] for p in parts])),
                   any_finite_invalid=any(p["any_finite_invalid"] for p in parts), eta0=parts[0]["eta0"])
        z = np.exp(lz)
        print(spec["family"], spec["N"], K, R, ess_min, "meanZ", z.mean(), "se", z.std() / math.sqrt(len(z)),
              "valid", rec["frac_valid"], "s", round(rec["seconds"], 1), flush=True)
        res["cells"].append(rec)
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
