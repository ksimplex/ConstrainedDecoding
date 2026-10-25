"""
E2 two-sided identity check (decode/native.py): per cell, `n` native paths with the K=1 importance weight
computed along each, and `n` K=1 SMC runs. For correct code E_q[w] = h0 and E_p[1_C / w] = 1, and the law of
log w under p(. | C) equals the q-law tilted by w / h0. Analysis: experiments/analysis/e2_analysis.py.

usage: python experiments/e2_twosided.py configs/twosided_cells.json twosided_results.json
"""
import json, sys, time
import numpy as np, torch
from twistroute.decode import twisted_smc, native_path_weights
from twistroute.models import semantic_classes
from twistroute.models.mdlm_owt import MDLM
from twistroute.torch_routes import build_route

T, B, dev = 16, 512, "cuda"


def main():
    cells, out = json.load(open(sys.argv[1])), sys.argv[2]
    M = MDLM(dev); base = semantic_classes(M.tok, M.V, M.mask_id).to(dev)
    gen = torch.Generator(device=dev).manual_seed(4321)
    res = []
    for spec in cells:
        route, cl, n, N = build_route(spec, dev), base.merge(spec["map"]), spec["n"], spec["N"]
        t0 = time.time(); lwn, val, lzq = [], [], []
        for i in range(0, n, B):
            a, v = native_path_weights(M, cl, route, N, T, min(B, n - i), gen); lwn.append(a); val.append(v)
        t1 = time.time()
        for i in range(0, n, B):
            lzq.append(twisted_smc(M, cl, route, N, T, min(B, n - i), 1, gen)["logZ"])
        res.append(dict(spec=spec, native_logw=np.concatenate(lwn).tolist(),
                        native_valid=np.concatenate(val).astype(int).tolist(),
                        q_logw=np.concatenate(lzq).tolist(), s_native=t1 - t0, s_q=time.time() - t1))
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
