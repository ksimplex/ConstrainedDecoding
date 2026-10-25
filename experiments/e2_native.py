"""E2 step 1: native (unconstrained) pools from OWT-130M under the random-order block-unmasking kernel.
Saves tokens (uint16) and the 4 semantic class codes per N; h0 for any class constraint is a pool average.

usage: python experiments/e2_native.py 32,64 200000
"""
import json, sys, time
import numpy as np, torch
from twistroute.decode import native_sample
from twistroute.models import semantic_classes
from twistroute.models.mdlm_owt import MDLM

NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32, 64]
NPOOL = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
T, B, dev = 16, 512, "cuda"
torch.manual_seed(0)
t0 = time.time(); M = MDLM(dev); cl = semantic_classes(M.tok, M.V, M.mask_id).to(dev)
rep = dict(load_s=time.time() - t0, info=M.info(), class_sizes=cl.sizes(), T=T, B=B, pools={})
for N in NS:
    ts, toks = time.time(), []
    while sum(len(v) for v in toks) < NPOOL:
        toks.append(native_sample(M, cl, N, T, B, return_tokens=True)[1].cpu())
    X = torch.cat(toks)[:NPOOL]
    codes = cl.token_class.cpu()[X].numpy().astype(np.uint8)
    np.savez_compressed(f"native_N{N}.npz", tokens=X.numpy().astype(np.uint16), classes=codes)
    rep["pools"][N] = dict(n=NPOOL, seconds=time.time() - ts, class_freq=np.bincount(codes.ravel(), minlength=4).tolist(),
                           examples=[M.tok.decode(X[i].tolist()) for i in range(5)])
    json.dump(rep, open("e2_native_report.json", "w"), indent=1)
