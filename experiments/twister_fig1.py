"""
Reproduce TWISTER Figure 1(a) (Thimmaiah et al., arXiv:2609.35609): trajectory bias of step-exact decoding
vs TWISTER (K = 4, 8) on OWT-130M for the constraint "a.*", following the Sec. 4.3 protocol:

  * token classes: rank tokens by probability under a fully masked input (averaged over positions), split the
    top `pool` tokens equally into classes a, b, c, d and map the rest to "x"; average over random partitions;
  * T in {1, 2, 4, 8, 16} denoising steps, random remasking, temperature 1;
  * reference: rejection sampling from 40k native samples (Eq. 7); 10k constrained samples per method;
  * metric: pairwise TV between class sequences, with a split-half noise floor of the rejection samples.

Unstated in the paper, chosen here (see README): sequence length N (default 32), pool size (default 8, giving
an acceptance rate of ~3%, consistent with the paper's ~0.09 floor), and the exact pairwise-TV averaging
(mean over all position pairs).

Outputs an .npz of class sequences per (partition seed, T, method) and per-run log Z_hat; analysis in
experiments/analysis/twister_fig1_analysis.py.
"""
import argparse, json, time
import numpy as np, torch
from twistroute.decode import native_sample
from twistroute.experiment import load_denoiser, masked_marginal, run_smc
from twistroute.models import ranked_classes
from twistroute.torch_routes import build_route


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--n-classes", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--Ts", default="1,2,4,8,16")
    ap.add_argument("--Ks", default="1,4,8")
    ap.add_argument("--n-native", type=int, default=40000)
    ap.add_argument("--n-con", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="twister_fig1")
    ap.add_argument("--synthetic", action="store_true", help="CPU dry run with models.synthetic.SyntheticMDLM")
    a = ap.parse_args(argv)
    N = a.N
    torch.manual_seed(0)
    gen = torch.Generator(device="cpu" if a.synthetic else "cuda").manual_seed(2026)
    M, dev = load_denoiser(N, a.synthetic)
    p_masked = masked_marginal(M, N, dev)
    seeds = [int(s) for s in a.seeds.split(",")]
    CL = {s: ranked_classes(p_masked, a.n_classes, a.pool, M.mask_id, seed=s).to(dev) for s in seeds}
    route = build_route(dict(kind="prefix", cls=0, A=a.n_classes + 1, N=N), dev)
    arrays = {}
    meta = dict(args=vars(a), info=M.info(), timing={},
                class_tokens={s: [M.tok.decode([int(v)]) for v in torch.nonzero(CL[s].token_class == 0).flatten()]
                              for s in seeds})
    for T in [int(t) for t in a.Ts.split(",")]:
        t0 = time.time()
        toks = torch.cat([native_sample(M, CL[seeds[0]], N, T, min(a.chunk, a.n_native - i), gen, return_tokens=True)[1]
                          for i in range(0, a.n_native, a.chunk)])        # native samples do not depend on classes
        meta["timing"][f"native_T{T}"] = time.time() - t0
        for s in seeds:
            cls = CL[s].token_class[toks]
            acc = route.accepts(cls)
            arrays[f"rej_s{s}_T{T}"] = cls[acc].cpu().numpy().astype(np.uint8)
            arrays[f"h0_s{s}_T{T}"] = np.array([acc.float().mean().item(), len(acc)])
            for K in [int(k) for k in a.Ks.split(",")]:
                r = run_smc(M, CL[s], route, N, T, a.n_con, K, gen, a.chunk,
                            select_seed=lambda j, i: i + 1000 * K + T)
                arrays[f"con_s{s}_T{T}_K{K}"] = r["samples"].astype(np.uint8)
                arrays[f"logZ_s{s}_T{T}_K{K}"] = r["logZ"]
                meta["timing"][f"s{s}_T{T}_K{K}"] = r["seconds"]
            print(T, s, "h0", arrays[f"h0_s{s}_T{T}"][0], "n_rej", len(arrays[f"rej_s{s}_T{T}"]),
                  {k: round(v) for k, v in meta["timing"].items() if f"T{T}" in k}, flush=True)
        np.savez_compressed(a.out + ".npz", **arrays)
        json.dump(meta, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
