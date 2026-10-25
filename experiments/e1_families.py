"""
E1: trajectory bias across constraint families on OWT-130M (claims C1, C3 of experiment_plan.md).

For each family (regular / counting / copy / Dyck-1 unambiguous / Dyck-1 ambiguous / RNA pairing; specs from
configs/e2_cells.json at N = 32), compare decoders against a rejection reference from native samples:
    step_exact    K = 1, weights ignored (Dang & Ermon 2026)
    K4_def        K = 4,  adaptive resampling (ESS < K/2), defensive mixing eps = 0.25
    K16_def       K = 16, same
    K16_plain     K = 16, multinomial resampling (eps = 0), the TWISTER default
    K16_off       K = 16, no resampling
Native kernel as in E2: T = 16 steps, N/T uniform reveals, temperature 1. One output per run is drawn from the
run's final weighted particles. Saves class sequences per decoder and per-run log Z_hat; TV and Z_hat analysis
is offline (experiments/analysis/e1_analysis.py).

Rejection reference: the E2 native pool (native_N32.npz, 200k samples, T = 16, semantic classes) plus, for
families whose pool has fewer than --min-ref accepted samples, extra native samples drawn here.

usage: python experiments/e1_families.py --pool native_N32.npz --families regular,counting,copy
"""
import argparse, json, time
import numpy as np, torch
from twistroute.decode import native_sample
from twistroute.experiment import load_denoiser, run_smc, z_summary
from twistroute.torch_routes import build_route

METHODS = {"step_exact": dict(K=1, ess_min=0.0, defensive=0.0),
           "K4_def": dict(K=4, ess_min=0.5, defensive=0.25),
           "K16_def": dict(K=16, ess_min=0.5, defensive=0.25),
           "K16_plain": dict(K=16, ess_min=0.5, defensive=0.0),
           "K16_off": dict(K=16, ess_min=0.0, defensive=0.0)}


def family_specs(cells_path, N):
    """First spec per family at length N from an E2 cell list (K and R dropped)."""
    specs = {}
    for c in json.load(open(cells_path)):
        if c["N"] == N and c["family"] not in specs:
            specs[c["family"]] = {k: v for k, v in c.items() if k not in ("K", "R")}
    return specs


def accepts_chunked(route, cls, dev, chunk=8192):
    """Membership of every row of cls (CPU tensor), in chunks so chart routes fit in GPU memory."""
    return torch.cat([route.accepts(cls[i:i + chunk].to(dev)).cpu() for i in range(0, len(cls), chunk)])


def language_key(spec):
    """Families with the same class map and language share native top-up samples (Dyck-1 unamb / amb)."""
    return json.dumps(spec["map"]) + spec.get("grammar", spec["kind"]).replace("_amb", "").replace("_unamb", "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="experiments/configs/e2_cells.json")
    ap.add_argument("--pool", default="native_N32.npz")
    ap.add_argument("--families", default="regular,counting,copy,dyck1_unamb,dyck1_amb,rna_pair")
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--runs", default="step_exact:10000,K4_def:8000,K16_def:4000,K16_plain:4000,K16_off:4000")
    ap.add_argument("--min-ref", type=int, default=2500)
    ap.add_argument("--max-topup", type=int, default=600_000)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="e1_families")
    ap.add_argument("--synthetic", action="store_true", help="CPU dry run with the synthetic denoiser")
    a = ap.parse_args(argv)
    runs = {k: int(v) for k, v in (x.split(":") for x in a.runs.split(","))}

    gen_dev = "cpu" if a.synthetic else "cuda"
    gen = torch.Generator(device=gen_dev).manual_seed(4711)
    M, dev = load_denoiser(a.N, a.synthetic)
    if a.synthetic:
        base = M.classes(4)
        pool_tok = torch.cat([native_sample(M, base, a.N, a.T, a.chunk, gen, return_tokens=True)[1].cpu()
                              for _ in range(20)])                      # small stand-in pool
    else:
        from twistroute.models import semantic_classes
        base = semantic_classes(M.tok, M.V, M.mask_id).to(dev)
        pool_tok = torch.from_numpy(np.load(a.pool)["tokens"].astype(np.int64))
    specs = family_specs(a.cells, a.N)

    arrays, meta, topup = {}, dict(args=vars(a), methods=METHODS, runs=runs, timing={}, ref={}), {}
    for fam in a.families.split(","):
        spec = specs[fam]
        route, cl = build_route(spec, dev), base.merge(spec["map"])
        # ---- rejection reference: pool, plus native top-up when it has fewer than min_ref accepted samples
        cls_pool = cl.token_class.cpu()[pool_tok]
        acc = accepts_chunked(route, cls_pool, dev)
        ref, n_nat, hits = [cls_pool[acc]], len(acc), int(acc.sum())
        if hits < a.min_ref:
            key = language_key(spec)
            if key not in topup:
                t0, extra = time.time(), []
                need = int(min(a.max_topup, (a.min_ref - hits) / max(hits / n_nat, 1e-6) * 1.1))
                for i in range(0, need, a.chunk):
                    extra.append(native_sample(M, base, a.N, a.T, min(a.chunk, need - i), gen,
                                               return_tokens=True)[1].cpu())
                topup[key] = torch.cat(extra)
                meta["timing"][f"topup_{fam}"] = time.time() - t0
            cls_ext = cl.token_class.cpu()[topup[key]]
            acc_e = accepts_chunked(route, cls_ext, dev)
            ref.append(cls_ext[acc_e]); n_nat += len(acc_e); hits += int(acc_e.sum())
        h0 = hits / n_nat
        arrays[f"ref_{fam}"] = torch.cat(ref).numpy().astype(np.uint8)
        arrays[f"h0_{fam}"] = np.array([h0, n_nat])
        meta["ref"][fam] = dict(h0=h0, n_native=n_nat, n_ref=hits)
        print(fam, "h0", round(h0, 5), "n_ref", hits, "n_native", n_nat, flush=True)
        # ---- decoders: one output per run, drawn from the run's final weighted particles
        for m, R in runs.items():
            p = dict(METHODS[m]); K = p.pop("K")
            r = run_smc(M, cl, route, a.N, a.T, R, K, gen, a.chunk, select_seed=lambda j, i: 10_000 * j + K, **p)
            assert not r["any_finite_invalid"], f"{fam} {m}: finite-weight invalid particle"
            arrays[f"con_{fam}_{m}"] = r["samples"].astype(np.uint8)
            arrays[f"logZ_{fam}_{m}"] = r["logZ"]
            arrays[f"nres_{fam}_{m}"] = r["n_res"]
            meta["timing"][f"{fam}_{m}"] = r["seconds"]
            zs = z_summary(r["logZ"], h0)
            print(" ", m, "Z/h0", round(zs["ratio"], 4), "se_rel", round(zs["se_rel"], 4), "s", round(r["seconds"]),
                  flush=True)
            np.savez_compressed(a.out + ".npz", **arrays)
            json.dump(meta, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
