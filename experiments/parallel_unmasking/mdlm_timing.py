"""Where does the time of one sampler step go on a real masked diffusion language model?

Times, on one GPU, each piece of a step for a batch of particles at a half-revealed state: the denoiser call, the
class projection, the route (evaluation, sampling, constrained marginals), the schedule (kernel sum, a whole
`reveal`), and the chain twist (value, marginals). Two constraints over three token classes: a count on a window of
a tenth of the positions, and a count on the whole sequence. Every time is also given in denoiser calls.

--synthetic runs on the small synthetic denoiser on CPU (a code check, not a result).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from twistroute import compile as C
from twistroute.decode import (CouplingPolicy, ChainTwist, fit_pair_kernel, fit_distance_kernel, COUNT_EDGES,
                               constrained_marginals, native_sample)
from twistroute.decode.policy import RevealState
from twistroute.decode.schedule import clamp_revealed
from twistroute.torch_routes import torch_route, TorchSubset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulib.summary import provenance                                                               # noqa: E402


def timed(fn, dev, reps):
    """Median wall time of fn() in milliseconds, after one warm-up call."""
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    fn(); sync(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(1e3 * (time.perf_counter() - t))
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[128, 256]); ap.add_argument("--batches", type=int, nargs="+", default=[8, 64, 256])
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--traces", type=int, default=8); ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="mdlm_timing.json")
    a = ap.parse_args()
    res = dict(args=vars(a), provenance=provenance(), rows=[])
    den_real = None
    for N in a.N:
        if a.synthetic:
            from twistroute.models import SyntheticMDLM
            dev = "cpu"; den = SyntheticMDLM(N, scale=3.0); base = den.classes(4)
        else:
            from twistroute.models import semantic_classes
            from twistroute.models.mdlm_owt import MDLM
            dev = "cuda"
            den_real = den_real or MDLM(dev); den = den_real
            base = semantic_classes(den.tok, den.V, den.mask_id).to(dev); res["model"] = den.info()
            res["gpu"] = torch.cuda.get_device_name(0)
        cl = base.merge([0, 1, 2, 2]); A = 3
        G = lambda s: torch.Generator(device=dev).manual_seed(a.seed * 1000 + s)
        t0 = time.time(); kern, _ = fit_distance_kernel(den, N, a.traces, G(1), device=dev, count_edges=COUNT_EDGES); t_kern = time.time() - t0
        t0 = time.time(); K = fit_pair_kernel(den, cl, a.pairs, G(2), N=N); t_pair = time.time() - t0
        w = max(8, N // 10); lo = N // 2 - w // 2; pos = list(range(lo, lo + w)); act = np.zeros(N, dtype=bool); act[pos] = True
        d_win, d_all = C.dfa_at_least_k(2, max(1, w // 3), A), C.dfa_at_least_k(2, max(1, N // 6), A)
        cons = dict(window=(TorchSubset(torch_route(d_win, dev), pos, N, device=dev), ChainTwist(d_win, K, active=act, device=dev)),
                    whole_sequence=(torch_route(d_all, dev), ChainTwist(d_all, K, device=dev)))
        for Bsz in a.batches:
            xs = native_sample(den, cl, N, 16, Bsz, G(3))                              # a plausible sequence per particle
            x = torch.where(torch.rand(xs.shape, generator=G(4), device=dev) < 0.5, xs, torch.full_like(xs, den.mask_id))
            masked = x == den.mask_id
            ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
            lp = den.logits(x); lw = cl.class_logw(lp); lwc = clamp_revealed(lw, x, ycls, den.mask_id)
            t_den = timed(lambda: den.logits(x), dev, a.reps)
            base_row = dict(N=N, batch=Bsz, denoiser_ms=t_den, denoiser_ms_per_particle=t_den / Bsz, pilot_kernel_s=t_kern, pilot_pairs_s=t_pair,
                            class_projection_ms=timed(lambda: cl.class_logw(lp), dev, a.reps),
                            kernel_sum_ms=timed(lambda: kern.apply(masked, masked.double()), dev, a.reps))
            for cname, (route, tw) in cons.items():
                st = RevealState(x, ycls, lw, den.mask_id, cl, route)
                pol = lambda **kw: CouplingPolicy(kern, 0.0, priority="lex", mixed=False, **kw)
                row = dict(base_row, constraint=cname, automaton_states=int(tw.Q),
                           route_logZ_ms=timed(lambda: route.logZ(lwc), dev, a.reps),
                           route_sample_ms=timed(lambda: route.sample(lwc, G(5)), dev, a.reps),
                           route_marginals_ms=timed(lambda: constrained_marginals(route, lwc), dev, a.reps),
                           reveal_sequential_ms=timed(lambda: pol(max_set=1).reveal(st, 0, N, G(6)), dev, a.reps),
                           reveal_parallel_ms=timed(lambda: pol(rate=1.0 / N, pull_forced=8).reveal(st, 0, N, G(6)), dev, a.reps),
                           chain_twist_ms=timed(lambda: tw.log_chain(lwc), dev, a.reps),
                           chain_marginals_ms=timed(lambda: tw.marginals(lwc), dev, a.reps))
                # one step of the sampler = denoiser + class projection + 2 route evaluations + 1 route sample + the schedule
                row["step_parallel_ms"] = t_den + row["class_projection_ms"] + 2 * row["route_logZ_ms"] + row["route_sample_ms"] + row["reveal_parallel_ms"]
                row["step_sequential_ms"] = row["step_parallel_ms"] - row["reveal_parallel_ms"] + row["reveal_sequential_ms"]
                for k in [k for k in row if k.endswith("_ms") and k != "denoiser_ms"]:
                    row[k[:-3] + "_in_denoiser_calls"] = row[k] / t_den
                res["rows"].append(row)
                json.dump(res, open(a.out, "w"), indent=1)
                print(f"N={N} batch={Bsz} {cname:14s} denoiser {t_den:7.1f} ms | route logZ {row['route_logZ_ms']:6.1f} sample {row['route_sample_ms']:6.1f} marginals {row['route_marginals_ms']:6.1f} | "
                      f"reveal seq {row['reveal_sequential_ms']:7.1f} par {row['reveal_parallel_ms']:7.1f} | chain twist {row['chain_twist_ms']:6.1f} marginals {row['chain_marginals_ms']:6.1f} | "
                      f"step = {row['step_parallel_in_denoiser_calls']:.2f} denoiser calls (parallel), chain twist adds {row['chain_twist_in_denoiser_calls']:.2f}", flush=True)


if __name__ == "__main__":
    main()
