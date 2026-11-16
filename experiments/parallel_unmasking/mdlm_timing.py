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


def reveal_stages(make_policy, st, dev, reps, gen):
    """Median time in ms of each stage of one reveal: the step context (one constrained-marginals pass), scoring,
    admission, and inside admission the forced-partner pull-in (its time and its number of route passes). Also the
    mean number of positions admitted per particle."""
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    out = dict(context=[], score=[], admit=[], pull=[], pulls=[], admitted=[])
    masked = st.x == st.mask_id
    for rep in range(reps + 1):                                     # the first pass is a warm-up
        pol = make_policy(); acc = dict(ms=0.0, n=0); inner = pol._pull

        def pull(*a, **k):
            sync(); t = time.perf_counter(); inner(*a, **k); sync()
            acc["ms"] += 1e3 * (time.perf_counter() - t); acc["n"] += 1
        pol._pull = pull
        with torch.no_grad():
            sync(); t0 = time.perf_counter(); s = pol.step_context(st, masked); sync(); t1 = time.perf_counter()
            score = pol.score(s, gen); sync(); t2 = time.perf_counter()
            chosen, _ = pol.admit(s, score, gen); sync(); t3 = time.perf_counter()
        if rep:
            for k, v in zip(out, (1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2), acc["ms"], acc["n"], float(chosen.sum(-1).double().mean()))):
                out[k].append(v)
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[128, 256]); ap.add_argument("--batches", type=int, nargs="+", default=[8, 64, 256])
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--traces", type=int, default=8); ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="mdlm_timing.json")
    ap.add_argument("--chunk", type=int, default=64, help="particles per denoiser call when drawing the test sequences")
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
            g3 = G(3)                                                                   # a plausible sequence per particle
            xs = torch.cat([native_sample(den, cl, N, 16, min(a.chunk, Bsz - i), g3) for i in range(0, Bsz, a.chunk)])
            x = torch.where(torch.rand(xs.shape, generator=G(4), device=dev) < 0.5, xs, torch.full_like(xs, den.mask_id))
            masked = x == den.mask_id
            ycls = cl.token_class[torch.where(masked, torch.zeros_like(x), x)]
            assert x.shape == (Bsz, N)
            try:                                # the full-vocabulary output of a large batch may not fit on the GPU
                t_den = timed(lambda: den.logits(x), dev, a.reps)                      # nothing large is held yet
                lp = den.logits(x); t_proj = timed(lambda: cl.class_logw(lp), dev, a.reps)
                lw = cl.class_logw(lp); del lp
            except torch.OutOfMemoryError as e:
                res["rows"].append(dict(N=N, batch=Bsz, error="out of GPU memory: " + str(e).split(".")[0]))
                json.dump(res, open(a.out, "w"), indent=1); print(f"N={N} batch={Bsz} out of GPU memory, cell skipped", flush=True)
                lp = None; torch.cuda.empty_cache(); continue
            lwc = clamp_revealed(lw, x, ycls, den.mask_id)
            assert lwc.shape[:2] == (Bsz, N)
            base_row = dict(N=N, batch=Bsz, denoiser_ms=t_den, denoiser_ms_per_particle=t_den / Bsz, pilot_kernel_s=t_kern, pilot_pairs_s=t_pair,
                            class_projection_ms=t_proj,
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
                row["reveal_parallel_nopull_ms"] = timed(lambda: pol(rate=1.0 / N).reveal(st, 0, N, G(6)), dev, a.reps)
                for tag, kw in (("parallel", dict(rate=1.0 / N, pull_forced=8)), ("parallel_nopull", dict(rate=1.0 / N)), ("sequential", dict(max_set=1))):
                    for k, v in reveal_stages(lambda: pol(**kw), st, dev, a.reps, G(6)).items():
                        row[f"stage_{tag}_{k}" + ("" if k in ("pulls", "admitted") else "_ms")] = v
                # one step of the sampler = denoiser + class projection + 2 route evaluations + 1 route sample + the schedule
                row["step_parallel_ms"] = t_den + row["class_projection_ms"] + 2 * row["route_logZ_ms"] + row["route_sample_ms"] + row["reveal_parallel_ms"]
                row["step_sequential_ms"] = row["step_parallel_ms"] - row["reveal_parallel_ms"] + row["reveal_sequential_ms"]
                for k in [k for k in row if k.endswith("_ms") and k != "denoiser_ms" and not k.startswith("stage_")]:
                    row[k[:-3] + "_in_denoiser_calls"] = row[k] / t_den
                res["rows"].append(row)
                json.dump(res, open(a.out, "w"), indent=1)
                print(f"N={N} batch={Bsz} {cname:14s} denoiser {t_den:7.1f} ms | route logZ {row['route_logZ_ms']:6.1f} sample {row['route_sample_ms']:6.1f} marginals {row['route_marginals_ms']:6.1f} | "
                      f"reveal seq {row['reveal_sequential_ms']:7.1f} par {row['reveal_parallel_ms']:7.1f} | chain twist {row['chain_twist_ms']:6.1f} marginals {row['chain_marginals_ms']:6.1f} | "
                      f"step = {row['step_parallel_in_denoiser_calls']:.2f} denoiser calls (parallel), chain twist adds {row['chain_twist_in_denoiser_calls']:.2f}", flush=True)
                q = lambda k: row[f"stage_parallel_{k}"]
                print(f"    parallel reveal stages: context {q('context_ms'):6.1f} score {q('score_ms'):5.1f} admit {q('admit_ms'):7.1f} of which pull-in {q('pull_ms'):7.1f} "
                      f"in {q('pulls'):.0f} route passes | without pull-in {row['reveal_parallel_nopull_ms']:7.1f} ms | "
                      f"admitted per particle {q('admitted'):.1f} (no pull-in {row['stage_parallel_nopull_admitted']:.1f})", flush=True)


if __name__ == "__main__":
    main()
