"""
Shared plumbing for the experiment scripts: denoiser loading, the TWISTER Fig. 1(a) setup, batched SMC runs,
native rejection counts and Z_hat summaries. Keeping these here means every script draws random numbers in the
same order for the same configuration, which the regression tests rely on.
"""
import math
import time
import numpy as np
import torch
from .decode import twisted_smc, native_sample, select_particles
from .models import ranked_classes
from .torch_routes import build_route


def load_denoiser(N, synthetic=False, device=None, V=40, scale=2.0):
    """OWT-130M on CUDA, or the synthetic denoiser (V tokens, context strength `scale`) for CPU dry runs.
    Returns (model, device)."""
    if synthetic:
        from .models import SyntheticMDLM
        dev = device or "cpu"
        M = SyntheticMDLM(N, V=V, scale=scale)
        if dev != "cpu":
            M.base, M.E, M.w = M.base.to(dev), M.E.to(dev), M.w.to(dev)
        M.info = lambda: dict(synthetic=True, V=V, scale=scale)
        M.tok = type("Tok", (), {"decode": staticmethod(lambda ids: f"<{ids[0]}>")})()
        return M, dev
    from .models.mdlm_owt import MDLM
    dev = device or "cuda"
    return MDLM(dev), dev


@torch.no_grad()
def masked_marginal(M, N, dev):
    """Token probabilities under the fully masked input, averaged over positions (TWISTER Sec. 4.3)."""
    return torch.exp(M.logits(torch.full((1, N), M.mask_id, device=dev)))[0].mean(0)


def fig1a_setup(M, N, dev, pool=8, seed=1, n_classes=4):
    """TWISTER Fig. 1(a) constraint "a.*": ranked token classes (top `pool` tokens split into n_classes, the rest
    in class n_classes) and the prefix route "first position in class 0". Returns (classes, route)."""
    cl = ranked_classes(masked_marginal(M, N, dev), n_classes, pool, M.mask_id, seed=seed).to(dev)
    route = build_route(dict(kind="prefix", cls=0, A=n_classes + 1, N=N), dev)
    return cl, route


def native_hits(M, classes, route, N, T, n, gen, chunk=512):
    """Number of n native samples (drawn in chunks) whose class sequence satisfies the route's constraint."""
    hits = 0
    for i in range(0, n, chunk):
        hits += int(route.accepts(native_sample(M, classes, N, T, min(chunk, n - i), gen)).sum())
    return hits


def run_smc(M, classes, route, N, T, runs, K, gen, chunk=512, select_seed=None, **smc_kwargs):
    """`runs` independent SMC runs of K particles, in chunks of max(1, chunk // K) runs per batch.

    select_seed   optional callable (chunk_index, first_run) -> int; if given, one particle per run is drawn
                  with numpy rng(select_seed(...)) and returned as 'samples' (runs, N)
    smc_kwargs    passed to twisted_smc (ess_min, defensive, twist_power, resampler, res_gen, return_trace, ...)
    Returns concatenated 'logZ', 'n_res', 'ess_min' (runs,), optional 'samples' and 'trace' (T, runs, K), plus
    'seconds', 'frac_valid_finite' (mean over chunks) and 'any_finite_invalid'.
    """
    per, t0 = max(1, chunk // K), time.time()
    keys, parts, samples, traces, fv, bad = ("logZ", "n_res", "ess_min"), [], [], [], [], False
    for j, i in enumerate(range(0, runs, per)):
        o = twisted_smc(M, classes, route, N, T, min(per, runs - i), K, gen,
                        return_particles=select_seed is not None, **smc_kwargs)
        parts.append([o[k] for k in keys])
        if select_seed is not None:
            samples.append(select_particles(o, np.random.default_rng(select_seed(j, i))))
        if "trace" in o:
            traces.append(o["trace"])
        fv.append(o["frac_valid_finite"]); bad |= o["any_finite_invalid"]
    out = {k: np.concatenate([p[n] for p in parts]) for n, k in enumerate(keys)}
    if samples:
        out["samples"] = np.concatenate(samples)
    if traces:
        out["trace"] = np.concatenate(traces, axis=1)
    out.update(seconds=time.time() - t0, frac_valid_finite=float(np.mean(fv)), any_finite_invalid=bool(bad))
    return out


def z_summary(logZ, h0):
    """Mean Z_hat / h0 and its naive relative standard error (std / sqrt(n) / h0)."""
    z = np.exp(np.asarray(logZ, float))
    return dict(ratio=float(z.mean() / h0), se_rel=float(z.std() / math.sqrt(len(z)) / h0))
