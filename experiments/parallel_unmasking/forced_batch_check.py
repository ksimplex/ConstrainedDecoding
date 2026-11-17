"""Exact check of the forced-only batch step (CouplingPolicy(forced_batch=...)) at a size where every sequence can be listed.

Constraint: "content* end+" over three symbols (0, 1 content, 2 end), the shape of a call followed by padding: once an end
symbol is revealed, every later position is forced. Target pi(x) = p(x) 1[x in C] / p(C), exact from the oracle's joint.
Free positions are taken in uniformly random order, one per step, so that an end symbol revealed mid-sequence forces a
block (under the sum rule the right-most positions go first and blocks are rare).
Models: the project's oracles, on which the end symbol is not sticky (the model stays unsure about padding), and
`sticky_end_<delta>`, a chain that leaves the end symbol with probability delta per step (padding as a language model sees it).
Arms: forced positions one at a time (the reference), all forced positions in one step whatever the model says
(tolerance inf), and the tolerance rule. Reported per arm and particle count K:
    mean of Z_hat / p(C)        1 when the weights are exact (lower or higher = weight error)
    KL(output law || pi)        from the empirical law of the returned sample, minus the plug-in bias (cells - 1) / (2 n)
    steps                       model calls per particle
"""
import argparse, json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulib.cells import make_model                                                                  # noqa: E402
from twistroute.compile.automata import WDFA                                                        # noqa: E402
from twistroute.models import ConstrainedTruth, OracleHMM                                                      # noqa: E402
from twistroute.torch_routes import TorchChain                                                      # noqa: E402
from twistroute.decode import twisted_smc, CouplingPolicy, fit_distance_kernel, COUNT_EDGES         # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=8); ap.add_argument("--models", nargs="+", default=["hidden_0.85", "chain_0.85", "latent_0.6_w0.5", "sticky_end_0.001", "sticky_end_0.02"])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 4]); ap.add_argument("--n", type=int, default=40000, help="returned samples per arm")
    ap.add_argument("--tols", type=float, nargs="+", default=[0.01, 0.1]); ap.add_argument("--out", default="forced_batch_check.json")
    a = ap.parse_args(); N, S = a.N, 3; res = dict(args=vars(a), cells={})
    d = WDFA(2, [(0, 0, 0), (0, 1, 0), (0, 2, 1), (1, 2, 1)], 0, [1], S, name="content*end+")
    xs = torch.tensor(np.array(np.meshgrid(*[np.arange(S)] * N, indexing="ij")).reshape(N, -1).T)
    ok = torch.tensor([d.accepts(r.tolist()) for r in xs]); pw = S ** torch.arange(N - 1, -1, -1)                    # index of a sequence in xs: first position most significant
    for m in a.models:
        if m.startswith("sticky_end_"):            # content symbols mix; the end symbol is left with probability delta per step
            dl = float(m[len("sticky_end_"):]); T = np.array([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [dl / 2, dl / 2, 1 - dl]])
            den = OracleHMM([0.45, 0.45, 0.1], np.tile(T, (N - 1, 1, 1)), np.tile(np.eye(S), (N, 1, 1)), name=m)
        else:
            den = make_model(m, N, S)
        cl = den.classes(); truth = ConstrainedTruth(den, cl, d); route = TorchChain(d)
        lj = den.log_joint(xs).double(); pi = torch.where(ok, torch.exp(lj - truth.log_pC), torch.zeros_like(lj))
        assert abs(float(pi.sum()) - 1) < 1e-8, float(pi.sum())
        kern, _ = fit_distance_kernel(den, N, 2000, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
        arms = [("one at a time", None), ("all forced in one step", float("inf"))] + [(f"tolerance {t:g}", t) for t in a.tols]
        for K in a.Ks:
            for name, fb in arms:
                pol = CouplingPolicy(kern, 0.0, priority="random", mixed=False, max_set=1, forced_batch=fb)
                o = twisted_smc(den, cl, route, N, N, nrun=a.n, K=K, gen=torch.Generator().manual_seed(100 + K), return_particles=True, schedule=pol)
                lz = np.asarray(o["logZ"], dtype=np.float64); z = np.exp(lz - truth.log_pC)
                g = np.random.default_rng(5).gumbel(size=np.asarray(o["logw"]).shape)
                pick = np.argmax(np.asarray(o["logw"]) + g, axis=1); tok = torch.as_tensor(np.asarray(o["tokens"]))[torch.arange(a.n), torch.as_tensor(pick)]
                q = torch.bincount((tok * pw).sum(-1), minlength=S ** N).double() / a.n
                assert float(q[~ok].sum()) == 0.0
                kl = float(torch.xlogy(q[ok], q[ok] / pi[ok]).sum()) - (int(ok.sum()) - 1) / (2 * a.n)
                lr = torch.where(q[ok] > 0, torch.log(q[ok].clamp_min(1e-300) / pi[ok]), torch.zeros_like(q[ok]))
                se = math.sqrt(max(float((q[ok] * lr ** 2).sum()) - float((q[ok] * lr).sum()) ** 2, 0.0) / a.n)      # s.e. of the plug-in KL
                steps = float(np.mean(np.asarray(o["steps"], dtype=np.float64)))
                r = dict(mean_Z_ratio=float(z.mean()), se_Z_ratio=float(z.std(ddof=1) / math.sqrt(len(z))), kl_output_law=kl, kl_noise=se, steps=steps,
                         forced_reveals_per_forced_step=(pol.forced_reveals / pol.forced_steps if pol.forced_steps else None))
                res["cells"][f"{m}|K{K}|{name}"] = r
                print(f"{m:16s} K {K}  {name:24s} mean Z_hat / p(C) {r['mean_Z_ratio']:.4f} +- {r['se_Z_ratio']:.4f}   KL(output law || target) {kl:+.5f} (s.e. {r['kl_noise']:.5f})   steps {steps:.2f}", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
