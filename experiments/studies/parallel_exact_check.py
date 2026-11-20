"""Cross-check of the sampler against the exact tables of parallel_exact.py at 8 positions.

For a deterministic budgeted schedule, the exact tables give P_par(C) / p(C): the value Z_hat / p(C) estimates when
several positions are revealed per step, for each choice of proposal (product, chain) and of reference law for a
co-revealed set (product of marginals, chain law). The sampler's average Z_hat / p(C) must agree with it.
"""
import argparse, json, math, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulib.cells import make_model, n_symbols                                                      # noqa: E402
from parallel_exact import PWorld                                                                  # noqa: E402
from surrogate_plan import constraint_for                                                          # noqa: E402
from twistroute.decode import twisted_smc, ChainTwist, fit_pair_kernel, CouplingPolicy, fit_distance_kernel, COUNT_EDGES   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--model", default="hidden_0.85"); ap.add_argument("--constraints", nargs="+", default=["counting", "forbidden"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.3, 1.0]); ap.add_argument("--nrun", type=int, default=6000); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default="parallel_exact_check.json")
    a = ap.parse_args(); torch.set_num_threads(2); N = a.N; res = []
    for cons in a.constraints:
        den = make_model(a.model, N, n_symbols(cons)); cl, route, truth = constraint_for(cons, den, N)
        kern, _ = fit_distance_kernel(den, N, 2000, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
        tw = ChainTwist(truth.wdfa, fit_pair_kernel(den, cl, 2000, torch.Generator().manual_seed(5)), active=getattr(truth, "active", None))
        for prop in ("product", "chain"):
            for ref in ("product", "chain"):
                W = PWorld(den, cons, N, prop, 2000, ref)
                for bud in a.budgets:
                    sc = W.score(W.repo_policy("lex", bud, kern)); want = math.exp(sc["log_z_bias"])
                    pol = CouplingPolicy(kern, 0.0, priority="lex", rate=bud / N)
                    out = twisted_smc(den, cl, route, N, N, nrun=a.nrun, K=a.K, gen=torch.Generator().manual_seed(7), ess_min=0.0, schedule=pol,
                                      proposal_twist=tw if prop == "chain" else None, reference_twist=tw if ref == "chain" else None)
                    z = np.exp(out["logZ"] - truth.log_pC); got, se = float(z.mean()), float(z.std(ddof=1) / math.sqrt(len(z)))
                    row = dict(constraint=cons, proposal=prop, reference=ref, budget=bud, steps_exact=sc["steps"], steps_sampler=float(np.mean(out["steps"])),
                               exact=want, sampler=got, se=se, z=(got - want) / se, bias_kl=sc["bias"])
                    res.append(row)
                    print(f"{cons:10s} {prop:7s} proposal, {ref:7s} reference, B={bud:g}: steps {sc['steps']:.2f} (sampler {row['steps_sampler']:.2f}) | "
                          f"P_par(C)/p(C) exact {want:.4f} | sampler {got:.4f} +- {se:.4f} ({row['z']:+.1f} s.e.) | KL of the limit from the target {sc['bias']:.4f}", flush=True)
    json.dump(dict(args=vars(a), rows=res), open(a.out, "w"), indent=1)
    assert max(abs(r["z"]) for r in res) < 4.0


if __name__ == "__main__":
    main()
