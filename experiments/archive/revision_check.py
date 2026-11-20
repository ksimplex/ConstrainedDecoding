"""Does the revision of the twist after a reveal predict the true per-step error?  Exact, every state, 10-12 positions.

After revealing value a at position i from state x the sampler re-runs the denoiser and its incremental weight is
    G(a) = psi(x + a) / psi~(x, a),   psi~(x, a) = psi(x) q_i(a) / m_i(a)
(psi = the approximate twist, q_i = the proposal, m_i = the model marginal).  Exactly,
    eps_i(x) = KL(q_i || pi_i) = E_q[-log G] + e(x) - E_q e(x + a),   e = log(psi* / psi)   (checked below)
so the MEAN revision sums to the order-dependent error over a decode but mixes in the correction the twist has to receive anyway,
and to second order eps_i(x) ~ Var_q(log G) / 2 when the remaining twist error e(x + a) does not vary with a.

Rules compared (exact KL of the sampler): ideal greedy on eps; greedy on Var(log G)/2; greedy on KL(q || q G / Z) (the same
quantity without the second-order expansion); greedy on the mean revision; least tilted; random; optimal.
These rules look one denoiser call ahead for every candidate, so they are a ceiling for a predictor learned from observed revisions.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "studies"))
import twist_error as te                                                                           # noqa: E402
from surrogate_plan import World, constraint_for                                                   # noqa: E402
from pulib.cells import make_model, n_symbols                                                      # noqa: E402
from pulib.summary import provenance                                                               # noqa: E402
from twistroute.decode.schedule import clamp_revealed                                              # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and ra.std() > 0 and rb.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10); ap.add_argument("--model", default="hidden_0.85"); ap.add_argument("--constraint", default="counting")
    ap.add_argument("--proposal", choices=("product", "chain"), default="product"); ap.add_argument("--match", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=2000); ap.add_argument("--out", default="revision_check.json")
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time(); N = a.N; S = n_symbols(a.constraint)
    den = make_model(a.model, N, S); W = World(den, a.constraint, N, proposal=a.proposal, pilot=a.pilot, match=a.match); B = W.B
    cl, route, truth = constraint_for(a.constraint, den, N); act = getattr(truth, "active", None)
    ref = te.Twists(den, truth.wdfa, act if act is not None else [True] * N); chain = None
    if a.proposal == "chain":
        from twistroute.decode import ChainTwist, fit_pair_kernel
        chain = ChainTwist(truth.wdfa, fit_pair_kernel(den, cl, a.pilot, torch.Generator().manual_seed(5)), active=act, match=a.match)
    dig = ((W.codes[:, None] // W.pw[None]) % (S + 1)).astype(np.int64); x_all = torch.from_numpy(dig)
    pm = np.zeros((B, N, S)); lpsi = np.zeros(B)
    for lo in range(0, B, 50000):
        x = x_all[lo:lo + 50000]; pm[lo:lo + 50000] = torch.exp(den.log_cond(x)).numpy()
        if chain is None:
            lpsi[lo:lo + 50000] = ref.product(x).numpy()
        else:
            ycls = cl.token_class[torch.where(x == S, torch.zeros_like(x), x)]
            lpsi[lo:lo + 50000] = chain.log_chain(clamp_revealed(cl.class_logw(den.logits(x)), x, ycls, S)).detach().numpy()
    e = np.where(W.alive, W.lt - lpsi, 0.0)
    mean = np.full((B, N), np.inf); half = np.full((B, N), np.inf); full = np.full((B, N), np.inf); worst = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(N):
            sel = W.codes[W.masked[:, i] & W.alive]; sc = W.succ(sel, i); q = W.q[sel, i]; ok = (q > 1e-300) & W.alive[sc] & np.isfinite(lpsi[sc])
            q = np.where(ok, q, 0.0); q = q / q.sum(1, keepdims=True)
            lg = np.where(ok, np.where(ok, lpsi[sc], 0.0) - lpsi[sel][:, None] - np.log(np.where(ok, q, 1.0)) + np.log(np.clip(pm[sel, i], 1e-300, None)), 0.0)
            d = -(q * lg).sum(1); mean[sel, i] = d; half[sel, i] = 0.5 * (q * (lg + d[:, None]) ** 2).sum(1)
            z = (q * np.exp(lg)).sum(1); full[sel, i] = np.log(np.clip(z, 1e-300, None)) + d              # KL(q || q G / Z)
            worst = max(worst, float(np.abs(W.eps[sel, i] - (d + e[sel] - (q * np.where(ok, e[sc], 0.0)).sum(1))).max()))
    e0 = W.e0; Vopt, pick_opt = W.solve(W.eps); res = dict(args=vars(a), provenance=provenance(), log_pC=W.log_pC, kl={}, rank={}, identity_max_abs_error=worst)
    res["kl"]["random"] = float(W.value_random()[e0]); res["kl"]["optimal"] = float(Vopt[e0])
    for name, tab in (("greedy_true_error", W.eps), ("greedy_half_variance_of_revision", half), ("greedy_one_step_corrected", full),
                      ("greedy_mean_revision", mean), ("least_tilted", W.tilt)):
        res["kl"][name] = float(W.value(W.argmin_pick(tab))[e0])
    # how well each score ranks the masked positions of a state against the true error: median Spearman over states with >= 4 masked
    rng = np.random.default_rng(0); st = W.codes[W.alive & (W.nm >= 4)]; st = rng.choice(st, size=min(4000, len(st)), replace=False)
    for name, tab in (("half_variance_of_revision", half), ("one_step_corrected", full), ("mean_revision", mean), ("tilt", W.tilt)):
        r = [spearman(W.eps[s][W.masked[s]], tab[s][W.masked[s]]) for s in st]; res["rank"][name] = float(np.nanmedian(r))
    print(f"{a.constraint} {a.model} N={N} {a.proposal}: identity eps = mean revision + e(x) - E e(x+a) holds to {worst:.1e}; {time.time() - t0:.0f} s")
    print("  exact KL (nats, lower is better): " + "  ".join(f"{k} {v:.4f}" for k, v in res["kl"].items()))
    print("  median rank correlation with the true error within a state (higher is better): " + "  ".join(f"{k} {v:.2f}" for k, v in res["rank"].items()))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
