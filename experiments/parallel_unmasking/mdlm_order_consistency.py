"""How order-consistent is a masked diffusion model? One token per step, the model assigns a sequence x the
probability p_sigma(x) = prod_t p(x_sigma(t) | x_sigma(<t)) under a reveal order sigma. A model with exact
conditionals gives the same value for every order; a trained denoiser does not. The reference law for constrained
decoding is the average over uniformly random orders, so the spread of log p_sigma(x) across orders says how far the
law under one particular order (for instance a constraint-aware one) can be from that reference.

For n sequences drawn from the model one token per step in a random order, and R further orders each (R - 2 random,
plus left-to-right and right-to-left), reports per sequence length N:
    sd_over_orders        standard deviation of log p_sigma(x) over the random orders, mean over sequences (nats)
    gap_to_mixture        log mean_sigma p_sigma(x) - mean_sigma log p_sigma(x), mean over sequences: the average KL
                          from the random-order mixture to a single random order, estimated from R - 2 orders (a
                          lower bound on the log of the mean, so a slight underestimate)
    left_to_right_shift   mean over sequences of log p_ltr(x) - mean_sigma log p_sigma(x); likewise right_to_left
    per_token             the same quantities divided by N, and the mean of -log p_sigma(x) / N for scale
--oracle runs an exact hidden-chain model, where every number must be zero (a check of this script); --synthetic
runs the small synthetic denoiser on CPU (a code check, not a result).
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulib.summary import provenance                                                               # noqa: E402


@torch.no_grad()
def path_logp(den, x, order, chunk, sample=False, gen=None):
    """log p_sigma(x) for each row: x (B, N) tokens, order (B, N) a permutation per row. With sample=True, x is
    ignored and the tokens are drawn along the order; returns (log p, x)."""
    B, N = order.shape; dev = order.device
    cur = torch.full((B, N), den.mask_id, dtype=torch.long, device=dev)
    lp = torch.zeros(B, dtype=torch.float64, device=dev); rows = torch.arange(B, device=dev)
    for t in range(N):
        pos = order[:, t]
        for lo in range(0, B, chunk):
            r = rows[lo:lo + chunk]
            lg = den.logits(cur[r])[torch.arange(len(r), device=dev), pos[r]].double()              # (b, V)
            if sample:
                tok = torch.multinomial(torch.exp(lg).float(), 1, generator=gen).squeeze(-1)
            else:
                tok = x[r, pos[r]]
            lp[r] += lg[torch.arange(len(r), device=dev), tok]
            cur[r, pos[r]] = tok
    return (lp, cur) if sample else lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[32, 64, 128]); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--orders", type=int, default=18); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="mdlm_order_consistency.json")
    a = ap.parse_args()
    res = dict(args=vars(a), provenance=provenance(), rows=[]); den_real = None
    for N in a.N:
        if a.oracle:
            from pulib.cells import make_model
            dev = "cpu"; den = make_model("hidden_0.85", N, 3)
        elif a.synthetic:
            from twistroute.models import SyntheticMDLM
            dev = "cpu"; den = SyntheticMDLM(N, scale=3.0)
        else:
            from twistroute.models.mdlm_owt import MDLM
            dev = "cuda"; den_real = den_real or MDLM(dev); den = den_real
            res["model"] = den.info(); res["gpu"] = torch.cuda.get_device_name(0)
        g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + N); t0 = time.time()
        perm = lambda B: torch.argsort(torch.rand(B, N, generator=g, device=dev), -1)
        lp0, x = path_logp(den, None, perm(a.n), a.chunk, sample=True, gen=g)          # sequences and their own order
        R = a.orders
        xs = x.repeat_interleave(R, 0); order = perm(a.n * R).view(a.n, R, N)
        order[:, -2] = torch.arange(N, device=dev); order[:, -1] = torch.arange(N - 1, -1, -1, device=dev)
        lp = path_logp(den, xs, order.view(-1, N), a.chunk).view(a.n, R).cpu().numpy()
        rnd, ltr, rtl = lp[:, :-2], lp[:, -2], lp[:, -1]
        m = rnd.mean(1); sd = rnd.std(1, ddof=1)
        lme = np.log(np.exp(rnd - rnd.max(1, keepdims=True)).mean(1)) + rnd.max(1)
        se = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v)))
        row = dict(N=N, sequences=a.n, random_orders=R - 2, mean_neg_logp_per_token=float(-m.mean() / N),
                   sd_over_orders=float(sd.mean()), sd_over_orders_se=se(sd), sd_over_orders_per_token=float(sd.mean() / N),
                   gap_to_mixture=float((lme - m).mean()), gap_to_mixture_se=se(lme - m),
                   left_to_right_shift=float((ltr - m).mean()), left_to_right_shift_se=se(ltr - m),
                   right_to_left_shift=float((rtl - m).mean()), right_to_left_shift_se=se(rtl - m),
                   own_order_shift=float((lp0.cpu().numpy() - m).mean()), own_order_shift_se=se(lp0.cpu().numpy() - m),
                   max_abs_spread=float((rnd.max(1) - rnd.min(1)).max()), seconds=round(time.time() - t0, 1))
        res["rows"].append(row); json.dump(res, open(a.out, "w"), indent=1)
        print(f"N={N}: -log p per token {row['mean_neg_logp_per_token']:.3f} | sd over orders {row['sd_over_orders']:.3f} +- {row['sd_over_orders_se']:.3f} nats "
              f"({row['sd_over_orders_per_token']:.4f} per token) | gap to mixture {row['gap_to_mixture']:.3f} +- {row['gap_to_mixture_se']:.3f} | "
              f"left-to-right {row['left_to_right_shift']:+.3f} +- {row['left_to_right_shift_se']:.3f}, right-to-left {row['right_to_left_shift']:+.3f} +- {row['right_to_left_shift_se']:.3f}, "
              f"own order {row['own_order_shift']:+.3f} +- {row['own_order_shift_se']:.3f} | {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
