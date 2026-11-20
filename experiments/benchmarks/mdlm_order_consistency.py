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

Also recorded, to judge whether the samples are degenerate: the model's entropy at the revealed position in each
quarter of the generating order, the share of distinct tokens in a sample, and the first few samples as text.
--real-text scores passages of ordinary English prose (below) under the same orders instead of model samples: there
is then no generating order, and the spread over orders and the gap to the mixture are the quantities of interest.
"""
PASSAGES = [
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters. My dear Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let at last? Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long has just been here, and she told me all about it. Mr. Bennet made no answer. Do you not want to know who has taken it? cried his wife impatiently. You want to tell me, and I have no objection to hearing it. This was invitation enough.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way; in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne of England; there were a king with a large jaw and a queen with a fair face, on the throne of France.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off, then, I account it high time to get to sea as soon as I can.",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book, thought Alice, without pictures or conversations? So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be late!",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate, we can not consecrate, we can not hallow this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract.",
    "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation. We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed.",
]

import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "studies"))
from pulib.summary import provenance                                                               # noqa: E402


@torch.no_grad()
def path_logp(den, x, order, chunk, sample=False, gen=None, ent=None):
    """log p_sigma(x) for each row: x (B, N) tokens, order (B, N) a permutation per row. With sample=True, x is
    ignored and the tokens are drawn along the order; returns (log p, x). `ent` (N,), if given, receives the mean
    entropy of the model's prediction at the revealed position, by step."""
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
            if ent is not None:
                ent[t] += float(-(torch.exp(lg) * torch.nan_to_num(lg, neginf=0.0)).sum(-1).sum()) / B
            cur[r, pos[r]] = tok
    return (lp, cur) if sample else lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[32, 64, 128]); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--orders", type=int, default=18); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--real-text", action="store_true", help="score the built-in prose passages instead of model samples")
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
        tok = getattr(den, "tok", None); ent = np.zeros(N); n = a.n
        if a.real_text:
            if tok is not None:
                ids = [tok(t)["input_ids"] for t in PASSAGES]
            else:                                                       # code check without a tokenizer
                ids = [[ord(ch) % (den.V - 1) for ch in t] for t in PASSAGES]
            ids = [torch.tensor(v[lo:lo + N]) for v in ids for lo in range(0, len(v) - N + 1, N)]
            if not ids:
                print(f"N={N}: no passage is {N} tokens long, skipped", flush=True); continue
            x = torch.stack(ids).to(dev); n = len(x); lp0 = None
        else:
            lp0, x = path_logp(den, None, perm(n), a.chunk, sample=True, gen=g, ent=ent)   # sequences and their own order
        R = a.orders
        xs = x.repeat_interleave(R, 0); order = perm(n * R).view(n, R, N)
        order[:, -2] = torch.arange(N, device=dev); order[:, -1] = torch.arange(N - 1, -1, -1, device=dev)
        lp = path_logp(den, xs, order.view(-1, N), a.chunk).view(n, R).cpu().numpy()
        rnd, ltr, rtl = lp[:, :-2], lp[:, -2], lp[:, -1]
        m = rnd.mean(1); sd = rnd.std(1, ddof=1)
        lme = np.log(np.exp(rnd - rnd.max(1, keepdims=True)).mean(1)) + rnd.max(1)
        se = lambda v: float(np.std(v, ddof=1) / math.sqrt(len(v)))
        own = lp0.cpu().numpy() - m if lp0 is not None else np.full(n, np.nan)
        q4 = [float(ent[j * N // 4:(j + 1) * N // 4].mean()) for j in range(4)] if lp0 is not None else None
        distinct = float(np.mean([len(set(r.tolist())) / N for r in x.cpu()]))
        texts = [tok.decode(r.tolist())[:400] for r in x[:4].cpu()] if tok is not None else None
        row = dict(N=N, source="real text" if a.real_text else "model samples", sequences=n, random_orders=R - 2, mean_neg_logp_per_token=float(-m.mean() / N),
                   entropy_by_quarter_of_own_order=q4, own_order_neg_logp_per_token=(float(-lp0.mean() / N) if lp0 is not None else None),
                   distinct_token_share=distinct, first_sequences=texts,
                   sd_over_orders=float(sd.mean()), sd_over_orders_se=se(sd), sd_over_orders_per_token=float(sd.mean() / N),
                   gap_to_mixture=float((lme - m).mean()), gap_to_mixture_se=se(lme - m),
                   left_to_right_shift=float((ltr - m).mean()), left_to_right_shift_se=se(ltr - m),
                   right_to_left_shift=float((rtl - m).mean()), right_to_left_shift_se=se(rtl - m),
                   own_order_shift=(float(own.mean()) if lp0 is not None else None), own_order_shift_se=(se(own) if lp0 is not None else None),
                   max_abs_spread=float((rnd.max(1) - rnd.min(1)).max()), seconds=round(time.time() - t0, 1))
        res["rows"].append(row); json.dump(res, open(a.out, "w"), indent=1)
        print(f"N={N}: -log p per token {row['mean_neg_logp_per_token']:.3f} | sd over orders {row['sd_over_orders']:.3f} +- {row['sd_over_orders_se']:.3f} nats "
              f"({row['sd_over_orders_per_token']:.4f} per token) | gap to mixture {row['gap_to_mixture']:.3f} +- {row['gap_to_mixture_se']:.3f} | "
              f"left-to-right {row['left_to_right_shift']:+.3f} +- {row['left_to_right_shift_se']:.3f}, right-to-left {row['right_to_left_shift']:+.3f} +- {row['right_to_left_shift_se']:.3f}, "
              + (f"own order {row['own_order_shift']:+.3f} +- {row['own_order_shift_se']:.3f}, entropy by quarter {' '.join('%.2f' % v for v in q4)}" if lp0 is not None else "real text")
              + f" | distinct tokens {distinct:.2f} | n={n} | {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
