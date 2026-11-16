"""Orders supplied to CouplingPolicy as functions of the step context: scores for "which position is cheapest to
reveal now", in the KL error it adds to the sampler's law.

    cheapest_order   the true error KL(product twist's constrained marginal || exact conditional) at each masked
                     position. Needs the exact twist, so it exists on oracle models only: the ideal reference.
    estimate_order   the same with the chain twist's constrained marginal in place of the exact one: no denoiser call.
The lowest-entropy order ("confident") is built into the policy.
"""
import torch


def kl(p, q):
    """KL(p || q) along the last axis; inf where p puts mass on a value q excludes."""
    t = torch.where(p > 0, p * (torch.log(p.clamp_min(1e-300)) - torch.log(q.clamp_min(1e-300))), torch.zeros_like(p))
    out = t.sum(-1).clamp_min(0.0)
    return torch.where(((p > 1e-12) & (q <= 0)).any(-1), torch.full_like(out, float("inf")), out)


def exact_conditionals(ref, den, x, mu, chunk=20000):
    """pi (B, N, S): p(class at i | revealed tokens of x, C) at the masked positions, by clamping each masked
    position to each class and evaluating the exact twist (`ref`: twist_error.Twists). Zeros at revealed positions
    and where the constraint is already broken."""
    B, N = x.shape
    b, i = (x == den.mask_id).nonzero(as_tuple=True)
    pi = torch.zeros(B, N, den.S, dtype=torch.float64)
    for lo in range(0, len(b), chunk):
        bb, ii = b[lo:lo + chunk], i[lo:lo + chunk]
        for a in range(den.S):
            xa = x[bb].clone(); xa[torch.arange(len(bb)), ii] = a
            pi[bb, ii, a] = mu[bb, ii, a] * torch.exp(ref.true(xa))
    return pi / pi.sum(-1, keepdim=True).clamp_min(1e-300)


def cheapest_order(ref, den):
    def score(s):
        x = s.st.x
        pi = exact_conditionals(ref, den, x, torch.exp(den.log_cond(x)))
        e = kl(s.mu, pi)
        return torch.where(torch.isfinite(e), -e, torch.full_like(e, -1e30))
    return score


def estimate_order(twist):
    def score(s):
        e = kl(s.mu, twist.marginals(s.lwc))
        return torch.where(torch.isfinite(e), -e, torch.full_like(e, -1e30))
    return score
