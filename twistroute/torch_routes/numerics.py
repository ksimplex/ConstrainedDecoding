"""Numerics shared by the batched routes."""
import torch


def logsumexp_safe(x, dim):
    """torch.logsumexp with a defined gradient where every entry along `dim` is -inf.

    The value there is -inf either way. The gradient of torch.logsumexp is NaN (0 * inf), and one NaN poisons every
    marginal obtained by differentiating logZ (decode.policy.constrained_marginals): charts and padded factor tables
    have such all -inf slices by construction. Here the gradient is 0 there. Without autograd the call is
    torch.logsumexp itself, so sampling and weights are unchanged bit for bit."""
    if not x.requires_grad:
        return torch.logsumexp(x, dim=dim)
    dead = torch.isneginf(x).all(dim=dim, keepdim=True)
    out = torch.logsumexp(torch.where(dead, torch.zeros_like(x), x), dim=dim)
    return torch.where(dead.squeeze(dim), torch.full_like(out, float("-inf")), out)
