"""Batched route wrappers over any batched route: per-position restrictions (TorchMasked) and a constraint on
a subset of positions with the rest free (TorchSubset)."""
import torch
from .routes import onehot_log, _gumbel_argmax

NEG = float("-inf")


class TorchMasked:
    """route AND (y_i in allowed[i]); allowed: (L, A) bool. Weights outside the allowed sets are set to -inf
    before the inner route's pass (compile.Masked is the CPU reference)."""

    def __init__(self, inner, allowed, device="cpu", name=None):
        self.inner = inner
        self.allowed = torch.as_tensor(allowed, dtype=torch.bool, device=device)
        self.L, self.A = self.allowed.shape
        self.device = device
        self.name = name or f"masked({getattr(inner, 'name', 'route')})"

    def _mask(self, lw):
        return torch.where(self.allowed, lw, torch.full_like(lw, NEG))

    def logZ(self, lw, **kw):
        return self.inner.logZ(self._mask(lw), **kw)

    def sample(self, lw, gen=None, **kw):
        return self.inner.sample(self._mask(lw), gen, **kw)

    def accepts(self, y):
        return torch.isfinite(self.logZ(onehot_log(y, self.A)))


class TorchSubset:
    """`inner` applied to the positions `positions` (in that order) of a length-L sequence; every other position
    is free. logZ(lw) = inner.logZ(lw[:, positions]) + sum over free positions of logsumexp(lw_i): the cost is
    the inner route's on len(positions) positions plus O(L A), so a constraint on a window of a long sequence
    does not pay for an automaton pass over the whole sequence. Differentiable wherever `inner` is."""

    def __init__(self, inner, positions, L, device="cpu", name=None):
        pos = [int(p) for p in positions]
        if len(set(pos)) != len(pos) or not all(0 <= p < L for p in pos):
            raise ValueError("positions must be distinct and lie in [0, L)")
        self.inner, self.L, self.A, self.device = inner, L, inner.A, device
        self.pos = torch.tensor(pos, dtype=torch.long, device=device)
        chosen = set(pos)
        self.free = torch.tensor([i for i in range(L) if i not in chosen], dtype=torch.long, device=device)
        self.name = name or f"subset({getattr(inner, 'name', 'route')}, {len(pos)} of {L})"

    def logZ(self, lw, **kw):
        z = self.inner.logZ(lw[:, self.pos], **kw).double()
        if self.free.numel():
            z = z + torch.logsumexp(lw[:, self.free].double(), -1).sum(-1)
        return z

    def sample(self, lw, gen=None, **kw):
        y = _gumbel_argmax(lw, gen)
        y[:, self.pos] = self.inner.sample(lw[:, self.pos], gen, **kw)
        return y

    def accepts(self, y):
        return self.inner.accepts(y[:, self.pos])
