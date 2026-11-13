"""Batched route wrappers: per-position restrictions (TorchMasked) over any batched route."""
import torch
from .routes import onehot_log

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
