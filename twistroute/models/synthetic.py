"""
Synthetic context-dependent denoiser with the same interface as models.mdlm_owt.MDLM, for tests and
controls. Logits at every position are a fixed base plus a linear function of the revealed tokens:
    logits[b, i, v] = base[i, v] + sum_j w[i, j] * E[x_j, v]
`scale` sets the strength of re-conditioning (how much revealed tokens move the other predictions), which
is the quantity the Feynman-Kac weights correct.
"""
import math
import torch
from .classes import TokenClasses


class SyntheticMDLM:
    def __init__(self, N, V=12, scale=1.5, seed=5):
        self.V, self.mask_id = V + 1, V
        g = torch.Generator().manual_seed(seed)
        self.base = torch.randn(N, self.V, generator=g)
        self.E = torch.randn(self.V, self.V, generator=g) * scale
        self.E[self.mask_id] = 0
        self.w = torch.randn(N, N, generator=g) / math.sqrt(N)

    def logits(self, x):
        lg = self.base[None] + torch.einsum("ij,bjv->biv", self.w, self.E[x])
        lg[..., self.mask_id] = float("-inf")
        return torch.log_softmax(lg, -1)

    def classes(self, A=4):
        """Token v (non-mask) belongs to class v mod A."""
        cls = torch.arange(self.V) % A
        return TokenClasses(cls, A, excluded=(self.mask_id,))
