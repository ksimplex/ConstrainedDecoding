"""The constraint interface (notes, section 2 and Proposition 1).

A constraint C is a set of allowed sequences of length L. The sampler never looks inside it. It
asks three questions, each about a factorised distribution q(x) = prod_i q_i(x_i) over token
classes:

    log_prob(log_q)   log q(C) = log sum_{x in C} prod_i q_i(x_i)        (denominator of eq. 3)
    marginals(log_q)  q(x_i = a | C) at every position                    (Proposition 1, item 1)
    sample(log_q)     a draw x ~ q(. | C)                                 (Proposition 1, item 2)

Any family that can answer the first exactly can answer the other two (Proposition 1). The
families of Table 1 each answer all three in one pass; their classes live beside this file.

Conventions
    log_q      tensor (batch, L, A): log q_i(a). Rows need not be normalised; log_prob then
               returns the log of the weighted sum. A position that is already unmasked is
               given the one-hot of its class (`fix_unmasked`).
    classes    A token classes: groups of tokens the constraint treats alike (section 2.1).
    sequences  integer tensors (batch, L) of class indices.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

NEG_INF = float("-inf")


def one_hot_log(classes: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(...,) class indices -> (..., A) log one-hot: 0 at the class, -inf elsewhere."""
    out = torch.full(
        (*classes.shape, n_classes), NEG_INF, dtype=torch.float64, device=classes.device
    )
    return out.scatter(-1, classes[..., None], 0.0)


def fix_unmasked(
    log_q: torch.Tensor, classes: torch.Tensor, unmasked: torch.Tensor
) -> torch.Tensor:
    """log_q with the unmasked positions replaced by the one-hot of their class.

    This turns the mean-field prediction at x_t into the factorised distribution whose mass on C
    is p(x_0 in C | x_t) in eq. (3): unmasked positions are certain, masked ones follow the model.
    """
    fixed = one_hot_log(classes, log_q.shape[-1]).to(log_q.dtype)
    return torch.where(unmasked[..., None], fixed, log_q)


class Constraint:
    """A constraint, wrapped around one of the exact implementations in `twistroute.torch_routes`.

    family     name of the family in Table 1 of the notes
    cost       cost of one pass, as in Table 1
    n_classes  A
    length     L if the constraint only makes sense at one length, else None
    reference  the CPU object the implementation was built from (used by `intersect`)

    For an ambiguous grammar every sequence is weighted by its number of parses (notes, section
    2.1); `log_prob` is then the log of that weighted sum.
    """

    def __init__(
        self,
        implementation: Any,
        *,
        family: str,
        cost: str,
        n_classes: int,
        length: Optional[int] = None,
        reference: Any = None,
        name: str = "",
    ) -> None:
        self._implementation = implementation
        self.family = family
        self.cost = cost
        self.n_classes = n_classes
        self.length = length
        self.reference = reference
        self.name = name or family

    def __repr__(self) -> str:
        described = f"family={self.family!r}, A={self.n_classes}, cost={self.cost}"
        return f"Constraint({self.name!r}, {described})"

    def log_prob(self, log_q: torch.Tensor) -> torch.Tensor:
        """(batch, L, A) -> (batch,): log q(C)."""
        return self._implementation.logZ(log_q)

    def marginals(self, log_q: torch.Tensor) -> torch.Tensor:
        """(batch, L, A) -> (batch, L, A): q(x_i = a | C). Rows with q(C) = 0 come back as zeros.

        Computed as the gradient of log q(C) with respect to log_q, which is the constrained
        marginal; one backward pass, the same cost as `log_prob`. Round-off can leave about
        1e-16 where the exact answer is 0 (a class the constraint excludes): compare with a
        tolerance, do not test for exact zeros.
        """
        with torch.enable_grad():
            v = log_q.detach().double().clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(self._implementation.logZ(v).sum(), v)
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    def sample(
        self, log_q: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """(batch, L, A) -> (batch, L): one exact draw from q(. | C) per row."""
        return self._implementation.sample(log_q, generator)

    def contains(self, sequences: torch.Tensor) -> torch.Tensor:
        """(batch, L) class indices -> (batch,) bool: is the sequence in C?"""
        return self._implementation.accepts(sequences)

    @property
    def route(self) -> Any:
        """The wrapped implementation, under its old name, for code written against
        `twistroute.decode` and `twistroute.torch_routes`."""
        return self._implementation
