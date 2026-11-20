"""The state of the sampler: partly masked sequences x_t and the model's prediction at them.

One row per particle. Everything a twist or a schedule may consult is here; neither ever sees
the tokens that are about to be drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .constraints.base import Constraint, fix_unmasked


@dataclass
class State:
    """Particles at one time.

    tokens         (P, L) x_t: token ids, `mask_id` at masked positions
    classes        (P, L) class of the token at each unmasked position (0 at masked ones)
    log_p          (P, L, V) log p_theta(x_0,i = v | x_t), the model's prediction (notes, eq. 2)
    log_q          (P, L, A) the same prediction summed within token classes
    mask_id        id of the mask token
    constraint     the constraint C
    token_classes  the map from tokens to classes (`twistroute.models.TokenClasses`)
    log_psi        (P,) log psi(x_t), filled in by the sampler
    cache          per-particle tensors a twist wants to reuse at the next step
    """

    tokens: torch.Tensor
    classes: torch.Tensor
    log_p: Optional[torch.Tensor]
    log_q: Optional[torch.Tensor]
    mask_id: int
    constraint: Constraint
    token_classes: Any
    log_psi: Optional[torch.Tensor] = None
    cache: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def masked(self) -> torch.Tensor:
        """(P, L) bool: the positions still masked."""
        return self.tokens == self.mask_id

    @property
    def complete(self) -> torch.Tensor:
        """(P,) bool: no masked position left, so x_t is a full sequence x_0."""
        return (self.tokens != self.mask_id).all(-1)

    def prediction(self, log_q: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The mean-field prediction at x_t as a factorised distribution over class sequences:
        the model's class marginals at masked positions, certainty at unmasked ones. `log_q`
        replaces the model's marginals (for instance by tempered ones)."""
        return fix_unmasked(self.log_q if log_q is None else log_q, self.classes, ~self.masked)

    def in_constraint(self) -> torch.Tensor:
        """(P,) log 1[x in C], read at the unmasked classes; meaningful on complete rows."""
        inside = self.constraint.contains(self.classes)
        zero = torch.zeros(inside.shape, dtype=torch.float64, device=inside.device)
        return torch.where(inside, zero, torch.full_like(zero, float("-inf")))

    def take(self, index: torch.Tensor) -> None:
        """Keep the rows `index` (with repetition): what resampling does to the particles."""
        self.tokens = self.tokens[index]
        self.classes = self.classes[index]
        self.log_p = self.log_p[index]
        self.log_q = self.log_q[index]
        if self.log_psi is not None:
            self.log_psi = self.log_psi[index]
        self.cache = {name: value[index] for name, value in self.cache.items()}


@dataclass
class Step:
    """One constrained denoising step from x_t to x_s (notes, section 2.3).

    unmask   (P, n) the positions U unmasked in this step
    valid    (P, n) bool, or None: rows may unmask different numbers of positions; the columns
             marked False are padding and unmask nothing
    tokens   (P, L) x_s
    classes  (P, L) classes of x_s at its unmasked positions
    drawn    (P, n) the classes drawn at U
    extra    what the twist's proposal wants to remember for the weight
    """

    unmask: torch.Tensor
    valid: Optional[torch.Tensor]
    tokens: torch.Tensor
    classes: torch.Tensor
    drawn: torch.Tensor
    extra: Dict[str, torch.Tensor] = field(default_factory=dict)

    def prediction_given_step(
        self, state: State, log_q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The prediction at x_t with the tokens of this step also fixed: the factorised
        distribution behind p(x_0 in C | x_t, x_0,U = x_s,U), the denominator of eq. (10)."""
        marginals = state.log_q if log_q is None else log_q
        return fix_unmasked(marginals, self.classes, self.tokens != state.mask_id)
