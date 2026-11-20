"""Twists and proposals (notes, section 2.3).

A constrained sampler is given by two choices at every partly masked sequence x_t:

    twist     psi(x_t) >= 0, an estimate of the probability that a run continued from x_t ends
              in C, with psi(x_0) = 1[x_0 in C] on full sequences;
    proposal  M(x_s | x_t), the distribution the next state is drawn from.

The weight of a step is eq. (5),

    w(x_t, x_s) = kappa(x_s | x_t) psi(x_s) / (psi(x_t) M(x_s | x_t)),

with kappa the unconstrained step of eq. (4). Every proposal here draws the unmasking set U
from the same schedule as kappa, so the schedule's probability cancels in the ratio and the
weight depends only on how the classes at U were drawn.

A `Twist` object is a twist together with its proposal. The sampler calls, in this order:

    draw_completion(state, generator, unmask, valid)
                                             draw a full class sequence (the notes' x-hat_0) from
                                             the proposal's law; the sampler commits it on U
    log_psi(next_state)                      log psi(x_s), after the model has been called at x_s
    log_weight(state, step, log_psi_next)    log w(x_t, x_s)

The sampler takes care of psi(x_0) = 1[x_0 in C] at the last step. A twist needs nothing else:
`state.cache` is the twist's own scratch space (the sampler only carries it through resampling).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..state import State, Step

NEG_INF = float("-inf")


class Twist:
    """Interface of a twist with its proposal."""

    #: True when the proposal's law depends on the unmasking set U, so U must be chosen first.
    #: Otherwise the completion is drawn first; U is chosen without seeing it either way.
    needs_unmask_set: bool = False

    def log_psi(self, state: State) -> torch.Tensor:
        """(P,) log psi(x_t). May store per-particle tensors in `state.cache` for its own
        `log_weight`; the sampler does not read them."""
        raise NotImplementedError

    def draw_completion(
        self,
        state: State,
        generator: Optional[torch.Generator],
        unmask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Draw a class sequence (P, L) from the proposal's law of x_0 given x_t. Returns it with
        whatever `log_weight` needs later (stored in `Step.extra`)."""
        raise NotImplementedError

    def class_marginals(self, state: State, position: torch.Tensor) -> torch.Tensor:
        """(P, A): the proposal's law of the class at `position` (P,), one position per row, when
        that position alone is unmasked. Used by the exact evaluation, not by the sampler."""
        raise NotImplementedError

    def log_weight(
        self, state: State, step: Step, log_psi_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(P,) log w(x_t, x_s) of eq. (5), and (P,) log p(x_0 in C | x_t, x_0,U = x_s,U), which
        the sampler uses to check that the proposal stayed inside the constraint's support."""
        raise NotImplementedError


def weight_from_ratio(
    state: State, log_psi_next: torch.Tensor, log_ratio: torch.Tensor
) -> torch.Tensor:
    """Eq. (5) in logs: log psi(x_s) - log psi(x_t) - log(M(x_s | x_t) / kappa(x_s | x_t)).

    A particle whose twist is already zero keeps weight zero whatever happens next.
    """
    log_w = log_psi_next - state.log_psi - log_ratio
    dead = torch.isneginf(state.log_psi) | torch.isnan(log_w)
    return torch.where(dead, torch.full_like(log_w, NEG_INF), log_w)


def drawn_log_prob(state: State, step: Step) -> torch.Tensor:
    """(P,) sum over U of log p_theta(class drawn at i | x_t): the class part of log kappa."""
    n_classes = state.log_q.shape[-1]
    rows = torch.gather(state.log_q, 1, step.unmask[..., None].expand(-1, -1, n_classes))
    picked = rows.gather(2, step.drawn[..., None]).squeeze(-1).double()
    if step.valid is not None:
        picked = torch.where(step.valid, picked, torch.zeros_like(picked))
    return picked.sum(-1)
