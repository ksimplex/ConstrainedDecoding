"""Unmasking schedules (notes, section 1: pi(U | x_t); section 4).

A schedule chooses, for every particle, the set U of masked positions that the next step
unmasks. It may look at x_t, at the model's prediction there and at the constraint, never at
the tokens about to be drawn. The same schedule defines the unconstrained process (eq. 4) and
is used by the constrained sampler, so its probability cancels in the weight (eq. 5).

    unmask(state, step, n_steps, generator) -> (positions (P, n), valid (P, n) bool or None)

Rows may unmask different numbers of positions: `positions` is padded to the largest, and
`valid` marks the real entries (None when every entry is real).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ..state import State

Unmasking = Tuple[torch.Tensor, Optional[torch.Tensor]]


class Schedule:
    """Interface of an unmasking schedule."""

    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        raise NotImplementedError


class RandomOrder(Schedule):
    """L / n_steps positions per step, uniformly at random among the masked ones: the schedule
    of MDLM-style samplers and of TWISTER's experiments. One position per step when
    n_steps = L."""

    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        masked = state.masked
        per_step = masked.shape[-1] // n_steps
        uniform = torch.rand(masked.shape, device=masked.device, generator=generator)
        uniform = torch.where(masked, uniform, torch.full_like(uniform, 2.0))
        return torch.topk(-uniform, per_step, dim=-1).indices, None


class FromPolicy(Schedule):
    """A schedule written against the earlier interface `policy.reveal(state, t, T, gen)` of
    `twistroute.decode.policy`."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        from ..decode.policy import RevealState

        view = RevealState(
            state.tokens,
            state.classes,
            state.log_q,
            state.mask_id,
            state.token_classes,
            state.constraint.route,
            lp=state.log_p,
        )
        return self.policy.reveal(view, step, n_steps, generator)


def as_schedule(schedule: Any) -> Schedule:
    """Accept a `Schedule`, or an object with the earlier `.reveal` interface."""
    if isinstance(schedule, Schedule):
        return schedule
    if hasattr(schedule, "reveal"):
        return FromPolicy(schedule)
    raise TypeError("a schedule needs .unmask(state, step, n_steps, generator)")


def pad(chosen: torch.Tensor) -> Unmasking:
    """(P, L) bool, the positions each row unmasks -> (positions, valid) padded to the largest
    row."""
    count = chosen.sum(-1)
    width = int(count.max().clamp_min(1))
    order = torch.argsort(chosen.to(torch.int8), dim=-1, descending=True, stable=True)
    valid = torch.arange(width, device=chosen.device)[None] < count[:, None]
    return order[:, :width], valid


def constrained_marginals(state: State) -> torch.Tensor:
    """(P, L, A) float64: p_theta(x_0,i = a | x_t, C) under the constrained prediction of eq. (9)
    (Proposition 1, item 1). One backward pass over the constraint; no model call."""
    return state.constraint.marginals(state.prediction())


def entropy(marginals: torch.Tensor) -> torch.Tensor:
    """(P, L) entropy in nats of each position's marginal."""
    return -torch.xlogy(marginals, marginals).sum(-1)


def forced(state: State, marginals: torch.Tensor) -> torch.Tensor:
    """(P, L) bool: positions whose token the constraint determines. The constrained marginal
    puts all its mass on one class and that class holds a single token, so unmasking the
    position asks nothing of the model."""
    class_sizes = (state.token_classes.onehot > 0).sum(0).to(marginals.device)
    return (entropy(marginals) < 1e-9) & (class_sizes[marginals.argmax(-1)] == 1)
