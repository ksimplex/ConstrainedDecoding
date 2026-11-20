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
