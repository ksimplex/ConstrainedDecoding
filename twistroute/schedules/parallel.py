"""Schedules that unmask several positions per step (notes, section 4).

    Blocks            the decoder LLaDA ships with and Dang and Ermon (arXiv:2607.07026) use:
                      blocks from left to right, a fixed number of positions per step inside the
                      current block, most confident first.
    BudgetedCoReveal  the constraint-aware parallel rule of this work, as it stands.

`RandomOrder` with fewer steps than positions is the third: a fixed number per step, uniformly
at random.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..state import State
from .base import FromPolicy, Schedule, Unmasking
from .sequential import Confidence


class Blocks(Confidence):
    """`per_step` positions per step, the most confident first, inside the left-most block of
    `block` positions that still has masked ones."""

    def __init__(
        self, block: int, per_step: int = 2, rule: str = "token", forced_first: bool = False
    ) -> None:
        super().__init__(rule, per_step, block, forced_first)


class BudgetedCoReveal(Schedule):
    """Unmask positions together as long as their estimated dependence stays within a budget.

    Masked positions are considered in the order of `schedules.sequential.Sum`. A position
    joins the step's set while the average load of the positions in the set, its own included,
    stays at most budget / L; its load is its summed dependence M(i, j) on the positions
    already in the set. Every position is unmasked in exactly one step, so the loads of a whole
    run sum to at most `budget` nats: an estimate of the KL cost of the parallel steps.

    pull_in       when > 0: for up to this many positions per step on which the constraint
                  acts, the positions that this unmasking will force are added to the same step
                  if their load fits (one extra constraint pass each, no model call).
    forced_batch  as in `Sum`.

    The rule is implemented by `twistroute.decode.policy.CouplingPolicy`; this class gives it
    the notes' vocabulary. Section 4 of the notes is where its replacement will be described.
    """

    def __init__(
        self, dependence: Any, budget: float, pull_in: int = 8, forced_batch: Optional[float] = None
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self.dependence = dependence
        self.budget = float(budget)
        self.pull_in = int(pull_in)
        self.forced_batch = forced_batch
        self._rule: Optional[FromPolicy] = None
        self._length: Optional[int] = None

    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        length = state.tokens.shape[1]
        if self._rule is None or self._length != length:
            from ..decode.policy import CouplingPolicy

            policy = CouplingPolicy(
                self.dependence,
                0.0,
                priority="sum",
                mixed=False,
                rate=self.budget / length,
                pull_forced=self.pull_in,
                forced_batch=self.forced_batch,
            )
            self._rule = FromPolicy(policy)
            self._length = length
        return self._rule.unmask(state, step, n_steps, generator)
