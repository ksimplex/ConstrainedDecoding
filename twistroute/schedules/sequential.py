"""Schedules that rank the masked positions and unmask the best (notes, section 4).

With `per_step = 1` these are sequential orders: left to right, by the model's confidence
under the constraint (the rule LLaDA and Dream ship with, read from the constrained
prediction as in Dang and Ermon, arXiv:2607.07026, section 5.1), and `Sum`, the
constraint-aware order of this work. At the last step every schedule unmasks whatever is
left, so `n_steps` is a cap on the number of steps.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

from ..state import State
from .base import Schedule, Unmasking, constrained_marginals, entropy, forced, pad

NEG_INF = float("-inf")
Scores = Tuple[torch.Tensor, Optional[torch.Tensor]]


class Ranked(Schedule):
    """Unmask the `per_step` masked positions with the highest score.

    block         None, or a block length: only the left-most block that still has masked
                  positions is eligible (LLaDA's semi-autoregressive decoding)
    forced_first  positions whose token the constraint determines go before all others
    A subclass gives `scores(state)`: (P, L) scores, higher first, and the constrained
    marginals if it computed them (else None).
    """

    def __init__(
        self, per_step: int = 1, block: Optional[int] = None, forced_first: bool = False
    ) -> None:
        if per_step < 1 or (block is not None and block < 1):
            raise ValueError("per_step must be >= 1 and block None or >= 1")
        self.per_step = int(per_step)
        self.block = block
        self.forced_first = forced_first

    def scores(self, state: State) -> Scores:
        raise NotImplementedError

    @torch.no_grad()
    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        masked = state.masked
        if step >= n_steps - 1:
            return pad(masked)
        scores, marginals = self.scores(state)
        eligible = masked
        if self.block is not None:
            length = masked.shape[1]
            index = torch.arange(length, device=masked.device)
            first_masked = (
                torch.where(masked, index[None], torch.full_like(index, length)[None])
                .min(-1)
                .values
            )
            eligible = masked & (
                (index[None] // self.block) == (first_masked // self.block)[:, None]
            )
        if self.forced_first:
            if marginals is None:
                marginals = constrained_marginals(state)
            scores = torch.where(
                forced(state, marginals), torch.full_like(scores, float("inf")), scores
            )
        scores = torch.where(eligible, scores, torch.full_like(scores, NEG_INF))
        top = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, : self.per_step]
        chosen = torch.zeros_like(masked).scatter(-1, top, torch.ones_like(top, dtype=torch.bool))
        return pad(chosen & eligible)


class LeftToRight(Ranked):
    """The left-most masked position first."""

    def scores(self, state: State) -> Scores:
        n_rows, length = state.tokens.shape
        return -torch.arange(length, device=state.tokens.device).double().expand(
            n_rows, length
        ), None


class FixedOrder(Ranked):
    """The positions in a given order: `order[0]` first. Used for the fixed random orders of the
    notes' Table 3."""

    def __init__(self, order: Sequence[int]) -> None:
        super().__init__(1, None, False)
        self.order = [int(position) for position in order]

    def scores(self, state: State) -> Scores:
        n_rows, length = state.tokens.shape
        if sorted(self.order) != list(range(length)):
            raise ValueError("order must be a permutation of the positions")
        rank = torch.empty(length, dtype=torch.float64, device=state.tokens.device)
        rank[torch.tensor(self.order, device=state.tokens.device)] = torch.arange(
            length, dtype=torch.float64
        )
        return -rank.expand(n_rows, length), None


class Confidence(Ranked):
    """The position the model is most confident about.

    rule  "token"    highest probability of the most likely token under the constrained
                     prediction, max_v p(v) mu(class of v) / q(class of v) (LLaDA's rule)
          "class"    highest probability of the most likely class, max_a mu(a)
          "entropy"  lowest entropy of the constrained class marginal (Dream's rule)
          "model"    highest probability of the model's most likely token, constraint ignored
                     (the order of the unconstrained decoder)
    where p is the model's token prediction, q its class marginal and mu the constrained class
    marginal at the position.
    """

    RULES = ("token", "class", "entropy", "model")

    def __init__(
        self,
        rule: str = "token",
        per_step: int = 1,
        block: Optional[int] = None,
        forced_first: bool = False,
    ) -> None:
        super().__init__(per_step, block, forced_first)
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}")
        self.rule = rule

    def scores(self, state: State) -> Scores:
        if self.rule == "model":
            return state.log_p.max(-1).values.double(), None
        marginals = constrained_marginals(state)
        if self.rule == "entropy":
            return -entropy(marginals), marginals
        if self.rule == "class":
            return marginals.max(-1).values, marginals
        # log of mu / q per class, added to every token of the class
        log_ratio = (torch.log(marginals.clamp_min(1e-300)) - state.log_q.double()).float()
        log_ratio = torch.where(marginals > 0, log_ratio, torch.full_like(log_ratio, NEG_INF))
        class_of_token = state.token_classes.token_class.to(marginals.device)
        best = []
        for row in range(state.tokens.shape[0]):  # one particle at a time: (L, V) memory
            best.append(
                (state.log_p[row].float() + log_ratio[row][:, class_of_token]).max(-1).values
            )
        return torch.stack(best).double(), marginals


class Sum(Ranked):
    """The constraint-aware sequential order: unmask the position with the largest

        D_i + sum_j M(i, j),      j over the other masked positions,

    where D_i = KL(mu_i || q_i) is how far the constraint moves the model's prediction at i
    (mu the constrained marginal, q the model's), and M(i, j) is the estimated dependence
    between positions i and j under the model (`schedules.dependence`). The first term sends
    the sampler to where the constraint acts; the second to positions whose value the model's
    other predictions depend on. One position per step; forced positions first by default.

    forced_batch  None, or a tolerance in nats. A particle that has forced positions then
                  unmasks forced positions only, as many as keep the bound
                  -log(1 - sum_j (1 - p_j)) on the change in its log-weight within the
                  tolerance (p_j the model's probability of the forced token), at least one.
    """

    def __init__(
        self, dependence: Any, forced_first: bool = True, forced_batch: Optional[float] = None
    ) -> None:
        super().__init__(1, None, forced_first)
        self.dependence = dependence
        self.forced_batch = None if forced_batch is None else float(forced_batch)

    def scores(self, state: State) -> Scores:
        masked = state.masked
        marginals = constrained_marginals(state)
        model = torch.exp(state.log_q.double())
        moved = torch.xlogy(marginals, marginals) - marginals * torch.log(model.clamp_min(1e-300))
        distortion = moved.sum(-1).clamp_min(0.0)  # D_i
        distortion = torch.where(masked, distortion, torch.zeros_like(distortion))
        dependence = self.dependence.apply(masked, masked.double()).to(masked.device)
        return distortion + dependence, marginals

    def forced_set(self, state: State, marginals: torch.Tensor) -> torch.Tensor:
        """(P, L) bool: the forced positions to unmask together under `forced_batch`."""
        top = marginals.argmax(-1)
        is_forced = state.masked & forced(state, marginals)
        miss = 1.0 - torch.exp(state.log_q.double()).gather(-1, top[..., None]).squeeze(-1).clamp(
            0.0, 1.0
        )
        miss = torch.where(is_forced, miss, torch.full_like(miss, float("inf")))
        value, index = torch.sort(miss, dim=-1, stable=True)
        total = torch.cumsum(torch.where(torch.isfinite(value), value, torch.zeros_like(value)), -1)
        bound = -torch.log1p(-total.clamp(max=1.0 - 1e-300))
        bound = torch.where(total < 1.0, bound, torch.full_like(total, float("inf")))
        take = torch.isfinite(value) & (bound <= self.forced_batch)
        take[:, 0] = torch.isfinite(value[:, 0])
        return torch.zeros_like(is_forced).scatter(-1, index, take)

    @torch.no_grad()
    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        positions, valid = super().unmask(state, step, n_steps, generator)
        if self.forced_batch is None or step >= n_steps - 1:
            return positions, valid
        chosen = torch.zeros_like(state.masked).scatter(-1, positions, valid)
        batch = self.forced_set(state, constrained_marginals(state))
        return pad(torch.where(batch.any(-1)[:, None], batch, chosen))
