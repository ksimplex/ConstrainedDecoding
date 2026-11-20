"""The product twist and proposal (notes, eq. 9 and 10), from TWISTER (arXiv:2609.35609).

The constrained prediction at x_t is the mean-field prediction restricted to C (eq. 9). The
product twist is its normaliser,

    psi(x_t) = p_theta(x_0 in C | x_t) = sum_{x_0 in C} prod_i p_theta(x_0,i | x_t),

one call of `Constraint.log_prob`. The product proposal draws a completion from the
constrained prediction (`Constraint.sample`) and commits it on U. For this pair the weight is

    w(x_t, x_s) = p_theta(x_0 in C | x_s) / p_theta(x_0 in C | x_t, x_0,U = x_s,U).        (10)

Two options change the twist to counter hidden mass (section 2.3); both keep the sampler exact.

    temperature  tau in (0, 1]. The twist is computed with tempered marginals
                 q_i(a) proportional to p_theta(x_0,i = a | x_t)^tau at masked positions: a
                 flatter prediction, so the twist is less sure that a completion is impossible.
                 The proposal is the best one for that twist: classes at U are drawn from the
                 model's own marginals, the positions outside U follow q, all restricted to C.
    power        beta in (0, 1]. The twist is psi^beta.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..state import State, Step
from .base import Twist, weight_from_ratio


def tempered(log_q: torch.Tensor, temperature: float) -> torch.Tensor:
    """log of q_i(a) proportional to exp(log_q_i(a))^temperature, normalised over a."""
    if temperature == 1.0:
        return log_q
    return torch.log_softmax(temperature * log_q, -1)


def log_product_twist(state: State, temperature: float = 1.0) -> torch.Tensor:
    """(P,) log p(x_0 in C | x_t) under the (tempered) mean-field prediction; exactly
    log 1[x_0 in C] on complete rows."""
    log_prob = state.constraint.log_prob(state.prediction(tempered(state.log_q, temperature)))
    complete = state.complete
    if complete.any():
        log_prob = torch.where(complete, state.in_constraint().to(log_prob.dtype), log_prob)
    return log_prob


class ProductTwist(Twist):
    def __init__(self, temperature: float = 1.0, power: float = 1.0) -> None:
        if not 0.0 < temperature <= 1.0:
            raise ValueError("temperature must lie in (0, 1]")
        if not 0.0 < power <= 1.0:
            raise ValueError("power must lie in (0, 1]")
        self.temperature = float(temperature)
        self.power = float(power)
        self.needs_unmask_set = self.temperature != 1.0

    def __repr__(self) -> str:
        return f"ProductTwist(temperature={self.temperature}, power={self.power})"

    def log_psi(self, state: State) -> torch.Tensor:
        log_product = log_product_twist(state, self.temperature)
        state.cache["log_product"] = log_product
        return self.power * log_product

    def complete(
        self,
        state: State,
        generator: Optional[torch.Generator],
        unmask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.temperature == 1.0:
            return state.constraint.sample(state.prediction(), generator), {}
        # Tempered twist: the model's marginals at U, the tempered ones at the other masked
        # positions.
        in_unmask = torch.zeros(state.tokens.shape, dtype=torch.bool, device=state.tokens.device)
        in_unmask.scatter_(
            1, unmask, torch.ones_like(unmask, dtype=torch.bool) if valid is None else valid
        )
        log_q_tempered = tempered(state.log_q, self.temperature)
        mixed = state.prediction(torch.where(in_unmask[..., None], state.log_q, log_q_tempered))
        return state.constraint.sample(mixed, generator), {
            "mixed": mixed,
            "tempered": log_q_tempered,
        }

    def class_marginals(self, state: State, position: torch.Tensor) -> torch.Tensor:
        log_q = state.log_q
        if self.temperature != 1.0:
            at_position = torch.zeros(state.tokens.shape, dtype=torch.bool, device=position.device)
            at_position.scatter_(1, position[:, None], True)
            log_q = torch.where(at_position[..., None], log_q, tempered(log_q, self.temperature))
        marginals = state.constraint.marginals(state.prediction(log_q))
        return marginals[torch.arange(len(position), device=position.device), position]

    def log_weight(
        self, state: State, step: Step, log_psi_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        constraint = state.constraint
        if self.temperature == 1.0:
            # eq. (10); with power beta the factor psi(x_t)^(1 - beta) is left over.
            log_fixed = constraint.log_prob(step.prediction_given_step(state))
            if self.power == 1.0:
                return log_psi_next - log_fixed, log_fixed
            return (1.0 - self.power) * state.cache[
                "log_product"
            ] + log_psi_next - log_fixed, log_fixed
        # M / kappa = q(C | x_t, x_s,U) / [mass of C under the mixed prediction the draw came from]
        log_fixed = constraint.log_prob(step.prediction_given_step(state, step.extra["tempered"]))
        log_ratio = log_fixed - constraint.log_prob(step.extra["mixed"])
        return weight_from_ratio(state, log_psi_next, log_ratio), log_fixed
