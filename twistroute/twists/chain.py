"""The chain twist and proposal (notes, eq. 11 to 13): an intermediate twist that is not
mean-field.

With mu_i(a) = p_theta(x_0,i = a | x_t) and a fixed pair table K(a, b) > 0 (an estimate of the
probability that class b follows class a at adjacent positions), let Q_i(a, b) =
u_i(a) K(a, b) v_i(b) be the joint whose two marginals are mu_i and mu_{i+1} (eq. 11, found by
`matching_sweeps` rounds of row and column scaling). The chain law r(. | x_t) starts from mu_1
and moves from position i to i + 1 by Q_i(a, b) / mu_i(a) (eq. 12). It has the model's marginal
at every position and the table's dependence between neighbours. The chain twist is the chain
law's mass on C, and the chain proposal draws the completion from r(. | x_t, C) (eq. 13).

Automaton constraints only: the mass of C under a chain is one forward pass over pairs
(class, automaton state), cost O(L A |E|).

This module gives the chain twist the interface of `twists.base`. The chain law and its passes
(the matching of eq. 11, the forward pass, the sampler of eq. 13) are in `twists.chain_law`.

Options
    matching_sweeps  0 uses the first-order ratio formula Q_i(a, b) proportional to
                     mu_{i+1}(b) K(a, b) / sum_a' mu_i(a') K(a', b) instead of eq. (11).
    product_share    e in (0, 1]: the twist is (1 - e) r(C | x_t) + e p_theta(C | x_t), which is
                     positive wherever the product twist is.
    proposal         "chain" (eq. 13) or "product" (eq. 9: the chain is then used as a twist only).
    proposal_product_share
                     d in [0, 1): each particle draws its completion from the product proposal
                     with probability d; the weight uses the mixture's density.
    temperature      tau in (0, 1]: the chain inside the twist is built on tempered marginals.
                     The proposal is not tempered.
    power            beta in (0, 1]: the twist is psi^beta.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

from ..constraints.base import Constraint
from ..state import State, Step
from .base import NEG_INF, Twist, drawn_log_prob, weight_from_ratio
from .chain_law import ChainLaw
from .product import log_product_twist


class ChainTwist(Twist):
    def __init__(
        self,
        automaton: Any,
        pair_table: torch.Tensor,
        *,
        active: Optional[torch.Tensor] = None,
        matching_sweeps: int = 5,
        product_share: float = 0.05,
        proposal: str = "chain",
        proposal_product_share: float = 0.0,
        temperature: float = 1.0,
        power: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """automaton: the `twistroute.compile.WDFA` of the constraint; active: (L,) bool, the
        positions the automaton reads (None = all); pair_table: (A, A), rows sum to one."""
        if proposal not in ("chain", "product"):
            raise ValueError("proposal must be 'chain' or 'product'")
        if not 0.0 <= proposal_product_share < 1.0:
            raise ValueError("proposal_product_share must lie in [0, 1)")
        if proposal == "product" and proposal_product_share > 0:
            raise ValueError("proposal_product_share needs the chain proposal")
        if not 0.0 < power <= 1.0:
            raise ValueError("power must lie in (0, 1]")
        if not 0.0 < product_share <= 1.0:
            raise ValueError("product_share must lie in (0, 1]")
        if not 0.0 < temperature <= 1.0:
            raise ValueError("temperature must lie in (0, 1]")
        self.law = ChainLaw(
            automaton, pair_table, active=active, matching_sweeps=matching_sweeps, device=device
        )
        self.matching_sweeps = int(matching_sweeps)
        self.product_share = float(product_share)
        self.temperature = float(temperature)
        self.proposal = proposal
        self.proposal_product_share = float(proposal_product_share)
        self.power = float(power)

    @classmethod
    def for_constraint(
        cls, constraint: Constraint, pair_table: torch.Tensor, **options: Any
    ) -> "ChainTwist":
        """The chain twist of an automaton constraint built by `twistroute.constraints`."""
        automaton = constraint.reference
        if not hasattr(automaton, "edges"):
            raise ValueError(
                f"the chain twist needs an automaton constraint, not {constraint.family!r}"
            )
        return cls(automaton, pair_table, **options)

    def __repr__(self) -> str:
        return (
            f"ChainTwist(matching_sweeps={self.matching_sweeps}, proposal={self.proposal!r}, "
            f"proposal_product_share={self.proposal_product_share}, power={self.power})"
        )

    def log_psi(self, state: State) -> torch.Tensor:
        log_product = log_product_twist(state)
        state.cache["log_product"] = log_product
        if self.product_share >= 1.0:
            return self.power * log_product
        prediction = state.prediction()
        if self.temperature != 1.0:
            prediction = torch.log_softmax(self.temperature * prediction.double(), -1)
        log_chain = self.law.log_mass(prediction) + math.log1p(-self.product_share)
        return self.power * torch.logaddexp(log_chain, log_product + math.log(self.product_share))

    def complete(
        self,
        state: State,
        generator: Optional[torch.Generator],
        unmask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prediction = state.prediction()
        if self.proposal == "product":
            return state.constraint.sample(prediction, generator), {}
        completion, log_chain_mass = self.law.sample(
            prediction, generator
        )  # x ~ r(. | x_t, C); log r(C | x_t)
        if self.proposal_product_share > 0:
            n_rows = state.tokens.shape[0]
            from_product = torch.rand(n_rows, generator=generator, device=state.tokens.device)
            from_product = from_product < self.proposal_product_share
            product_draw = state.constraint.sample(prediction, generator)
            completion = torch.where(from_product[:, None], product_draw, completion)
        return completion, {"prediction": prediction, "log_chain_mass": log_chain_mass}

    def log_weight(
        self, state: State, step: Step, log_psi_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_fixed = state.constraint.log_prob(step.prediction_given_step(state))
        log_product = state.cache["log_product"]
        if self.proposal == "product":
            return weight_from_ratio(state, log_psi_next, log_fixed - log_product), log_fixed
        # M / kappa = r(classes drawn at U, C | x_t) / r(C | x_t)
        #             / prod_{i in U} mu_i(class drawn at i)
        log_joint = self.law.log_mass_with(
            step.extra["prediction"], step.unmask, step.drawn, step.valid
        )
        log_ratio = log_joint - step.extra["log_chain_mass"] - drawn_log_prob(state, step)
        log_ratio = torch.nan_to_num(log_ratio, nan=NEG_INF, neginf=NEG_INF)
        share = self.proposal_product_share
        if share > 0:
            product_ratio = torch.nan_to_num(
                log_fixed.double() - log_product.double(), nan=NEG_INF, neginf=NEG_INF
            )
            log_ratio = torch.logaddexp(
                math.log1p(-share) + log_ratio, math.log(share) + product_ratio
            )
        return weight_from_ratio(state, log_psi_next, log_ratio), log_fixed
