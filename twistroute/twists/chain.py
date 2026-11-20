"""The chain twist and proposal (notes, section 2.3, eq. 11 to 13): an intermediate twist that
is not mean-field.

Notation of the notes: mu_i(a) = p_theta(x_0,i = a | x_t) is the model's marginal at position i
and K(a, b) > 0 a fixed pair table. For each adjacent pair, Q_i(a, b) = u_i(a) K(a, b) v_i(b) is
the joint whose two marginals are mu_i and mu_{i+1}. Then

    (11)  chain prediction      r(x_0 | x_t) = mu_1(x_0,1) prod_i Q_i(x_0,i, x_0,i+1) / mu_i(x_0,i)
    (12)  constrained chain prediction r(x_0 | x_t, C), with normaliser r(x_0 in C | x_t)
          chain twist           psi(x_t) = r(x_0 in C | x_t)
          chain proposal        draw a completion from r(. | x_t, C) and commit it on U
    (13)  weight of a step      eq. (5) for this twist and proposal

Where each is computed
    Q_i / mu_i            `ChainLaw.transitions`       (how Q_i is found: see `matching_sweeps`)
    r(x_0 in C | x_t)     `ChainLaw.log_mass`          a forward pass over (class, automaton state)
    draw from eq. (12)    `ChainLaw.sample`            a backward pass, then a forward sweep
    psi(x_t)              `ChainTwist.log_psi`
    eq. (13)              `ChainTwist.log_weight`

Automaton constraints only: the passes run over pairs (class, automaton state).

Differences from the notes, all optional except the first
    product_share    e in [0, 1]. The twist is (1 - e) r(C | x_t) + e p_theta(C | x_t), a mixture
                     with the product twist of eq. (9). The default is 0.05, NOT the notes' twist;
                     e = 0 is the notes' twist. With e > 0 the weight is eq. (5) for the mixed
                     twist and differs from eq. (13) as printed (by a few percent at e = 0.05).
    matching_sweeps  m rounds of row and column scaling find Q_i. The row marginal mu_i is fitted
                     exactly, the column marginal mu_{i+1} up to the error left after m rounds
                     (about 1e-4 at the default m = 5), so "the second factor of eq. (13) equals 1
                     when one position is unmasked" holds up to that error. The weight never uses
                     that simplification. m = 0 is not eq. (11): it uses the closed formula
                     Q_i(a, b) / mu_i(a) proportional to
                     mu_{i+1}(b) K(a, b) / sum_a' mu_i(a') K(a', b),
                     the "unmatched chain" of Figure 2, whose marginals are not the model's.
    proposal         "chain" (the notes) or "product" (the completion is drawn from the constrained
                     prediction of eq. 9: the chain is then a twist only).
    proposal_product_share
                     d in [0, 1): each particle draws its completion from the product proposal
                     with probability d; the weight uses the mixture's density.
    temperature      tau in (0, 1]: the chain inside the twist is built on tempered marginals.
                     The proposal is not tempered.
    power            beta in (0, 1]: the twist is psi^beta.

Any of these keeps the sampler exact: the weight is always eq. (5) for the twist and proposal
actually used.

Cost per particle and step, with A classes, S automaton states and m sweeps: the matching is
O(m L A^2); each pass (`log_mass`, `log_mass_with`, the backward pass of `sample`) is O(L A^2 S)
time. One step calls the matching and a pass three times (the twist at x_s, the draw, the fixed
mass), against O(L |E|) for the product twist.
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
    """The chain twist with its proposal; see the module docstring for the options."""

    def __init__(
        self,
        automaton: Any,
        pair_table: torch.Tensor,
        *,
        active: Optional[torch.Tensor] = None,
        matching_sweeps: int = 5,
        # Differs from the notes on purpose: the notes' chain twist is product_share = 0. The 5%
        # share of the product twist is kept as a safeguard (the twist then stays positive
        # wherever the product twist is), and the weight is eq. (5) for this mixed twist.
        product_share: float = 0.05,
        proposal: str = "chain",
        proposal_product_share: float = 0.0,
        temperature: float = 1.0,
        power: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """automaton: the `twistroute.compile.WDFA` of the constraint; active: (L,) bool, the
        positions the automaton reads (None = all), needed for a constraint on a subset of
        positions; pair_table: (A, A) with positive entries (`twists.tables`). With
        matching_sweeps > 0 the scale of each row of the table drops out; with 0 sweeps it does
        not, and rows should sum to one."""
        if proposal not in ("chain", "product"):
            raise ValueError("proposal must be 'chain' or 'product'")
        if not 0.0 <= proposal_product_share < 1.0:
            raise ValueError("proposal_product_share must lie in [0, 1)")
        if proposal == "product" and proposal_product_share > 0:
            raise ValueError("proposal_product_share needs the chain proposal")
        if not 0.0 < power <= 1.0:
            raise ValueError("power must lie in (0, 1]")
        if not 0.0 <= product_share <= 1.0:
            raise ValueError("product_share must lie in [0, 1]")
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
        """The chain twist of an automaton constraint built by `twistroute.constraints`
        (one that reads every position; for a constraint on a subset of positions call the
        constructor with `active`)."""
        automaton = constraint.reference
        if not hasattr(automaton, "edges"):
            raise ValueError(
                f"the chain twist needs an automaton constraint, not {constraint.family!r}"
            )
        return cls(automaton, pair_table, **options)

    def __repr__(self) -> str:
        return (
            f"ChainTwist(matching_sweeps={self.matching_sweeps}, "
            f"product_share={self.product_share}, proposal={self.proposal!r}, "
            f"proposal_product_share={self.proposal_product_share}, "
            f"temperature={self.temperature}, power={self.power})"
        )

    def log_psi(self, state: State) -> torch.Tensor:
        """log psi(x_t) = beta log[(1 - e) r(x_0 in C | x_t) + e p_theta(x_0 in C | x_t)]."""
        log_product = log_product_twist(state)
        state.cache["log_product"] = log_product  # reused by `log_weight`
        if self.product_share >= 1.0:
            return self.power * log_product
        prediction = state.prediction()
        if self.temperature != 1.0:
            prediction = torch.log_softmax(self.temperature * prediction.double(), -1)
        if self.product_share == 0.0:  # the notes' twist
            return self.power * self.law.log_mass(prediction)
        log_chain = self.law.log_mass(prediction) + math.log1p(-self.product_share)
        return self.power * torch.logaddexp(log_chain, log_product + math.log(self.product_share))

    def draw_completion(
        self,
        state: State,
        generator: Optional[torch.Generator],
        unmask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """A completion from the constrained chain prediction r(. | x_t, C) of eq. (12). Keeps
        log r(x_0 in C | x_t), which the same pass returns, for the weight."""
        prediction = state.prediction()
        if self.proposal == "product":
            return state.constraint.sample(prediction, generator), {}
        completion, log_chain_mass = self.law.sample(prediction, generator)
        if self.proposal_product_share > 0:
            n_rows = state.tokens.shape[0]
            from_product = torch.rand(n_rows, generator=generator, device=state.tokens.device)
            from_product = from_product < self.proposal_product_share
            product_draw = state.constraint.sample(prediction, generator)
            completion = torch.where(from_product[:, None], product_draw, completion)
        return completion, {"prediction": prediction, "log_chain_mass": log_chain_mass}

    def class_marginals(self, state: State, position: torch.Tensor) -> torch.Tensor:
        """r(x_0,i = a | x_t, C) at one position per row (mixed with the product proposal's
        marginal when `proposal_product_share` > 0)."""
        prediction = state.prediction()
        rows = torch.arange(len(position), device=position.device)
        product = state.constraint.marginals(prediction)[rows, position]
        if self.proposal == "product":
            return product
        chain = self.law.marginals(prediction)[rows, position]
        return (1.0 - self.proposal_product_share) * chain + self.proposal_product_share * product

    def log_weight(
        self, state: State, step: Step, log_psi_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The weight of a chain-proposal step: eq. (13) of the notes.

        Computed as eq. (5), w = psi(x_s) / psi(x_t) / (M / kappa), with

            M / kappa = r(x_0,U = x_s,U, x_0 in C | x_t) / r(x_0 in C | x_t)
                        / prod_{i in U} mu_i(x_s,i).

        With psi = r(x_0 in C | .) this is eq. (13): psi(x_t) cancels r(x_0 in C | x_t), and
        r(x_U, C | x_t) = r(C | x_t, x_U) r(x_U | x_t) splits the rest into the two factors of
        eq. (13). The two factors are never formed separately, and the general form stays right
        when the twist is mixed with the product twist or the matching is not converged.

        Also returns log p_theta(x_0 in C | x_t, x_0,U = x_s,U) (the denominator of eq. 10),
        which the sampler uses only to count support violations.
        """
        log_fixed = state.constraint.log_prob(step.prediction_given_step(state))
        log_product = state.cache["log_product"]
        if self.proposal == "product":
            return weight_from_ratio(state, log_psi_next, log_fixed - log_product), log_fixed
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
