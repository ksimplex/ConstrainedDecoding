"""Exact laws at a short length (notes, Table 3 and Figure 1).

On a model whose joint law is known (the oracles of `twistroute.models`) and at a length where
every sequence can be listed, the law of the sampler's output is computed exactly and compared
with the target P(x_0 | x_0 in C):

    target_law         pi(x) = P(x) 1[x in C] / P(C)
    one_particle_law   q(x): the law of the sequence one particle ends at (no weights), for a
                       sequential schedule and any twist's proposal; uses the same `Schedule`
                       and `Twist` objects as the sampler
    output_law         q_K(x): the law of the sequence returned by K particles without
                       resampling, when the schedule is deterministic
    kl                 KL(p || q) in nats

Sequences are listed in the order of `all_sequences`: the first position is the most
significant digit.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from ..constraints.base import Constraint
from ..sampler import model_prediction
from ..schedules.base import RandomOrder, Schedule
from ..twists.base import Twist
from ..twists.product import ProductTwist


def all_sequences(n_tokens: int, length: int) -> torch.Tensor:
    """(n_tokens^length, length) every sequence of tokens 0 .. n_tokens - 1."""
    return torch.cartesian_prod(*[torch.arange(n_tokens)] * length).reshape(-1, length)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats; infinite if p puts mass where q has none."""
    support = p > 0
    if np.any(q[support] <= 0):
        return float("inf")
    return float((p[support] * np.log(p[support] / q[support])).sum())


def target_law(model: Any, token_classes: Any, constraint: Constraint, length: int) -> np.ndarray:
    """pi over `all_sequences`: the model's joint law restricted to C and normalised. Needs a
    model with `.log_joint(tokens)` (an oracle)."""
    sequences = all_sequences(model.S, length)
    inside = constraint.contains(token_classes.token_class[sequences]).numpy()
    joint = np.where(inside, np.exp(model.log_joint(sequences).numpy()), 0.0)
    return joint / joint.sum()


def log_prob_of_constraint(
    model: Any, token_classes: Any, constraint: Constraint, length: int
) -> float:
    """log P(x_0 in C) under the model's joint law, by listing every sequence (an oracle)."""
    sequences = all_sequences(model.S, length)
    inside = constraint.contains(token_classes.token_class[sequences])
    return float(torch.logsumexp(model.log_joint(sequences)[inside].double(), 0))


def _merge(
    tokens: torch.Tensor, mass: torch.Tensor, base: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add up the mass of identical states."""
    weights = base ** torch.arange(tokens.shape[1] - 1, -1, -1)
    codes, inverse = torch.unique((tokens * weights).sum(-1), return_inverse=True)
    merged = torch.zeros(len(codes), dtype=torch.float64).index_add(0, inverse, mass)
    first = torch.zeros(len(codes), dtype=torch.long).scatter(
        0, inverse, torch.arange(len(inverse))
    )
    return tokens[first], merged


@torch.no_grad()
def one_particle_law(
    model: Any,
    token_classes: Any,
    constraint: Constraint,
    length: int,
    *,
    schedule: Schedule,
    twist: Optional[Twist] = None,
    chunk: int = 20000,
) -> np.ndarray:
    """q over `all_sequences`: the exact law of the sequence one particle ends at when it
    follows the twist's proposal and never uses its weight (Dang and Ermon's sampler when the
    twist is the product twist).

    The schedule must unmask one position per step. It is either deterministic (the position
    is a function of the state) or `RandomOrder`, for which the law is averaged over the
    uniformly chosen position. Every reachable partly masked state is visited once.
    """
    twist = ProductTwist() if twist is None else twist
    n_tokens = model.S
    mask = model.mask_id
    tokens = torch.full((1, length), mask, dtype=torch.long)
    mass = torch.ones(1, dtype=torch.float64)
    for step in range(length):
        next_tokens = []
        next_mass = []
        for low in range(0, len(tokens), chunk):
            part = tokens[low : low + chunk]
            part_mass = mass[low : low + chunk]
            classes = token_classes.token_class[
                torch.where(part == mask, torch.zeros_like(part), part)
            ]
            state = model_prediction(model, token_classes, constraint, part, classes)
            if isinstance(schedule, RandomOrder):
                rows, positions = state.masked.nonzero(as_tuple=True)
                share = 1.0 / state.masked.sum(-1)[rows]
            else:
                unmask, valid = schedule.unmask(state, step, length + 1, None)
                if unmask.shape[1] != 1 or (valid is not None and not bool(valid.all())):
                    raise ValueError(
                        "the exact law needs a schedule that unmasks one position per step"
                    )
                rows = torch.arange(len(part))
                positions = unmask[:, 0]
                share = torch.ones(len(part), dtype=torch.float64)
            sub = _take(state, rows)
            class_law = twist.class_marginals(sub, positions)  # (n, A)
            # token within the class, from the model's prediction at the position
            log_p = sub.log_p[torch.arange(len(rows)), positions].double()  # (n, V)
            log_q = sub.log_q[torch.arange(len(rows)), positions].double()  # (n, A)
            for token in range(n_tokens):
                cls = int(token_classes.token_class[token])
                within = torch.exp(log_p[:, token] - log_q[:, cls])
                moved = part_mass[rows] * share * class_law[:, cls] * within
                keep = moved > 0
                new = part[rows][keep].clone()
                new[torch.arange(int(keep.sum())), positions[keep]] = token
                next_tokens.append(new)
                next_mass.append(moved[keep])
        tokens, mass = _merge(torch.cat(next_tokens), torch.cat(next_mass), n_tokens + 1)
    law = np.zeros(n_tokens**length)
    index = (tokens * n_tokens ** torch.arange(length - 1, -1, -1)).sum(-1).numpy()
    np.add.at(law, index, mass.numpy())
    # Arithmetic leaves about 1e-17 of mass on sequences outside C and about 1e-7 of total mass
    # unaccounted for (the model's predictions are single precision). Remove both.
    inside = constraint.contains(token_classes.token_class[all_sequences(n_tokens, length)]).numpy()
    if law[~inside].sum() > 1e-9 or abs(law.sum() - 1.0) > 1e-5:
        raise ArithmeticError("the one-particle law lost mass or left the constraint")
    law = np.where(inside, law, 0.0)
    return law / law.sum()


def _take(state: Any, rows: torch.Tensor) -> Any:
    """The rows `rows` of a state, as a new state."""
    from ..state import State

    return State(
        state.tokens[rows],
        state.classes[rows],
        state.log_p[rows],
        state.log_q[rows],
        state.mask_id,
        state.constraint,
        state.token_classes,
    )


def output_law(
    one_particle: np.ndarray,
    target: np.ndarray,
    particles: int,
    *,
    n_draws: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """q_K over `all_sequences`: the law of the sequence returned by K particles without
    resampling, for a deterministic schedule.

    Each particle is an independent draw from q with weight w = pi / q, and one particle is
    returned in proportion to its weight, so

        q_K(x) = K q(x) w(x) E[ 1 / (w(x) + S) ],

    with S the sum of the weights of the other K - 1 particles. The sum over x is exact; the
    expectation over S is a mean over `n_draws` draws of K - 1 weights.
    """
    if particles == 1:
        return one_particle.copy()
    rng = np.random.default_rng(0) if rng is None else rng
    support = one_particle > 0
    weight = np.where(support, target / np.where(support, one_particle, 1.0), 0.0)
    others = weight[rng.choice(len(weight), size=(n_draws, particles - 1), p=one_particle)].sum(1)
    law = np.zeros(len(weight))
    live = np.nonzero(weight > 0)[0]
    block = max(1, 40_000_000 // n_draws)  # rows per block: about 320 MB of doubles
    for low in range(0, len(live), block):
        rows = live[low : low + block]
        inverse = (1.0 / (weight[rows][:, None] + others[None])).mean(1)
        law[rows] = particles * one_particle[rows] * weight[rows] * inverse
    return law / law.sum()
