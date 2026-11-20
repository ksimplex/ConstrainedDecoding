"""The oracle model and the constraints of the notes' experiments.

The model is a hidden Markov chain used as a consistent oracle (notes, section 2.3): its
predictions p(x_0,i | x_t) are the exact conditionals of one joint law, which is known, so the
target P(x_0 | x_0 in C) can be computed exactly. Tokens are classes.
"""

from __future__ import annotations

from typing import Any, Tuple

from twistroute import constraints
from twistroute.constraints import Constraint
from twistroute.models import noisy_hmm

#: number of tokens for each constraint
N_TOKENS = {"counting": 3, "dyck": 2, "forbidden": 3}
#: the names used in the notes' tables
TITLES = {"counting": "Counting", "dyck": "Dyck", "forbidden": "Forbidden motifs"}


def hidden_chain(length: int, n_tokens: int) -> Any:
    """The hidden-chain oracle: a hidden state that stays put with probability 0.85 and is
    observed through noise 0.3, so dependence between positions decays with distance and no
    single unmasked token separates the two sides."""
    return noisy_hmm(length, n_tokens, 0.85, 0.3, seed=1, concentration=2.0)


def constraint(name: str, length: int) -> Constraint:
    """counting: token 0 occurs exactly L / 3 times; dyck: balanced brackets (0 opens, 1
    closes); forbidden: the motifs 00 and 12 never occur."""
    if name == "counting":
        return constraints.exactly_k(0, length // 3, 3)
    if name == "dyck":
        return constraints.dyck1(length)
    if name == "forbidden":
        return constraints.forbidden_motifs([(0, 0), (1, 2)], 3)
    raise ValueError(f"unknown constraint {name!r}")


def cell(name: str, length: int) -> Tuple[Any, Any, Constraint]:
    """(model, token classes, constraint) for one constraint at one length."""
    model = hidden_chain(length, N_TOKENS[name])
    return model, model.classes(), constraint(name, length)
