"""Constraints that single out positions (notes, Table 1, rows 2 and 3).

on_positions       a constraint on a subset of m positions; the others are free.
                   Cost: the inner constraint at length m, plus O(L A).
with_allowed_sets  a per-position set of allowed classes on top of another constraint.
                   Cost: the inner constraint, plus O(L A).
"""

from __future__ import annotations

from typing import Sequence

from ..torch_routes import TorchMasked, TorchSubset
from .base import Constraint


def on_positions(
    inner: Constraint, positions: Sequence[int], length: int, *, device: str = "cpu"
) -> Constraint:
    """`inner` applied to the listed positions (in that order) of a length-L sequence."""
    positions = [int(p) for p in positions]
    impl = TorchSubset(inner.route, positions, length, device)
    return Constraint(
        impl,
        family="constraint on a subset of positions",
        cost=f"{inner.cost} at length m, plus O(L A)",
        n_classes=inner.n_classes,
        length=length,
        reference=(inner.reference, positions),
        name=f"{inner.name} on {len(positions)} positions",
    )


def with_allowed_sets(inner: Constraint, allowed, *, device: str = "cpu") -> Constraint:
    """`inner` and, at every position i, the class must be in the allowed set: allowed is
    (L, A) bool."""
    impl = TorchMasked(inner.route, allowed, device)
    return Constraint(
        impl,
        family="per-position allowed sets",
        cost=f"{inner.cost}, plus O(L A)",
        n_classes=inner.n_classes,
        length=len(allowed),
        reference=(inner.reference, allowed),
        name=f"{inner.name} with allowed sets",
    )
