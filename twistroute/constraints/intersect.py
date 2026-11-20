"""Intersection of a constraint with a finite automaton (notes, Table 2).

automaton x automaton     product automaton
grammar x automaton       grammar (Bar-Hillel product), on the same chart
factor graph x automaton  factor graph over (class, automaton state)
"""

from __future__ import annotations

from .. import compile as _compile
from ..torch_routes.build import torch_route
from .automaton import automaton
from .base import Constraint


def intersect(
    constraint: Constraint, with_automaton: Constraint, *, device: str = "cpu"
) -> Constraint:
    """The constraint 'both hold'. The second argument must be a finite automaton."""
    dfa = with_automaton.reference
    other = constraint.reference
    if not isinstance(dfa, _compile.WDFA):
        raise TypeError("the second constraint must be a finite automaton")
    name = f"{constraint.name} and {with_automaton.name}"
    if isinstance(other, _compile.WDFA):
        return automaton(
            _compile.product_dfa(other, dfa, dfa.A),
            length=constraint.length or with_automaton.length,
            device=device,
            name=name,
        )
    if isinstance(other, _compile.Inside):
        product = _compile.intersect_dfa_cfg(dfa, other)
        return Constraint(
            torch_route(product, device),
            family="context-free grammar",
            cost="O(L^3 |R| |Q2|^3) worst case",
            n_classes=constraint.n_classes,
            reference=product,
            name=name,
        )
    if isinstance(other, _compile.FactorGraph):
        product = _compile.DFAFactorProduct(dfa, other)
        return Constraint(
            torch_route(product, device),
            family="factor graph",
            cost="O(L D^(w'+1))",
            n_classes=constraint.n_classes,
            length=constraint.length,
            reference=product,
            name=name,
        )
    raise TypeError(f"no intersection for the family {constraint.family!r}")
