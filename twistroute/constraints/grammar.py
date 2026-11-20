"""Context-free grammars in Chomsky normal form (notes, Table 1, row 5).

One pass is the inside algorithm: O(L^3 |R|), R the rules. A grammar with span limits (the
longest stretch a nonterminal may cover) costs less; see the notes for the bound.

A sequence with several parses is weighted by its number of parses (notes, section 2.1).

Precision: the pass works in single precision, so log q(C) and the marginals are exact to about
1e-6, not to machine precision as for the automaton families.
"""
from __future__ import annotations

from typing import Optional

from .. import compile as _compile
from ..torch_routes import TorchChart
from .base import Constraint

DYCK1 = _compile.DYCK1_UNAMBIGUOUS       # balanced brackets, one kind, unambiguous
DYCK1_AMBIGUOUS = _compile.DYCK1_AMBIGUOUS
DYCK2 = _compile.DYCK2                   # balanced brackets, two kinds
DYCK2_WITH_ATOMS = _compile.DYCK2_ATOMS  # the same with a free atom symbol, e.g. (x[x]()x)


def grammar(cnf, *, span_limits: Optional[dict] = None, device: str = "cpu", name: str = "") -> Constraint:
    """cnf: (start symbol, binary rules, lexical rules, number of classes), as the constants above."""
    inside = _compile.Inside(*cnf, span_limits=span_limits)
    return Constraint(TorchChart(inside, device), family="context-free grammar", cost="O(L^3 |R|)", n_classes=cnf[3],
                      reference=inside, name=name or "grammar")
