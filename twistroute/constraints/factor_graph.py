"""Factor graphs (notes, Table 1, row 4). One pass costs O(L A^(w+1)), w the induced width.

`copy_pairs` is the special case 'these two positions must be equal', which has a closed form
and costs O(L A).

Precision: the pass works in single precision, so log q(C) and the marginals are exact to about
1e-6, not to machine precision as for the automaton families.
"""
from __future__ import annotations

from typing import Sequence, Tuple

from .. import compile as _compile
from ..torch_routes import TorchCopy
from ..torch_routes.factor_graph import TorchFactorGraph
from .base import Constraint


def factor_graph(length: int, n_classes: int, factors, *, order=None, device: str = "cpu", name: str = "") -> Constraint:
    """factors: list of (scope, table): scope a tuple of positions, table a 0/1 (or weight) array
    with one axis of size A per position in the scope."""
    factors = [(tuple(scope), table) for scope, table in factors]
    order = order or _compile.min_degree_order(length, [scope for scope, _ in factors])
    fg = _compile.FactorGraph(length, n_classes, factors, order=order, name=name or "factor graph")
    return Constraint(TorchFactorGraph(fg, device), family="factor graph", cost="O(L A^(w+1))", n_classes=n_classes,
                      length=length, reference=fg, name=name or "factor graph")


def copy_pairs(pairs: Sequence[Tuple[int, int]], length: int, n_classes: int, *, device: str = "cpu") -> Constraint:
    """Position i must equal position j for every pair (i, j); each position in at most one pair."""
    pairs = [(int(i), int(j)) for i, j in pairs]
    return Constraint(TorchCopy(pairs, length, n_classes, device), family="factor graph (copy)", cost="O(L A)",
                      n_classes=n_classes, length=length, reference=None, name=f"copy, {len(pairs)} pairs")
