"""Finite automata (notes, Table 1, first row). One pass costs O(L |E|), E the edges over classes.

`automaton` wraps any `twistroute.compile.WDFA`. The functions below build the automata used in
the notes' experiments.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .. import compile as _compile
from ..torch_routes import TorchChain
from .base import Constraint

COST = "O(L |E|)"


def automaton(
    dfa: "_compile.WDFA", *, length: int | None = None, device: str = "cpu", name: str = ""
) -> Constraint:
    """The constraint 'the automaton accepts the sequence of classes'."""
    return Constraint(
        TorchChain(dfa, device),
        family="finite automaton",
        cost=COST,
        n_classes=dfa.A,
        length=length,
        reference=dfa,
        name=name or getattr(dfa, "name", "automaton"),
    )


def forbidden_motifs(motifs: Iterable[Sequence[int]], n_classes: int, **kw) -> Constraint:
    """No motif (a short sequence of classes) occurs anywhere."""
    return automaton(
        _compile.dfa_forbidden([tuple(m) for m in motifs], n_classes), name="forbidden motifs", **kw
    )


def exactly_k(cls: int, k: int, n_classes: int, **kw) -> Constraint:
    """Class `cls` occurs exactly k times ("counting" in the notes)."""
    return automaton(
        _compile.dfa_exactly_k(cls, k, n_classes), name=f"exactly {k} of class {cls}", **kw
    )


def at_least_k(cls: int, k: int, n_classes: int, **kw) -> Constraint:
    """Class `cls` occurs at least k times."""
    return automaton(
        _compile.dfa_at_least_k(cls, k, n_classes), name=f"at least {k} of class {cls}", **kw
    )


def dyck1(length: int, **kw) -> Constraint:
    """Balanced brackets over two classes (open, close) at a fixed length: Dyck-1 as an automaton
    whose state is the depth, which is finite once the length is fixed."""
    return automaton(_compile.dfa_dyck1(length), length=length, name="Dyck-1", **kw)
