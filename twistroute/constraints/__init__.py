"""Constraints (notes, section 2): the interface of Proposition 1 and the families of Table 1.

    base          Constraint: log_prob, marginals, sample, contains; fix_unmasked
    automaton     finite automata: forbidden motifs, counting, Dyck-1, any WDFA
    positions     a constraint on a subset of positions; per-position allowed sets
    factor_graph  factor graphs; copy constraints
    grammar       context-free grammars in Chomsky normal form
    intersect     any of the above with a finite automaton (Table 2)
    tokens        a regular expression over characters as a constraint over a model's tokens

The exact algorithms are those of `twistroute.compile` (CPU reference) and
`twistroute.torch_routes` (batched); this package gives them the names used in the notes.
"""

from .automaton import at_least_k, automaton, dyck1, exactly_k, forbidden_motifs
from .base import Constraint, fix_unmasked, one_hot_log
from .factor_graph import copy_pairs, factor_graph
from .grammar import DYCK1, DYCK1_AMBIGUOUS, DYCK2, DYCK2_WITH_ATOMS, grammar
from .intersect import intersect
from .positions import on_positions, with_allowed_sets
from .tokens import from_regex, token_codes

__all__ = [
    "Constraint",
    "fix_unmasked",
    "one_hot_log",
    "automaton",
    "forbidden_motifs",
    "exactly_k",
    "at_least_k",
    "dyck1",
    "on_positions",
    "with_allowed_sets",
    "factor_graph",
    "copy_pairs",
    "grammar",
    "DYCK1",
    "DYCK1_AMBIGUOUS",
    "DYCK2",
    "DYCK2_WITH_ATOMS",
    "intersect",
    "from_regex",
    "token_codes",
]
