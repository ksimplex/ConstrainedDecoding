"""
CPU reference compiler: every constraint family compiles to a *route* exposing

    accepts(seq) -> bool                  exact membership (the terminal twist 1_C)
    logZ(logW)   -> float                 log sum_{y in C} d(y) prod_i exp(logW[i, y_i])
    sample(logW, rng) -> tuple            exact draw from that law
    marginals(logW) (where available)     per-position constrained marginals

with logW[L, A] factorized per-position log-weights over a small symbol alphabet. d(y) = 1 for
deterministic routes and the parse count for ambiguous grammars. These implementations are written for
clarity and are verified against brute-force enumeration (tests/test_compile.py); the batched GPU
versions used for decoding live in twistroute.torch_routes.
"""
from ._util import CompilerRefuse, NEG
from .automata import (WDFA, build_dfa, dfa_forbidden, dfa_contains, dfa_exactly_k, dfa_fraction_band,
                       dfa_regex_a_then_no_bb, dfa_copy_positional, product_dfa, dfa_window_alldiff,
                       dfa_dyck2_depth, dfa_prefix_class)
from .grammars import (Inside, count_parses, random_cnf, DYCK2, DYCK1_AMBIGUOUS, DYCK1_UNAMBIGUOUS,
                       RNA_PAIRS, rna_pair_grammar)
from .factor_graphs import CopyFG, FactorGraph, ve_logZ
from .trees import Arborescence
