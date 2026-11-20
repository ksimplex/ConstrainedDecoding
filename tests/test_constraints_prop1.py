"""The constraint interface of Proposition 1 (notes, section 2.2), on every family of Table 1.

Each case pairs a constraint with a plain Python predicate written independently of the library.
With all A^L sequences enumerated, log q(C), the constrained marginals and membership are checked
exactly, and exact draws are checked to lie in C and to follow the constrained marginals.
"""
import itertools

import numpy as np
import pytest
import torch

import twistroute.constraints as tc

L = 6


def balanced(x):
    depth = 0
    for v in x:
        depth += 1 if v == 0 else -1
        if depth < 0:
            return False
    return depth == 0


def no_motif(x, motif):
    m = len(motif)
    return all(tuple(x[i:i + m]) != tuple(motif) for i in range(len(x) - m + 1))


ALLOWED = np.ones((L, 3), dtype=bool)
ALLOWED[0, 0] = False
ALLOWED[3, 2] = False
DIFFERENT = 1.0 - np.eye(3)
EQUAL = np.eye(3)

CASES = {
    "counting": (lambda: tc.exactly_k(0, 2, 3), 3, lambda x: x.count(0) == 2),
    "at least k": (lambda: tc.at_least_k(1, 3, 3), 3, lambda x: x.count(1) >= 3),
    "forbidden motifs": (lambda: tc.forbidden_motifs([(0, 0), (1, 2, 1)], 3), 3,
                         lambda x: no_motif(x, (0, 0)) and no_motif(x, (1, 2, 1))),
    "Dyck-1, automaton": (lambda: tc.dyck1(L), 2, balanced),
    "subset of positions": (lambda: tc.on_positions(tc.exactly_k(0, 1, 3), [1, 4, 5], L), 3,
                            lambda x: [x[1], x[4], x[5]].count(0) == 1),
    "allowed sets": (lambda: tc.with_allowed_sets(tc.exactly_k(0, 2, 3), ALLOWED), 3,
                     lambda x: x.count(0) == 2 and all(ALLOWED[i, v] for i, v in enumerate(x))),
    "copy": (lambda: tc.copy_pairs([(0, 3), (1, 4)], L, 3), 3, lambda x: x[0] == x[3] and x[1] == x[4]),
    "factor graph": (lambda: tc.factor_graph(L, 3, [((0, 2), DIFFERENT), ((2, 5), EQUAL), ((1, 2), DIFFERENT)]), 3,
                     lambda x: x[0] != x[2] and x[2] == x[5] and x[1] != x[2]),
    "Dyck-1, grammar": (lambda: tc.grammar(tc.DYCK1), 2, balanced),
    "automaton and automaton": (lambda: tc.intersect(tc.exactly_k(0, 2, 3), tc.forbidden_motifs([(0, 0)], 3)), 3,
                                lambda x: x.count(0) == 2 and no_motif(x, (0, 0))),
    "grammar and automaton": (lambda: tc.intersect(tc.grammar(tc.DYCK1), tc.forbidden_motifs([(0, 1, 0, 1)], 2)), 2,
                              lambda x: balanced(x) and no_motif(x, (0, 1, 0, 1))),
    "factor graph and automaton": (lambda: tc.intersect(tc.factor_graph(L, 3, [((0, 2), DIFFERENT), ((2, 5), EQUAL)]),
                                                        tc.exactly_k(0, 2, 3)), 3,
                                   lambda x: x[0] != x[2] and x[2] == x[5] and x.count(0) == 2),
}


def enumerate_exact(log_q, predicate):
    """Brute force for one row: log q(C) and the constrained marginals (L, A)."""
    n_classes = log_q.shape[-1]
    seqs = np.array(list(itertools.product(range(n_classes), repeat=L)))
    inside = np.array([predicate(list(s)) for s in seqs])
    weight = np.exp(log_q.numpy()[np.arange(L)[None], seqs].sum(1)) * inside
    marginals = np.stack([np.bincount(seqs[:, i], weights=weight, minlength=n_classes) for i in range(L)]) / weight.sum()
    return np.log(weight.sum()), marginals, seqs, inside


@pytest.mark.parametrize("name", list(CASES))
def test_log_prob_marginals_and_membership_are_exact(name):
    build, n_classes, predicate = CASES[name]
    constraint = build()
    assert constraint.n_classes == n_classes
    # the grammar and factor-graph passes work in single precision; the automaton families in double
    tolerance = 2e-6 if ("grammar" in constraint.family or constraint.family == "factor graph") else 1e-12
    log_q = torch.log_softmax(torch.randn(3, L, n_classes, dtype=torch.float64, generator=torch.Generator().manual_seed(1)), -1)
    log_prob = constraint.log_prob(log_q)
    marginals = constraint.marginals(log_q)
    for row in range(3):
        want_log_prob, want_marginals, seqs, inside = enumerate_exact(log_q[row], predicate)
        assert abs(float(log_prob[row]) - want_log_prob) < tolerance
        assert np.abs(marginals[row].numpy() - want_marginals).max() < tolerance
    assert np.array_equal(constraint.contains(torch.as_tensor(seqs)).numpy(), inside)


@pytest.mark.parametrize("name", list(CASES))
def test_draws_lie_in_the_constraint_and_follow_its_marginals(name):
    build, n_classes, predicate = CASES[name]
    constraint = build()
    log_q = torch.log_softmax(torch.randn(1, L, n_classes, dtype=torch.float64, generator=torch.Generator().manual_seed(2)), -1)
    n = 4000
    draws = constraint.sample(log_q.expand(n, L, n_classes), torch.Generator().manual_seed(3)).numpy()
    assert all(predicate(list(s)) for s in draws[:500])
    _, want, _, _ = enumerate_exact(log_q[0], predicate)
    got = np.stack([np.bincount(draws[:, i], minlength=n_classes) for i in range(L)]) / n
    assert np.abs(got - want).max() < 4 * np.sqrt(0.25 / n)


def test_fix_unmasked_conditions_on_the_revealed_positions():
    constraint = tc.exactly_k(0, 2, 3)
    log_q = torch.log_softmax(torch.randn(1, L, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4)), -1)
    classes = torch.tensor([[0, 2, 0, 0, 1, 0]])
    unmasked = torch.tensor([[True, False, False, False, True, False]])
    fixed = tc.fix_unmasked(log_q, classes, unmasked)
    # p(C | x_t): position 0 is class 0 and position 4 is class 1, so one more 0 among positions 1, 2, 3, 5
    p = torch.exp(log_q[0])
    rest = [1, 2, 3, 5]
    want = sum(p[i, 0] * torch.prod(torch.stack([1 - p[j, 0] for j in rest if j != i])) for i in rest)
    assert abs(float(torch.exp(constraint.log_prob(fixed))[0]) - float(want)) < 1e-12
