"""The new core checked on its own terms (no comparison with the earlier sampler).

Chain law (eq. 11 to 13): against enumeration of all class sequences at a short length.
Weighted sampler (eq. 4 to 6): the estimate of P(x_0 in C) is unbiased for every twist, the
returned sequences lie in C, and the unconstrained step changes only the unmasking set.
"""

import itertools
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "studies"))
from pulib.cells import make_constraint, make_model, n_symbols  # noqa: E402

from twistroute import compile as automata  # noqa: E402
from twistroute.constraints import Constraint  # noqa: E402
from twistroute.resampling import Resampling  # noqa: E402
from twistroute.sampler import model_prediction, unconstrained_step, weighted_sampler  # noqa: E402
from twistroute.schedules import RandomOrder  # noqa: E402
from twistroute.twists import ChainLaw, ChainTwist, ProductTwist, counted_pair_table  # noqa: E402

LENGTH = 5
N_CLASSES = 3


def chain_law_by_enumeration(law, prediction):
    """All sequences, their probability under the chain law of eq. (12), and membership in C."""
    mu = law.normalised(prediction[None])[0]
    conditional = law.transitions(mu[None])[0]
    sequences = list(itertools.product(range(N_CLASSES), repeat=LENGTH))
    probability = []
    for sequence in sequences:
        p = float(mu[0, sequence[0]])
        for i in range(LENGTH - 1):
            p *= float(conditional[i, sequence[i], sequence[i + 1]])
        probability.append(p)
    return sequences, np.array(probability)


@pytest.fixture(scope="module")
def small_chain():
    generator = torch.Generator().manual_seed(3)
    automaton = automata.dfa_forbidden([(0, 1), (2, 2)], N_CLASSES)
    table = torch.rand(N_CLASSES, N_CLASSES, generator=generator, dtype=torch.float64) + 0.2
    table = table / table.sum(-1, keepdim=True)
    law = ChainLaw(automaton, table, matching_sweeps=200)
    prediction = torch.log_softmax(
        torch.randn(LENGTH, N_CLASSES, generator=generator, dtype=torch.float64), -1
    )
    inside = [
        not any(s[i : i + 2] in ((0, 1), (2, 2)) for i in range(LENGTH - 1))
        for s in itertools.product(range(N_CLASSES), repeat=LENGTH)
    ]
    return law, prediction, np.array(inside)


def test_matched_chain_law_is_a_distribution_with_the_models_marginals(small_chain):
    law, prediction, _ = small_chain
    sequences, probability = chain_law_by_enumeration(law, prediction)
    assert abs(probability.sum() - 1.0) < 1e-12
    mu = law.normalised(prediction[None])[0].numpy()
    for i in range(LENGTH):
        for a in range(N_CLASSES):
            marginal = sum(p for s, p in zip(sequences, probability) if s[i] == a)
            assert abs(marginal - mu[i, a]) < 1e-9


def test_chain_mass_and_fixed_mass_match_enumeration(small_chain):
    law, prediction, inside = small_chain
    sequences, probability = chain_law_by_enumeration(law, prediction)
    assert abs(float(law.log_mass(prediction[None])) - math.log(probability[inside].sum())) < 1e-12
    positions = torch.tensor([[1, 3]])
    classes = torch.tensor([[2, 0]])
    fixed = sum(
        p for s, p, ok in zip(sequences, probability, inside) if ok and s[1] == 2 and s[3] == 0
    )
    assert (
        abs(float(law.log_mass_with(prediction[None], positions, classes)) - math.log(fixed))
        < 1e-12
    )


def test_chain_sampler_draws_from_the_chain_law_given_the_constraint(small_chain):
    law, prediction, inside = small_chain
    sequences, probability = chain_law_by_enumeration(law, prediction)
    target = np.where(inside, probability, 0.0)
    target = target / target.sum()
    n_draws = 40000
    draws, log_mass = law.sample(
        prediction[None].expand(n_draws, -1, -1), torch.Generator().manual_seed(5)
    )
    assert abs(float(log_mass[0]) - math.log(probability[inside].sum())) < 1e-12
    index = (draws * (N_CLASSES ** torch.arange(LENGTH - 1, -1, -1))).sum(-1).numpy()
    observed = np.bincount(index, minlength=len(sequences)) / n_draws
    assert observed[~inside].sum() == 0
    standard_error = np.sqrt(target * (1 - target) / n_draws)
    assert np.abs(observed - target)[inside].max() < 5 * standard_error[inside].max()


TWISTS = {
    "product": lambda automaton, table, active: ProductTwist(),
    "product, tempered 0.5": lambda automaton, table, active: ProductTwist(temperature=0.5),
    "product, power 0.5": lambda automaton, table, active: ProductTwist(power=0.5),
    "chain": lambda automaton, table, active: ChainTwist(automaton, table, active=active),
    "chain, product share 0.1": lambda automaton, table, active: ChainTwist(
        automaton, table, active=active, proposal_product_share=0.1
    ),
    "chain twist, product proposal": lambda automaton, table, active: ChainTwist(
        automaton, table, active=active, proposal="product"
    ),
}


@pytest.fixture(scope="module")
def cell():
    length = 12
    model = make_model("hidden_0.85", length, n_symbols("forbidden"))
    classes, route, truth = make_constraint("forbidden", model, length)
    constraint = Constraint(
        route, family="finite automaton", cost="O(L |E|)", n_classes=classes.A, length=length
    )
    table = counted_pair_table(model, classes, 2000, torch.Generator().manual_seed(1))
    return model, classes, constraint, truth, table, length


@pytest.mark.parametrize("name", list(TWISTS))
@pytest.mark.parametrize(
    "rule",
    [Resampling(threshold=0.0), Resampling(), Resampling(defensive=0.25)],
    ids=["no resampling", "plain", "defensive"],
)
def test_estimate_of_p_C_is_unbiased(cell, name, rule):
    model, classes, constraint, truth, table, length = cell
    twist = TWISTS[name](truth.wdfa, table, getattr(truth, "active", None))
    out = weighted_sampler(
        model,
        classes,
        constraint,
        length,
        length,
        runs=1500,
        particles=4,
        twist=twist,
        resampling=rule,
        generator=torch.Generator().manual_seed(11),
    )
    ratio = np.exp(out.log_Z_hat - truth.log_pC)
    z = (ratio.mean() - 1.0) / (ratio.std(ddof=1) / math.sqrt(len(ratio)))
    assert abs(z) < 4.0, (name, ratio.mean(), z)
    assert not any(out.support.values())


def test_returned_sequences_are_in_the_constraint(cell):
    model, classes, constraint, truth, table, length = cell
    out = weighted_sampler(
        model,
        classes,
        constraint,
        length,
        length,
        runs=200,
        particles=4,
        generator=torch.Generator().manual_seed(2),
    )
    returned = out.returned(np.random.default_rng(0))
    assert (returned >= 0).all()
    assert bool(constraint.contains(classes.token_class[torch.as_tensor(returned)]).all())


def test_unconstrained_step_changes_only_the_unmasking_set(cell):
    model, classes, constraint, truth, table, length = cell
    tokens = torch.full((50, length), model.mask_id, dtype=torch.long)
    state = model_prediction(model, classes, constraint, tokens, torch.zeros_like(tokens))
    step = unconstrained_step(
        state, RandomOrder(), 0, length // 3, torch.Generator().manual_seed(4)
    )
    changed = step.tokens != state.tokens
    assert (changed.sum(-1) == 3).all()
    assert bool(torch.gather(changed, 1, step.unmask).all())
    assert bool((step.tokens[changed] != model.mask_id).all())
