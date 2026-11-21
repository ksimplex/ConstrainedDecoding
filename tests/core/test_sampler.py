"""The new core checked on its own terms (no comparison with the earlier sampler).

Chain prediction (eq. 11 and 12) and the chain weight (eq. 13): against enumeration of all class
sequences at a short length.
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "notes"))
from cells import cell as notes_cell  # noqa: E402

from twistroute import compile as automata  # noqa: E402
from twistroute.evaluation import log_prob_of_constraint  # noqa: E402
from twistroute.resampling import Resampling  # noqa: E402
from twistroute.sampler import (  # noqa: E402
    commit,
    model_prediction,
    unconstrained_step,
    weighted_sampler,
)
from twistroute.schedules import RandomOrder  # noqa: E402
from twistroute.twists import ChainLaw, ChainTwist, ProductTwist, counted_pair_table  # noqa: E402

LENGTH = 5
N_CLASSES = 3


def chain_law_by_enumeration(law, prediction):
    """All sequences and their probability under the chain prediction of eq. (11)."""
    length, n_classes = prediction.shape
    mu = law.normalised(prediction[None])[0]
    conditional = law.transitions(mu[None])[0]
    sequences = list(itertools.product(range(n_classes), repeat=length))
    probability = []
    for sequence in sequences:
        p = float(mu[0, sequence[0]])
        for i in range(length - 1):
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


def chain_steps(twist, n_states=6, seed=3):
    """Constrained denoising steps taken by hand with the sampler's own calls, at a short length,
    from partly unmasked states, unmasking one or two positions. Yields what eq. (13) needs."""
    length = 6
    model, classes, constraint = notes_cell("forbidden", length)
    generator = torch.Generator().manual_seed(seed)
    inside = constraint.contains(
        classes.token_class[torch.cartesian_prod(*[torch.arange(3)] * length)]
    )
    allowed = torch.cartesian_prod(*[torch.arange(3)] * length)[inside]
    for number in range(n_states):
        tokens = allowed[torch.randint(len(allowed), (1,), generator=generator)].clone()
        n_masked = 2 + number % 4  # two to five masked positions
        masked = torch.randperm(length, generator=generator)[:n_masked]
        tokens[0, masked] = model.mask_id
        unmasked_class = torch.where(tokens == model.mask_id, torch.zeros_like(tokens), tokens)
        state = model_prediction(
            model, classes, constraint, tokens, classes.token_class[unmasked_class]
        )
        state.log_psi = twist.log_psi(state)
        unmask = masked[: 1 + number % 2][None]  # |U| = 1 or 2
        completion, extra = twist.draw_completion(state, generator)
        step = commit(state, completion, unmask, None, generator)
        step.extra = extra
        new_state = model_prediction(model, classes, constraint, step.tokens, step.classes)
        log_psi_next = twist.log_psi(new_state)
        log_w, _ = twist.log_weight(state, step, log_psi_next)
        in_C = inside.numpy()
        yield state, step, new_state, float(log_w), in_C


def chain_quantities(twist, state, step, new_state, in_C):
    """By listing all sequences: r(C | x_t), r(x_U | x_t), r(x_U, C | x_t), r(C | x_s), and
    prod_U mu_i(x_s,i)."""
    sequences, before = chain_law_by_enumeration(twist.law, state.prediction()[0])
    _, after = chain_law_by_enumeration(twist.law, new_state.prediction()[0])
    positions = step.unmask[0].tolist()
    drawn = step.drawn[0].tolist()
    agrees = np.array([all(s[i] == c for i, c in zip(positions, drawn)) for s in sequences])
    mu = twist.law.normalised(state.prediction())[0].numpy()
    product = float(np.prod([mu[i, c] for i, c in zip(positions, drawn)]))
    return (
        before[in_C].sum(),
        before[agrees].sum(),
        before[agrees & in_C].sum(),
        after[in_C].sum(),
        product,
    )


def test_chain_weight_is_equation_13():
    """With the notes' twist (no share of the product twist) and converged matching, the weight
    computed by `ChainTwist.log_weight` is eq. (13), for one and for two unmasked positions; and
    its second factor is 1 when one position is unmasked."""
    model, classes, constraint = notes_cell("forbidden", 6)
    table = counted_pair_table(model, classes, 2000, torch.Generator().manual_seed(1))
    twist = ChainTwist.for_constraint(constraint, table, product_share=0.0, matching_sweeps=300)
    sizes = set()
    for state, step, new_state, log_w, in_C in chain_steps(twist):
        mass, drawn_mass, drawn_and_C, mass_next, product = chain_quantities(
            twist, state, step, new_state, in_C
        )
        first_factor = mass_next / (drawn_and_C / drawn_mass)  # r(C | x_s) / r(C | x_t, x_U)
        second_factor = product / drawn_mass  # prod_U mu_i / r(x_U | x_t)
        assert abs(log_w - math.log(first_factor * second_factor)) < 1e-6
        if step.unmask.shape[1] == 1:
            assert abs(second_factor - 1.0) < 1e-6
        sizes.add(step.unmask.shape[1])
    assert sizes == {1, 2}


def test_chain_weight_with_default_options_is_equation_5():
    """With the defaults (a 0.05 share of the product twist, 5 matching sweeps) the weight is
    eq. (5) for the twist actually used, psi = 0.95 r(C | .) + 0.05 p(C | .); it is not eq. (13)."""
    model, classes, constraint = notes_cell("forbidden", 6)
    table = counted_pair_table(model, classes, 2000, torch.Generator().manual_seed(1))
    twist = ChainTwist.for_constraint(constraint, table)
    largest_gap = 0.0
    for state, step, new_state, log_w, in_C in chain_steps(twist):
        mass, drawn_mass, drawn_and_C, mass_next, product = chain_quantities(
            twist, state, step, new_state, in_C
        )
        product_now = math.exp(float(constraint.log_prob(state.prediction())))
        product_next = math.exp(float(constraint.log_prob(new_state.prediction())))
        psi_now = 0.95 * mass + 0.05 * product_now
        psi_next = 0.95 * mass_next + 0.05 * product_next
        proposal_over_kappa = drawn_and_C / mass / product  # M / kappa
        assert abs(log_w - math.log(psi_next / psi_now / proposal_over_kappa)) < 1e-6
        equation_13 = mass_next / (drawn_and_C / drawn_mass) * product / drawn_mass
        largest_gap = max(largest_gap, abs(math.exp(log_w) / equation_13 - 1.0))
    assert largest_gap > 1e-4


TWISTS = {
    "product, as in the notes": lambda constraint, table: ProductTwist(temperature=1.0),
    "product, default (tempered 0.7)": lambda constraint, table: ProductTwist(),
    "product, tempered 0.5": lambda constraint, table: ProductTwist(temperature=0.5),
    "product, power 0.5": lambda constraint, table: ProductTwist(power=0.5),
    "chain": lambda constraint, table: ChainTwist.for_constraint(constraint, table),
    "chain, product share 0.1": lambda constraint, table: ChainTwist.for_constraint(
        constraint, table, proposal_product_share=0.1
    ),
    "chain twist, product proposal": lambda constraint, table: ChainTwist.for_constraint(
        constraint, table, proposal="product"
    ),
}


@pytest.fixture(scope="module")
def cell():
    length = 12
    model, classes, constraint = notes_cell("forbidden", length)
    log_p_C = log_prob_of_constraint(model, classes, constraint, length)
    table = counted_pair_table(model, classes, 2000, torch.Generator().manual_seed(1))
    return model, classes, constraint, log_p_C, table, length


@pytest.mark.parametrize("name", list(TWISTS))
@pytest.mark.parametrize(
    "rule",
    [Resampling(threshold=0.0), Resampling(), Resampling(defensive=0.25)],
    ids=["no resampling", "plain", "defensive"],
)
def test_estimate_of_p_C_is_unbiased(cell, name, rule):
    model, classes, constraint, log_p_C, table, length = cell
    twist = TWISTS[name](constraint, table)
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
    ratio = np.exp(out.log_Z_hat - log_p_C)
    z = (ratio.mean() - 1.0) / (ratio.std(ddof=1) / math.sqrt(len(ratio)))
    assert abs(z) < 4.0, (name, ratio.mean(), z)
    assert not any(out.support.values())


def test_returned_sequences_are_in_the_constraint(cell):
    model, classes, constraint, log_p_C, table, length = cell
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
    model, classes, constraint, log_p_C, table, length = cell
    tokens = torch.full((50, length), model.mask_id, dtype=torch.long)
    state = model_prediction(model, classes, constraint, tokens, torch.zeros_like(tokens))
    step = unconstrained_step(
        state, RandomOrder(), 0, length // 3, torch.Generator().manual_seed(4)
    )
    changed = step.tokens != state.tokens
    assert (changed.sum(-1) == 3).all()
    assert bool(torch.gather(changed, 1, step.unmask).all())
    assert bool((step.tokens[changed] != model.mask_id).all())
