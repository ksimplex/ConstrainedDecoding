"""The exact evaluation agrees with the sampler it describes, and the bounds bracket the exact
KL (hidden-chain oracle, 8 positions)."""

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "notes"))
from cells import cell as notes_cell  # noqa: E402

from twistroute.evaluation import (  # noqa: E402
    kl,
    leading_term,
    one_particle_law,
    output_law,
    target_law,
    upper_bound,
)
from twistroute.resampling import NO_RESAMPLING  # noqa: E402
from twistroute.sampler import weighted_sampler  # noqa: E402
from twistroute.schedules import LeftToRight, RandomOrder  # noqa: E402
from twistroute.twists import ChainTwist, ProductTwist, counted_pair_table  # noqa: E402

LENGTH = 8
STORED = os.path.join(
    HERE, "..", "..", "results", "parallel_unmasking", "particle_rule", "kl_bounds_check.json"
)


@pytest.fixture(scope="module")
def cell():
    model, classes, constraint = notes_cell("forbidden", LENGTH)
    table = counted_pair_table(model, classes, 2000, torch.Generator().manual_seed(5))
    chain = ChainTwist.for_constraint(constraint, table)
    return model, classes, constraint, chain


def frequencies(tokens, n_tokens):
    index = (tokens * n_tokens ** np.arange(LENGTH - 1, -1, -1)).sum(-1)
    return np.bincount(index, minlength=n_tokens**LENGTH) / len(tokens)


def assert_same_law(observed, law, n):
    error = np.sqrt(np.maximum(law * (1 - law), 1e-12) / n)
    assert observed[law == 0].sum() == 0
    assert (np.abs(observed - law) / error).max() < 5.0


@pytest.mark.parametrize(
    "schedule", [LeftToRight(), RandomOrder()], ids=["left to right", "random order"]
)
@pytest.mark.parametrize("twist_name", ["product", "tempered product", "chain"])
def test_one_particle_law_is_the_law_of_one_particle(cell, schedule, twist_name):
    model, classes, constraint, chain = cell
    twist = {
        "product": ProductTwist(),
        "tempered product": ProductTwist(temperature=0.5),
        "chain": chain,
    }[twist_name]
    law = one_particle_law(model, classes, constraint, LENGTH, schedule=schedule, twist=twist)
    n = 60000
    out = weighted_sampler(
        model,
        classes,
        constraint,
        LENGTH,
        LENGTH,
        runs=n,
        particles=1,
        twist=twist,
        schedule=schedule,
        generator=torch.Generator().manual_seed(7),
    )
    assert_same_law(frequencies(out.tokens[:, 0], model.S), law, n)


def test_output_law_is_the_law_of_the_returned_particle(cell):
    model, classes, constraint, chain = cell
    target = target_law(model, classes, constraint, LENGTH)
    law = output_law(
        one_particle_law(model, classes, constraint, LENGTH, schedule=LeftToRight()), target, 4
    )
    n = 60000
    out = weighted_sampler(
        model,
        classes,
        constraint,
        LENGTH,
        LENGTH,
        runs=n,
        particles=4,
        schedule=LeftToRight(),
        resampling=NO_RESAMPLING,
        generator=torch.Generator().manual_seed(8),
    )
    assert_same_law(frequencies(out.returned(np.random.default_rng(1)), model.S), law, n)


def test_exact_kl_matches_the_values_behind_table_3(cell):
    model, classes, constraint, chain = cell
    stored = {
        (r["proposal"], r["K"]): r["exact"]
        for r in json.load(open(STORED))["cases"]
        if r["constraint"] == "forbidden" and r["order"] == "left to right"
    }
    target = target_law(model, classes, constraint, LENGTH)
    product = one_particle_law(model, classes, constraint, LENGTH, schedule=LeftToRight())
    matched = one_particle_law(
        model, classes, constraint, LENGTH, schedule=LeftToRight(), twist=chain
    )
    assert abs(kl(product, target) - stored[("product", 1)]) < 1e-6
    assert abs(kl(matched, target) - stored[("matched chain", 1)]) < 1e-6
    assert abs(kl(output_law(product, target, 4), target) - stored[("product", 4)]) < 5e-4
    assert abs(kl(output_law(product, target, 16), target) - stored[("product", 16)]) < 5e-5


def test_bounds_bracket_the_exact_kl(cell):
    model, classes, constraint, chain = cell
    target = target_law(model, classes, constraint, LENGTH)
    product = one_particle_law(model, classes, constraint, LENGTH, schedule=LeftToRight())
    exact = kl(output_law(product, target, 4), target)
    out = weighted_sampler(
        model,
        classes,
        constraint,
        LENGTH,
        LENGTH,
        runs=20000,
        particles=4,
        schedule=LeftToRight(),
        resampling=NO_RESAMPLING,
        generator=torch.Generator().manual_seed(9),
    )
    assert upper_bound(out.log_Z_hat) > exact
    assert 0.7 < leading_term(out.log_weights) / exact < 1.4
