"""The new core (twistroute.sampler, twists, schedules, resampling) reproduces the sampler as it
stood at tag `pre-notes-refactor`, exactly: same log Z_hat, particles, weights, steps and
resampling counts on every case of tests/reference/sampler_reference.npz (150 cases: six
oracle cells by every twist, schedule and resampling option the core keeps)."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "reference"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments", "studies"))
from make_sampler_reference import CELLS, K, L, NRUN, gen  # noqa: E402
from pulib.cells import make_constraint, make_model, n_symbols  # noqa: E402

from twistroute.constraints import Constraint  # noqa: E402
from twistroute.resampling import Resampling  # noqa: E402
from twistroute.sampler import weighted_sampler  # noqa: E402
from twistroute.schedules import (  # noqa: E402
    Blocks,
    BudgetedCoReveal,
    Confidence,
    Dilated,
    LeftToRight,
    Sum,
    fit_distance_dependence,
)
from twistroute.twists import ChainTwist, ProductTwist, counted_pair_table  # noqa: E402

REFERENCE = np.load(os.path.join(HERE, "reference", "sampler_reference.npz"))


def configurations(model, classes, truth):
    """name -> (keyword arguments of weighted_sampler as a function, number of steps)."""
    dependence = fit_distance_dependence(model, L, 300, gen("kernel"))
    out = {
        "product | random | plain": (lambda: dict(), L),
        "product | random | no resampling": (lambda: dict(resampling=Resampling(threshold=0.0)), L),
        "product | random | defensive 0.25": (
            lambda: dict(resampling=Resampling(defensive=0.25)),
            L,
        ),
        "product | random | weight power 0.5": (
            lambda: dict(resampling=Resampling(weight_power=0.5)),
            L,
        ),
        "product | random | tempered twist tau 0.5": (
            lambda: dict(twist=ProductTwist(temperature=0.5)),
            L,
        ),
        "product | random | twist power 0.5": (lambda: dict(twist=ProductTwist(power=0.5)), L),
        "product | left to right": (lambda: dict(schedule=LeftToRight(forced_first=True)), L),
        "product | confidence, token": (lambda: dict(schedule=Confidence("token")), L),
        "product | confidence, class": (lambda: dict(schedule=Confidence("class")), L),
        "product | confidence, entropy": (
            lambda: dict(schedule=Confidence("entropy", forced_first=True)),
            L,
        ),
        "product | blocks of 8, two per step": (
            lambda: dict(schedule=Blocks(8, 2, "token")),
            L // 2,
        ),
        "product | sum, one per step": (lambda: dict(schedule=Sum(dependence)), L),
        "product | sum, budget 0.03, pull-in": (
            lambda: dict(schedule=BudgetedCoReveal(dependence, 0.03)),
            L,
        ),
        "product | sum, budget 0.3": (
            lambda: dict(schedule=BudgetedCoReveal(dependence, 0.3, pull_in=0)),
            L,
        ),
        "product | sum, one per step, forced batch 0.01": (
            lambda: dict(schedule=Sum(dependence, forced_batch=0.01)),
            L,
        ),
        "product | dilated, base 2": (lambda: dict(schedule=Dilated(2)), L),
        "product | two per step, random": (lambda: dict(), L // 2),
    }
    if hasattr(truth, "wdfa"):
        table = counted_pair_table(model, classes, 2000, gen("pair table"))
        active = getattr(truth, "active", None)

        def chain(**options):
            return ChainTwist(truth.wdfa, table, active=active, **options)

        out.update(
            {
                "matched chain | random | plain": (lambda: dict(twist=chain()), L),
                "matched chain | random | product share 0.1": (
                    lambda: dict(twist=chain(proposal_product_share=0.1)),
                    L,
                ),
                "matched chain | random | defensive 0.25": (
                    lambda: dict(twist=chain(), resampling=Resampling(defensive=0.25)),
                    L,
                ),
                "matched chain | left to right": (
                    lambda: dict(twist=chain(), schedule=LeftToRight(forced_first=True)),
                    L,
                ),
                "matched chain | sum, one per step": (
                    lambda: dict(twist=chain(), schedule=Sum(dependence)),
                    L,
                ),
                "matched chain | sum, budget 0.03, pull-in": (
                    lambda: dict(twist=chain(), schedule=BudgetedCoReveal(dependence, 0.03)),
                    L,
                ),
                "matched chain as twist only | random": (
                    lambda: dict(twist=chain(proposal="product")),
                    L,
                ),
                "unmatched chain | random | plain": (
                    lambda: dict(twist=chain(matching_sweeps=0)),
                    L,
                ),
            }
        )
    return out


@pytest.mark.parametrize("model_name,constraint_name", CELLS)
def test_core_reproduces_the_frozen_sampler(model_name, constraint_name):
    model = make_model(model_name, L, n_symbols(constraint_name))
    classes, route, truth = make_constraint(constraint_name, model, L)
    constraint = Constraint(route, family="reference cell", cost="", n_classes=classes.A, length=L)
    cases = configurations(model, classes, truth)
    expected = [
        k
        for k in REFERENCE.files
        if k.startswith(f"{model_name} | {constraint_name} |") and k.endswith("log_Z_hat")
    ]
    assert len(expected) == len(cases)
    for name, (options, n_steps) in cases.items():
        key = f"{model_name} | {constraint_name} | {name}"
        out = weighted_sampler(
            model,
            classes,
            constraint,
            L,
            n_steps,
            runs=NRUN,
            particles=K,
            generator=gen(key),
            **options(),
        )
        assert np.array_equal(out.log_Z_hat, REFERENCE[key + " | log_Z_hat"]), key
        assert np.array_equal(out.tokens, REFERENCE[key + " | tokens"]), key
        assert np.array_equal(out.log_weights, REFERENCE[key + " | log_weights"]), key
        assert np.array_equal(out.steps, REFERENCE[key + " | steps"]), key
        assert np.array_equal(out.resamplings, REFERENCE[key + " | resamplings"]), key


def test_options_outside_the_frozen_file_match_the_old_sampler():
    """Options the core keeps but the frozen file does not cover, compared with the old sampler
    run side by side: systematic resampling, one particle, a chain twist with power or
    temperature, defensive resampling combined with a weight power."""
    from twistroute.decode import ChainTwist as OldChainTwist
    from twistroute.decode import twisted_smc

    model = make_model("hidden_0.85", L, n_symbols("forbidden"))
    classes, route, truth = make_constraint("forbidden", model, L)
    constraint = Constraint(route, family="reference cell", cost="", n_classes=classes.A, length=L)
    table = counted_pair_table(model, classes, 2000, gen("pair table"))
    pairs = [
        (dict(resampling=Resampling(scheme="systematic")), dict(resampler="systematic"), K),
        (
            dict(resampling=Resampling(defensive=0.25, weight_power=0.5)),
            dict(defensive=0.25, resample_power=0.5),
            K,
        ),
        (dict(), dict(), 1),
        (
            dict(twist=ChainTwist(truth.wdfa, table, power=0.5)),
            dict(
                twist=OldChainTwist(truth.wdfa, table, match=5),
                proposal_twist=OldChainTwist(truth.wdfa, table, match=5),
                twist_power=0.5,
            ),
            K,
        ),
        (
            dict(twist=ChainTwist(truth.wdfa, table, temperature=0.7, proposal="product")),
            dict(twist=OldChainTwist(truth.wdfa, table, match=5, temp=0.7)),
            K,
        ),
        (
            dict(
                twist=ProductTwist(temperature=0.7, power=0.5),
                resampling=Resampling(defensive=0.25),
            ),
            dict(twist_temp=0.7, twist_power=0.5, defensive=0.25),
            K,
        ),
    ]
    for number, (new_options, old_options, particles) in enumerate(pairs):
        new = weighted_sampler(
            model,
            classes,
            constraint,
            L,
            L,
            runs=NRUN,
            particles=particles,
            generator=gen("extra", number),
            **new_options,
        )
        old = twisted_smc(
            model,
            classes,
            route,
            L,
            L,
            nrun=NRUN,
            K=particles,
            gen=gen("extra", number),
            return_particles=True,
            **old_options,
        )
        assert np.array_equal(new.log_Z_hat, old["logZ"]), number
        assert np.array_equal(new.tokens, old["tokens"]), number
        assert np.array_equal(new.log_weights, old["logw"]), number


def test_a_different_option_does_not_match():
    """Control for the comparison itself: a slightly different resampling rule must not reproduce
    the reference."""
    model = make_model("hidden_0.85", L, n_symbols("forbidden"))
    classes, route, truth = make_constraint("forbidden", model, L)
    constraint = Constraint(route, family="reference cell", cost="", n_classes=classes.A, length=L)
    key = "hidden_0.85 | forbidden | product | random | defensive 0.25"
    out = weighted_sampler(
        model,
        classes,
        constraint,
        L,
        L,
        runs=NRUN,
        particles=K,
        generator=gen(key),
        resampling=Resampling(defensive=0.26),
    )
    assert not np.array_equal(out.log_Z_hat, REFERENCE[key + " | log_Z_hat"])
