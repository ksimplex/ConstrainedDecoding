"""Frozen reference for the restructuring of the sampler (branch notes-structure, 2026-11-20).

Runs the sampler as it stood at tag `pre-notes-refactor` (twistroute.decode.twisted_smc) on small oracle cells, over every
combination of twist, schedule and resampling option that stays in the new core, with fixed random streams, and stores the
per-run log Z_hat, the particles' tokens and log-weights, and the steps. The new core must reproduce the file exactly
(tests/test_core_matches_legacy.py). Regenerate only from the tag:

    PYTHONPATH=. python tests/reference/make_sampler_reference.py
"""
import os
import sys
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "studies"))
from pulib.cells import make_constraint, make_model, n_symbols  # noqa: E402
from twistroute.decode import (COUNT_EDGES, ChainTwist, ConfidenceSchedule, CouplingPolicy, DilatedSchedule,  # noqa: E402
                               fit_distance_kernel, fit_pair_kernel, twisted_smc)

L, NRUN, K = 16, 24, 4
CELLS = [("hidden_0.85", "counting"), ("hidden_0.85", "dyck"), ("hidden_0.85", "forbidden"), ("hidden_0.85", "window"),
         ("chain_0.85", "counting"), ("latent_0.6_w0.5", "forbidden")]


def seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def gen(*parts):
    return torch.Generator().manual_seed(seed(*parts))


def configurations(model, classes, truth):
    """name -> (keyword arguments of twisted_smc as a function, number of steps)."""
    kern = fit_distance_kernel(model, L, 300, gen("kernel"), count_edges=COUNT_EDGES)[0]
    out = {
        "product | random | plain": (lambda: dict(), L),
        "product | random | no resampling": (lambda: dict(ess_min=0.0), L),
        "product | random | defensive 0.25": (lambda: dict(defensive=0.25), L),
        "product | random | weight power 0.5": (lambda: dict(resample_power=0.5), L),
        "product | random | tempered twist tau 0.5": (lambda: dict(twist_temp=0.5), L),
        "product | random | twist power 0.5": (lambda: dict(twist_power=0.5), L),
        "product | left to right": (lambda: dict(schedule=ConfidenceSchedule(1, None, "left", forced_first=True)), L),
        "product | confidence, token": (lambda: dict(schedule=ConfidenceSchedule(1, None, "token")), L),
        "product | confidence, class": (lambda: dict(schedule=ConfidenceSchedule(1, None, "class")), L),
        "product | confidence, entropy": (lambda: dict(schedule=ConfidenceSchedule(1, None, "entropy", forced_first=True)), L),
        "product | blocks of 8, two per step": (lambda: dict(schedule=ConfidenceSchedule(2, 8, "token")), L // 2),
        "product | sum, one per step": (lambda: dict(schedule=CouplingPolicy(kern, 0.0, priority="sum", mixed=False, max_set=1)), L),
        "product | sum, budget 0.03, pull-in": (lambda: dict(schedule=CouplingPolicy(kern, 0.0, priority="sum", mixed=False, rate=0.03 / L, pull_forced=8)), L),
        "product | sum, budget 0.3": (lambda: dict(schedule=CouplingPolicy(kern, 0.0, priority="sum", mixed=False, rate=0.3 / L)), L),
        "product | sum, one per step, forced batch 0.01": (lambda: dict(schedule=CouplingPolicy(kern, 0.0, priority="sum", mixed=False, max_set=1, forced_batch=0.01)), L),
        "product | dilated, base 2": (lambda: dict(schedule=DilatedSchedule(2)), L),
        "product | two per step, random": (lambda: dict(), L // 2),
    }
    if hasattr(truth, "wdfa"):
        table = fit_pair_kernel(model, classes, 2000, gen("pair table"))
        chain = ChainTwist(truth.wdfa, table, active=getattr(truth, "active", None), match=5)
        ratio = ChainTwist(truth.wdfa, table, active=getattr(truth, "active", None), match=0)
        both = dict(twist=chain, proposal_twist=chain)
        out.update({
            "matched chain | random | plain": (lambda: dict(both), L),
            "matched chain | random | product share 0.1": (lambda: dict(both, proposal_mix=0.1), L),
            "matched chain | random | defensive 0.25": (lambda: dict(both, defensive=0.25), L),
            "matched chain | left to right": (lambda: dict(both, schedule=ConfidenceSchedule(1, None, "left", forced_first=True)), L),
            "matched chain | sum, one per step": (lambda: dict(both, schedule=CouplingPolicy(kern, 0.0, priority="sum", mixed=False, max_set=1)), L),
            "matched chain | sum, budget 0.03, pull-in": (lambda: dict(both, schedule=CouplingPolicy(kern, 0.0, priority="sum", mixed=False, rate=0.03 / L, pull_forced=8)), L),
            "matched chain as twist only | random": (lambda: dict(twist=chain), L),
            "unmatched chain | random | plain": (lambda: dict(twist=ratio, proposal_twist=ratio), L),
        })
    return out


def main(path=os.path.join(HERE, "sampler_reference.npz")):
    store = {}
    for mname, cname in CELLS:
        model = make_model(mname, L, n_symbols(cname)); classes, constraint, truth = make_constraint(cname, model, L)
        for name, (kwargs, steps) in configurations(model, classes, truth).items():
            key = f"{mname} | {cname} | {name}"
            o = twisted_smc(model, classes, constraint, L, steps, nrun=NRUN, K=K, gen=gen(key), return_particles=True, **kwargs())
            store[key + " | log_Z_hat"] = np.asarray(o["logZ"], dtype=np.float64)
            store[key + " | tokens"] = np.asarray(o["tokens"]).astype(np.int16)
            store[key + " | log_weights"] = np.asarray(o["logw"], dtype=np.float64)
            store[key + " | steps"] = np.asarray(o["steps"]).astype(np.int16)
            store[key + " | resamplings"] = np.asarray(o["n_res"], dtype=np.float64)
        store[f"{mname} | {cname} | log_p_C"] = np.asarray(truth.log_pC, dtype=np.float64)
        print(f"{mname:16s} {cname:10s} {sum(1 for k in store if k.startswith(f'{mname} | {cname} |') and k.endswith('log_Z_hat')):3d} configurations", flush=True)
    np.savez_compressed(path, **store)
    return store


if __name__ == "__main__":
    s = main(); print(len([k for k in s if k.endswith("log_Z_hat")]), "cases ->", "tests/reference/sampler_reference.npz")
