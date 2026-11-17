"""Forced-only batch step of CouplingPolicy (forced_batch=...): padding-like constraint on a small chain."""
import math
import numpy as np
import torch
from twistroute.compile.automata import WDFA
from twistroute.models import ConstrainedTruth, OracleHMM
from twistroute.torch_routes import TorchChain
from twistroute.decode import twisted_smc, CouplingPolicy, fit_distance_kernel, COUNT_EDGES

N, S = 8, 3


def setup(delta):
    T = np.array([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [delta / 2, delta / 2, 1 - delta]])
    den = OracleHMM([0.45, 0.45, 0.1], np.tile(T, (N - 1, 1, 1)), np.tile(np.eye(S), (N, 1, 1)))
    d = WDFA(2, [(0, 0, 0), (0, 1, 0), (0, 2, 1), (1, 2, 1)], 0, [1], S)             # content* end+
    cl = den.classes(); kern, _ = fit_distance_kernel(den, N, 400, torch.Generator().manual_seed(1), count_edges=COUNT_EDGES)
    return den, cl, d, TorchChain(d), ConstrainedTruth(den, cl, d), kern


def run(den, cl, route, kern, fb, nrun=6000, K=2):
    pol = CouplingPolicy(kern, 0.0, priority="random", mixed=False, max_set=1, forced_batch=fb)
    o = twisted_smc(den, cl, route, N, N, nrun=nrun, K=K, gen=torch.Generator().manual_seed(7), return_particles=True, schedule=pol)
    return o, pol


def test_batch_saves_steps_and_stays_valid():
    den, cl, d, route, truth, kern = setup(0.001)
    o, pol = run(den, cl, route, kern, float("inf"))
    toks = np.asarray(o["tokens"]).reshape(-1, N)
    assert all(d.accepts(r.tolist()) for r in toks[:500])
    assert float(np.mean(np.asarray(o["steps"], dtype=float))) < N - 0.5
    assert pol.forced_reveals > pol.forced_steps                                    # some step revealed several forced positions


def test_tolerance_keeps_the_estimate_within_its_bound():
    for delta, tol in ((0.001, 0.01), (0.3, 0.01), (0.3, 0.2)):
        den, cl, d, route, truth, kern = setup(delta)
        o, _ = run(den, cl, route, kern, tol)
        z = np.exp(np.asarray(o["logZ"], dtype=np.float64) - truth.log_pC); se = z.std(ddof=1) / math.sqrt(len(z))
        assert math.exp(-tol) - 4 * se <= z.mean() <= math.exp(tol) + 4 * se, (delta, tol, z.mean(), se)


def test_without_tolerance_nothing_changes():
    den, cl, d, route, truth, kern = setup(0.3)
    a, _ = run(den, cl, route, kern, None, nrun=200); b, pol = run(den, cl, route, kern, None, nrun=200)
    assert np.array_equal(np.asarray(a["tokens"]), np.asarray(b["tokens"])) and pol.forced_steps == 0
