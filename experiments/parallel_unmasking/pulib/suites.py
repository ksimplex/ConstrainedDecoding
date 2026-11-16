"""Named benchmark suites: a fixed set of cells and the run settings they have been run with.

`compare.py --suite NAME` takes the cells and the defaults from here; explicit flags still override. A result
reported "on long256" then always means the same 16 cells, and two runs of a suite can be compared cell by cell.
"""
LONG = ("window", "two_window", "copy_spread", "copy_block")
CORE = ("forbidden", "counting", "copy", "dyck")

SUITES = {
    # whole-sequence constraints at a length where every exact answer is cheap
    "core64": dict(N=[64], models=["chain_0.6", "chain_0.85", "hidden_0.85", "segments_0.85"], constraints=list(CORE),
                   nrun=2000, K=4, pilot=2000),
    # constraints on a tenth of the positions, Markov and hidden chains
    "long256": dict(N=[256], models=["chain_0.6", "chain_0.85", "hidden_0.85", "repel_0.8"], constraints=list(LONG),
                    nrun=100, K=16, pilot=2000),
    # the evaluation set: hidden chains at three noise levels (nothing separates exactly)
    "hidden256": dict(N=[256], models=["hidden_0.85_n0.15", "hidden_0.85", "hidden_0.85_n0.5"], constraints=list(LONG),
                      nrun=100, K=16, pilot=2000),
    "long1024": dict(N=[1024], models=["chain_0.85", "hidden_0.85"], constraints=["copy_spread", "two_window"],
                     nrun=50, K=16, pilot=300),
    # seconds, for checking that a command line works
    "smoke": dict(N=[32], models=["hidden_0.85"], constraints=["copy", "counting"], nrun=8, K=4, pilot=60),
}


def cells(name):
    s = SUITES[name]
    return [(m, c, N) for N in s["N"] for m in s["models"] for c in s["constraints"]]
