"""Table 3 of the notes: KL divergence KL(q_K || pi), in nats, lower is better, from the law
q_K of the sampler's output to the constrained target pi, with one position unmasked per step,
the product proposal and no resampling.

    python experiments/notes/table3.py                      # prints the table, about 1 minute
    python experiments/notes/table3.py --out table3.json    # also writes the values before
                                                            # rounding, their Monte Carlo
                                                            # standard errors and the setting

results/notes/table3.json is the output of `--batches 40 --draws 200000` (about 10 minutes).

Setting (`cells.py`)
    model        hidden-chain oracle: a hidden state that stays put with probability 0.85,
                 observed through noise 0.3; L = 8 positions
    constraints  Counting: token 0 occurs exactly L / 3 = 2 times (3 tokens);
                 Dyck: balanced brackets (2 tokens); Forbidden motifs: 00 and 12 never occur
                 (3 tokens)
    orders       left to right, and two fixed random permutations of the positions per
                 constraint (`RANDOM_ORDERS`: the ones the notes' table was made with, recovered
                 from the random stream of experiments/studies/kl_bounds_check.py)

Columns
    Unweighted   one particle and no weights (the sampler of Dang and Ermon). Exact.
    K = 4, 16    the weighted sampler with K particles. The sum over sequences is exact. The
                 expectation over the other particles' weights is exact when every combination
                 can be listed (Dyck at K = 4) and otherwise a mean over `--batches` x `--draws`
                 draws, with the standard error stored. An entry whose fourth decimal could
                 change within two standard errors is marked with *.

"Exact" in the notes' caption means that the KL divergence itself is computed, not a proxy
for it. The K > 1 columns still carry Monte Carlo error in the last digit shown: the notes'
table was made with 20 000 draws, and results/notes/table3.json holds the values before
rounding with their standard errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import TITLES, cell  # noqa: E402

from twistroute.evaluation import kl_of_output, one_particle_law, target_law  # noqa: E402
from twistroute.schedules import FixedOrder, LeftToRight  # noqa: E402
from twistroute.twists import ProductTwist  # noqa: E402

LENGTH = 8
PARTICLES = (1, 4, 16)
RANDOM_ORDERS = {
    "counting": {"random A": [2, 4, 3, 6, 5, 0, 1, 7], "random B": [6, 2, 7, 4, 5, 1, 0, 3]},
    "dyck": {"random A": [0, 5, 4, 1, 2, 7, 3, 6], "random B": [0, 6, 3, 1, 7, 2, 5, 4]},
    "forbidden": {"random A": [4, 2, 5, 3, 1, 6, 0, 7], "random B": [2, 1, 4, 0, 6, 7, 5, 3]},
}


def shown(value: float, error: float) -> str:
    """Four decimals; * when two standard errors reach the nearest rounding boundary."""
    distance = abs((value * 1e4) % 1.0 - 0.5) * 1e-4
    return f"{value:.4f}" + ("*" if 2.0 * error > distance else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--draws", type=int, default=100_000, help="draws per batch when K > 1")
    parser.add_argument("--batches", type=int, default=8, help="independent batches when K > 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the unrounded values to this file")
    args = parser.parse_args()
    rows = []
    for name in ("counting", "dyck", "forbidden"):
        model, token_classes, constraint = cell(name, LENGTH)
        target = target_law(model, token_classes, constraint, LENGTH)
        schedules = {"left to right": LeftToRight()}
        for order_name, order in RANDOM_ORDERS[name].items():
            schedules[order_name] = FixedOrder(order)
        for order_name, schedule in schedules.items():
            one = one_particle_law(
                model, token_classes, constraint, LENGTH, schedule=schedule, twist=ProductTwist()
            )
            values = {}
            errors = {}
            for particles in PARTICLES:
                values[particles], errors[particles] = kl_of_output(
                    one,
                    target,
                    particles,
                    n_draws=args.draws,
                    n_batches=args.batches,
                    seed=args.seed,
                )
            rows.append(
                {"constraint": name, "order": order_name, "kl": values, "standard_error": errors}
            )
    print("| Constraint | Unmasking order | Unweighted | K = 4 | K = 16 |")
    print("|---|---|---|---|---|")
    for row in rows:
        cells = " | ".join(shown(row["kl"][k], row["standard_error"][k]) for k in PARTICLES)
        print(f"| {TITLES[row['constraint']]} | {row['order']} | {cells} |")
    if args.out is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    setting = {"length": LENGTH, "draws": args.draws, "batches": args.batches, "seed": args.seed}
    with open(args.out, "w") as handle:
        json.dump({**setting, "random_orders": RANDOM_ORDERS, "rows": rows}, handle, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
