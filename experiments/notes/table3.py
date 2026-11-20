"""Table 3 of the notes: exact KL divergence (nats, lower is better) from the law of the
sampler's output to the constrained target, on the hidden-chain oracle with L = 8 positions,
one position unmasked per step, product proposal, no resampling.

    python experiments/notes/table3.py

prints the table and writes results/notes/table3.json.

Columns: "Unweighted" is one particle (the sampler of Dang and Ermon); 4 and 16 are the number
of particles K of the weighted sampler. The one-particle column is exact. For K > 1 the sum
over sequences is exact and the expectation over the other particles' weights is a mean over
`--draws` draws (see `twistroute.evaluation.output_law`), so the last digit can move by one.

The two random orders are fixed permutations of the positions, drawn once per constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import TITLES, cell  # noqa: E402

from twistroute.evaluation import kl, one_particle_law, output_law, target_law  # noqa: E402
from twistroute.schedules import FixedOrder, LeftToRight  # noqa: E402
from twistroute.twists import ProductTwist  # noqa: E402

LENGTH = 8
PARTICLES = (1, 4, 16)
RANDOM_ORDERS = {
    "counting": {"random A": [2, 4, 3, 6, 5, 0, 1, 7], "random B": [6, 2, 7, 4, 5, 1, 0, 3]},
    "dyck": {"random A": [0, 5, 4, 1, 2, 7, 3, 6], "random B": [0, 6, 3, 1, 7, 2, 5, 4]},
    "forbidden": {"random A": [4, 2, 5, 3, 1, 6, 0, 7], "random B": [2, 1, 4, 0, 6, 7, 5, 3]},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--draws", type=int, default=200_000, help="draws for the expectation when K > 1"
    )
    parser.add_argument("--out", default=os.path.join("results", "notes", "table3.json"))
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
            for particles in PARTICLES:
                law = output_law(
                    one, target, particles, n_draws=args.draws, rng=np.random.default_rng(0)
                )
                values[particles] = kl(law, target)
            rows.append({"constraint": name, "order": order_name, "kl": values})
    print("| Constraint | Unmasking order | Unweighted | K = 4 | K = 16 |")
    print("|---|---|---|---|---|")
    for row in rows:
        values = " | ".join(f"{row['kl'][particles]:.4f}" for particles in PARTICLES)
        print(f"| {TITLES[row['constraint']]} | {row['order']} | {values} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(
            {"length": LENGTH, "draws": args.draws, "random_orders": RANDOM_ORDERS, "rows": rows},
            handle,
            indent=1,
        )
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
