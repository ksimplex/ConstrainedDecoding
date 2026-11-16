# Parallel unmasking experiments

Experiments on the constraint-aware unmasking schedule: which positions to reveal next, and how many at once, under
twisted SMC for a masked diffusion model with a hard constraint. Findings and their status are in
`docs/parallel_unmasking_status.md`; the plan is in `docs/parallel_unmasking_plan.md`.

The sampler and the schedule live in the package (`twistroute/decode/smc.py`, `twistroute/decode/policy.py`).
This directory holds the studies that measure them.

## Layout

| Path | Contents |
|---|---|
| `compare.py` | Command line for the comparison studies on oracle models (exact answers known). |
| `pulib/cells.py` | Named oracle models and constraints; the random stream of each (cell, arm). |
| `pulib/arms.py` | Arm sets, one function per study, registered under the `--arms` name. |
| `pulib/runner.py` | Runs a cell arm by arm, with resume; assembles the output file. |
| `pulib/metrics.py` | Metrics of an arm against the exact constrained law; memory footprint of a worker. |
| `pulib/gate.py` | Gated resampling: a pilot without resampling decides where resampling is allowed. |
| `weight_trust.py` | Do intermediate weights predict final weights? |
| `twist_error.py` | Error of the product twist against the exact twist, and of a first-order chain stand-in. |
| `twist_smc.py` | Reference sequential sampler with a pluggable twist stand-in. |
| `mdlm_first_look.py` | Kernel, unconstrained audit, order inconsistency and constrained runs on MDLM (GPU). |
| `control_check.py`, `hidden_mass_check.py`, `first_look.py` | Earlier checks of the one-per-step control and of hidden mass. |
| `jobs/` | Queue scripts as they were run (local overnight queues, cluster batch files). |

Results are in `results/parallel_unmasking/`.

## Running a study

```
PYTHONPATH=. python experiments/parallel_unmasking/compare.py --arms fix --N 256 \
    --models hidden_0.85 --constraints copy_block copy_spread window two_window \
    --nrun 100 --K 16 --pilot 2000 --workers 3 --resume --out fix_N256.json
```

- `--arms` selects the study: `full`, `core`, `order`, `budget`, `interact`, `thick`, `seqorder`, `fix`, `resample`.
  Each is described by the docstring of its function in `pulib/arms.py`.
- `--resume` keeps finished cells, and finished arms within an interrupted cell. Every arm draws from its own
  named random stream, so a resumed run equals an uninterrupted one.
- `--only` keeps the arms whose name starts with one of the given prefixes.
- Output: `<out>` (rows per arm, meta per cell, the arguments) and `<out stem>.logZ.npz` (per-run log Z-hat minus
  the exact log p(C)).

## Adding things

- **A study.** Write a function of the cell in `pulib/arms.py` that returns a list of `Arm`, and decorate it with
  `@arm_set("name")`. Arm names are part of the results: the random stream is derived from the name, and analysis
  selects arms by name.
- **An order.** `CouplingPolicy(priority=f)` accepts a function of the step context that returns one score per
  position, higher first. Scoring only orders the candidates; admission (threshold, budget, guard, pull-in, cap)
  is unchanged. A named order is a branch in `CouplingPolicy.score`.
- **A model or constraint.** Add a name to `make_model` or `make_constraint` in `pulib/cells.py`. A constraint
  needs a route and an exact truth object.

## Tests and the no-regression rule

| Test file | What it fixes |
|---|---|
| `tests/test_pu_regression.py` | Every arm set on small cells against outputs frozen in `tests/golden/pu_compare.json` (296 arms, 15 cells), field by field. |
| `tests/test_policy.py`, `tests/test_policy_split.py` | The policy: route queries, the kernel, exactness, the admission rule, scoring as a function. |
| `tests/test_longchain.py`, `tests/test_resampling.py`, `tests/test_oracle.py` | Long-sequence schedules, resampling options, oracle models. |

A change to the sampler, the policy or a twist goes in under this protocol:

1. With the new option off, the frozen reference is reproduced exactly.
2. With it on, Z-hat stays unbiased where the theory says so (tested against the exact p(C)).
3. It is compared with the previous default cell by cell on the benchmark, not on medians alone.
4. It stays opt-in until it has passed the first three.

The reference is regenerated only when a change in behaviour is intended (`python tests/test_pu_regression.py`),
and the commit says so. It is tied to the platform and torch version it was made on; elsewhere the comparison is
skipped, because multinomial draws can flip on last-bit floating-point differences.
