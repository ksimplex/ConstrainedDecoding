# Parallel unmasking: status

**Branch:** `parallel-unmasking` (from `main` at `a085185`). **Started:** 2026-11-14.
This file tracks the constraint-aware parallel unmasking thread only. `docs/status.md` remains the record for
everything else and is not edited from this branch. The design is in `docs/parallel_unmasking_plan.md`.

Labels: **[verified]** = checked by a test or a run named here; **[derived]** = worked out, not run;
**[hypothesis]** = untested.

## 1. Scope

Reduce denoiser calls in trajectory-exact constrained decoding by choosing which positions to reveal together.
The sampler of the method notes is unchanged; only the reveal rule changes. The mRNA code (`twistroute_bio/`,
`tests/test_bio_*`, the mRNA experiments) is out of scope: not used, not modified, not run.

First build, agreed 2026-11-14: 1-D oracle denoisers, the schedule policy, the r measurement, tests.

## 2. Working agreements (2026-11-14)

- Ask before a major experiment (a lot of compute or a lot of code) and before a major change of direction.
- Cheap, fast checks need no approval; say what is being run and why.
- Exploratory phase. The long-run tests of the direction are (a) compute spent against compute saved at
  reasonable scales, and (b) whether the sparsity it needs occurs in practice.
- This is for a research paper: every addition on top of the base algorithm has to be justified. Prefer
  general fixes to local patches; do not tune to the chosen test cases.
- Parameters, sample sizes and test layout are the implementer's call.
- Replies end with "needed from you" and "awaiting approval" lists only when there is something in them.
- Another session may edit the repository later; the user will say when that starts.

## 3. Done

| date | item | status |
|---|---|---|
| 2026-11-14 | Branch created; cluster reachable; 2-CPU `mit_quicktest` smoke job ran end to end in about 3 min | [verified] |
| 2026-11-14 | `twistroute/models/oracle.py`: `OracleHMM` (exact conditionals of a known hidden Markov model, denoiser interface), constructors `lazy_chain`, `noisy_hmm`, `independent`, `parallel_log_prob` (for the r measurement), `ConstrainedTruth` (exact p(. \| C) for an automaton constraint) | [verified] by `tests/test_oracle.py`, 14 tests |
| 2026-11-14 | Full non-mRNA test suite on this branch, local CPU, torch 2.14.1: 217 passed (203 existing + 14 new) in 12 min 44 s; the five `tests/test_bio_*` files excluded | [verified] |

## 4. Findings so far

- **Random co-reveals already move the target on a toy chain.** [verified, one configuration] Lazy chain, N = 8,
  3 tokens, lam = 0.6, constraint "exactly three tokens of class 0", exact p(C) = 0.13337. With the existing
  fixed schedule and 6,000 runs: one reveal per step gives mean Z_hat 0.13187 (K = 1, s.e. 0.00102) and 0.13443
  (K = 4, s.e. 0.00054); two random reveals per step give 0.15839 (s.e. 0.00147, 17 s.e. from p(C)); four give
  0.19314 (s.e. 0.00154, 39 s.e.). The sampler is exact for each schedule's own law; the laws differ.
- **Exact constrained marginals are one backward pass away for automaton and copy routes.** [verified, small
  cases] Autograd through `logZ` matches enumerated marginals to about 3e-8 for `TorchChain` and 7e-8 for
  `TorchCopy`. For `TorchChart` the gradient came back NaN; not investigated.
- **The product compiler does not cover model x constraint.** Routes take per-position weights only, so the
  exact constrained target for a chain model needed its own forward pass (`ConstrainedTruth`).
- **`SyntheticMDLM` is not a sparsity testbed.** It couples every pair of positions through a dense random
  matrix.

## 5. How the plan maps onto the existing code

- `reveal_indices` already returns padded per-particle reveal sets with a validity mask; a new policy can
  return the same shape as a third `schedule` option.
- The loop runs exactly T steps and calls `accepts` only at the last one. A variable-size policy needs early
  finishers handled, including the switch to `accepts` when a particle completes.
- Fixed-seed golden tests pin the random-number order, so existing schedules must stay bit-identical.

## 6. Constraint set (proposal, awaiting approval)

Chosen so that the constraints couple positions in different ways.

| constraint | builder | how it couples positions |
|---|---|---|
| forbidden motifs | `dfa_forbidden` | local, short memory; the automaton state is pinned by the last few tokens |
| counting (exactly k, or a band) | `dfa_exactly_k`, `dfa_fraction_band` | global and weak per pair; never factorises until it saturates |
| copy | `TorchCopy` | hard, pairwise, long-range; exact blocks of two |
| Dyck-1, unambiguous | chart | nested, through a counter; forcing depends on revealed values |
| required motif somewhere | `dfa_contains` | disjunctive and position-uncertain; weak until few opportunities remain, then forcing |
| counts per segment | new, via `build_dfa` | exact multi-position blocks that never interact; can be aligned or misaligned with the model's segments |
| sliding-window all-different | `dfa_window_alldiff` | dense, local, hard; long forcing cascades |
| parity | new, via `build_dfa` | every position coupled until the last one, yet only one bit crosses any cut |
| forbidden motifs and counting | `product_dfa` | local and global together |

First four as the core set; the other five as stress cases. Ambiguous grammars are left out for now.

## 7. Next steps

1. Constraint set approved or amended.
2. Schedule policy as a third `schedule` option, with early finishers handled.
3. Pilot kernel estimation on the oracle chains, compared with the exact `pair_mi`.
4. r measurement for the baseline schedule and for the policy.
5. First comparison on the copy family (plan, section 8, check 6).
