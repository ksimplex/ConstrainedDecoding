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
| 2026-11-14 | `twistroute/decode/policy.py`: `CouplingPolicy` (greedy weakly-coupled reveal sets), `DistanceKernel`, `fit_distance_kernel` (pilot), exact `constrained_marginals` and `pairwise_mi` by autograd through `logZ`. `twisted_smc` accepts a policy object as `schedule`, handles particles that finish early, stops when all are complete, and returns `steps` and an optional `path_stat`. `native_sample` accepts the policy too. Named schedules are unchanged | [verified] by `tests/test_policy.py`, 10 tests; full non-mRNA suite 227 passed, golden tests included |
| 2026-11-14 | First-look smoke comparison, `experiments/parallel_unmasking/first_look.py`, N = 16, 300 runs x 4 particles, results in `results/parallel_unmasking/first_look_smoke.json` | [verified] that it runs; too small for conclusions |

## 4. Findings so far

- **Random co-reveals already move the target on a toy chain.** [verified, one configuration] Lazy chain, N = 8,
  3 tokens, lam = 0.6, constraint "exactly three tokens of class 0", exact p(C) = 0.13337. With the existing
  fixed schedule and 6,000 runs: one reveal per step gives mean Z_hat 0.13187 (K = 1, s.e. 0.00102) and 0.13443
  (K = 4, s.e. 0.00054); two random reveals per step give 0.15839 (s.e. 0.00147, 17 s.e. from p(C)); four give
  0.19314 (s.e. 0.00154, 39 s.e.). The sampler is exact for each schedule's own law; the laws differ.
- **Forced positions are not free to co-reveal.** [verified by a test] The first implementation revealed forced
  positions unconditionally, on the reasoning that they carry no randomness. That holds under the constrained
  law but not under the native law the weights are defined against: the native kernel draws a forced position
  from the model's marginal, so revealing it with a position the model couples it to drops
  p(x_a | x_forced) / p(x_a). On a chain with a count constraint, several adjacent positions become forced
  together when the count saturates, and the gap r reached 3.3 nats on single paths. Forced positions now only
  get priority; they pass through the coupling test like any other. The plan document is corrected (its
  section 12).
- **First look, smoke scale only (N = 16, 300 runs x 4 particles; the marginal-TV noise floor is about 0.02).**
  [verified that these are the numbers in the JSON; not a conclusion]
  - Chain lam = 0.6, exactly-k: fixed 2/step takes 8.0 steps with sd(r) 0.837 and fixed 4/step 4.0 steps with
    1.391; the policy at theta = 0 (no mixed term) takes 7.6 steps with sd(r) 0.000, at theta = 0.01 4.8 steps
    with 0.092, at theta = 0.03 4.5 steps with 0.174.
  - Chain lam = 0.6, forbidden motifs: fixed 4/step has marginal TV 0.0549 and sd(r) 1.363; the policy at
    theta = 0.03 takes 5.7 steps with marginal TV 0.0157 and sd(r) 0.242.
  - Hidden chain (no exact screening), exactly-k: theta = 0 falls back to 16 steps, as it should. theta = 0.01
    takes 6.7 steps with sd(r) 0.283 against fixed 2/step at 8.0 steps with 0.586; theta = 0.03 takes 4.9 steps
    with 0.508 against fixed 4/step with 0.886.
  - The mixed-path term made the schedule more conservative (chain, exactly-k, theta = 0: 12.5 steps against
    7.6) and cost 19.1 s against 1.1 s of wall time for that arm. Nothing in this run shows it buying a higher
    effective sample size. [hypothesis: at this scale it behaves like a lower theta]
  - Effective sample size is not higher under the policy than under the fixed schedule.
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

## 6. Constraint set (approved 2026-11-14)

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

## 7. Decisions since the plan document (v3), 2026-11-14 and 15

- **Target.** The model's underlying joint conditioned on the constraint, which is what one reveal per step
  samples. Every schedule, including the notes' fixed schedule with more than one reveal per step, is reported
  against it. The theorem stays "exact for the schedule's own law"; distance to the target is the quality metric.
- **Workflow.** Audit the gap r once per model and constraint family, choose theta, then stop checking. No
  rejection step at the end. Weighting outputs by exp(r) is reported as a side column only.
- **Baseline.** "Naive" means one reveal per step. Results are reported as a net account over every cell at
  theta = 0 (gains, ties and losses), with theta > 0 trade-offs separately. No worst-case safeguards unless a
  case is very bad.
- **Mixed-path term.** Off by default; every comparison that includes it is also run without it. It needs about
  3 route passes per position per step (97 at N = 32) and is not usable on long sequences.
- **Forced positions** get priority only and pass the coupling test like any other position.
- **Constraint coupling** is the exact normalised mutual information from autograd through `logZ`. Chart routes
  are unsupported until the NaN gradient of `TorchChart` is fixed; Dyck-1 uses the height automaton `dfa_dyck1`.
- **Offline planning** is limited to coarse quantities (set sizes per step, a switch between "create separation"
  and "exploit it"), never a fixed list of positions. [hypothesis, not built]
- **Scale.** N = 32 and 64 are a test of the mechanism, not a demonstration of speed. Wall time is not reported
  at this scale. The first real test is long chains, N = 128 to 1,024, Markov and hidden, with (1) copy on about
  10% of positions and (2) "at least k of n" on a contiguous window of about 10% with k binding. A chain with
  anticorrelated neighbours is to be added so that model and constraint can oppose each other. Purpose-built
  code is allowed for these. Not started; the run needs the user's go-ahead.

## 8. Comparison at N = 64 (cluster job `1c274eb8`, 2026-11-14 23:55 to 2026-11-15 00:38)

`experiments/parallel_unmasking/compare.py --N 64 --nrun 2000 --nrun-mixed 0`, 16 workers on `mit_preemptable`.
Results: `results/parallel_unmasking/compare_N64.json` (240 arms in 16 cells, 2504 s
wall, no non-finite values, every sample valid). Cluster unit tests: 30 passed (torch 2.5.1). No mixed arms.

**Steps at theta = 0, against 64 for one reveal per step.** [verified: read from the JSON]

| model | forbidden | counting | copy | dyck |
|---|---|---|---|---|
| `segments_0.85` | 6.6 | 6.1 | 5.0 | 6.4 |
| `chain_0.6` | 27.6 | 8.7 | 11.9 | 14.4 |
| `chain_0.85` | 18.4 | 9.1 | 14.0 | 14.8 |
| `hidden_0.85` | 63.0 | 62.8 | 63.0 | 56.9 |

- Segments: about 10 to 13 times fewer steps. Chains: 2.3 to 7.4 times fewer, and the constraint matters
  (chain 0.6: counting 8.7, forbidden motifs 27.6). Hidden chain: no gain at theta = 0, as expected.
- Chain 0.6 with counting took 7.9 steps at N = 32 (200-run timing run) and 8.7 at N = 64.

**The gap at theta = 0 is not exactly zero at this length.** [verified] sd(r) ranges from 4.5e-15 to
1.3e-04 over the 16 cells. [hypothesis] The fitted kernel underflows to zero at long range and admits a
few distant unscreened pairs.

**Wall times in this file are not comparable between arms.** [verified] The same one-per-step arm took 37 s
in one cell and 101 s in another of the same size, with 16 workers sharing the node. Not to be used.

**Control: mean Z_hat / p(C) for one reveal per step (jackknife s.e.).** [verified: read from the JSON]

| model | forbidden | counting | copy | dyck |
|---|---|---|---|---|
| `segments_0.85` | 0.849 (0.037) | 1.016 (0.046) | 0.868 (0.092) | 0.948 (0.044) |
| `chain_0.6` | 1.041 (0.094) | 0.995 (0.006) | 0.827 (0.039) | 1.041 (0.057) |
| `chain_0.85` | 0.956 (0.040) | 0.900 (0.034) | 0.812 (0.058) | 1.066 (0.087) |
| `hidden_0.85` | 0.943 (0.015) | 0.991 (0.009) | 0.887 (0.044) | 0.994 (0.020) |

**Same quantity for the policy at theta = 0.**

| model | forbidden | counting | copy | dyck |
|---|---|---|---|---|
| `segments_0.85` | 1.031 (0.074) | 0.966 (0.017) | 1.008 (0.005) | 0.969 (0.015) |
| `chain_0.6` | 0.978 (0.018) | 1.022 (0.012) | 1.007 (0.025) | 1.006 (0.028) |
| `chain_0.85` | 1.022 (0.044) | 0.941 (0.034) | 1.031 (0.027) | 0.945 (0.021) |
| `hidden_0.85` | 0.994 (0.017) | 0.991 (0.015) | 1.006 (0.015) | 1.004 (0.015) |

The control is low in 11 of 16 cells and by 2.5 to 4.4 s.e. in six. The policy arms at theta = 0 use the same
ground truth and sit at 1, so the truth is not the suspect. All arms ran with defensive resampling at 0.25 and
with the calibrated twist and tempering off (tau = 1, beta = 1).

**Resolution (2026-11-15).** [verified] The shortfall was one unlucky seed read against standard errors that
were too small. It is not a bias and not a difference between scripts.

- The exact p(C) is right: an independent forward recursion reproduces `ConstrainedTruth` to ten decimals on the
  five forbidden-motif cells, and direct sampling at N = 32 (hidden chain, forbidden motifs, 4,000,000 samples)
  gives 1.003 (0.007) of it.
- `experiments/parallel_unmasking/control_check.py` reruns the control with many seeds. Seed 3, the one every
  cell of the comparison shared, reproduces the original value exactly in all 16 cells
  (`control_seed3_check.json`; 22 to 26 seeds of 2,000 runs per cell before the job was pre-empted, pooled with
  the 20 seeds of the first job where the cell was in both).

| cell | seeds | original | seed 3 | pooled mean (s.e.) | true s.e. of a 2,000-run mean | seed 3 in true s.e. |
|---|---|---|---|---|---|---|
| `chain_0.6` copy | 26 | 0.827 | 0.827 | 1.004 (0.013) | 0.088 | -2.0 |
| `chain_0.6` counting | 23 | 0.995 | 0.995 | 0.996 (0.002) | 0.008 | -0.2 |
| `chain_0.6` dyck | 23 | 1.041 | 1.041 | 1.002 (0.003) | 0.020 | +2.0 |
| `chain_0.6` forbidden | 25 | 1.041 | 1.041 | 0.993 (0.012) | 0.062 | +0.8 |
| `chain_0.85` copy | 26 | 0.812 | 0.812 | 0.964 (0.035) | 0.239 | -0.6 |
| `chain_0.85` counting | 23 | 0.900 | 0.900 | 0.998 (0.008) | 0.053 | -1.8 |
| `chain_0.85` dyck | 23 | 1.066 | 1.066 | 0.964 (0.016) | 0.079 | +1.3 |
| `chain_0.85` forbidden | 25 | 0.956 | 0.956 | 0.997 (0.010) | 0.051 | -0.8 |
| `hidden_0.85` copy | 26 | 0.887 | 0.887 | 1.002 (0.013) | 0.089 | -1.3 |
| `hidden_0.85` counting | 23 | 0.991 | 0.991 | 0.998 (0.002) | 0.009 | -0.8 |
| `hidden_0.85` dyck | 22 | 0.994 | 0.994 | 0.999 (0.004) | 0.021 | -0.2 |
| `hidden_0.85` forbidden | 25 | 0.943 | 0.943 | 0.995 (0.004) | 0.027 | -1.9 |
| `segments_0.85` copy | 25 | 0.868 | 0.868 | 1.032 (0.084) | 0.421 | -0.4 |
| `segments_0.85` counting | 22 | 1.016 | 1.016 | 1.020 (0.015) | 0.070 | -0.0 |
| `segments_0.85` dyck | 22 | 0.948 | 0.948 | 0.993 (0.017) | 0.081 | -0.6 |
| `segments_0.85` forbidden | 25 | 0.849 | 0.849 | 1.017 (0.013) | 0.090 | -1.9 |

  Against the true standard error, seed 3 is within 2.0 of the pooled mean in every cell (sum of squares
  24.9 on 16 cells). It is the lowest of its seeds in 3 cells and the highest in 1.
- The jackknife standard errors of the comparison understate the error where Z_hat is heavy-tailed: by a factor
  of about 2 on chain 0.6 copy (0.039 reported, 0.088 true) and about 4 on chain 0.85 copy (0.058 against 0.239).
- Runs within one call are independent, so standard errors across seeds are no better than within-run ones at
  equal total runs. The remedy is more runs and a tail diagnostic, not more seeds.

## 8a. Hidden mass on the one-per-step control at N = 64

`control_check_N64.json`: 20 seeds x 2,000 runs x 4 particles per arm, seven cells, cluster job `613b10a7`
(18 min on 32 CPUs). Entries: mean Z_hat / p(C) (s.e. over 40,000 runs); seeds with mean below 1; share of the
total carried by the single largest run. [verified: read from the JSON]

| cell | resampling off | plain multinomial | defensive 0.25 | defensive 0.25, tau = 0.5 |
|---|---|---|---|---|
| `chain_0.6` copy | 1.009 (0.018); 10/20; 0.1% | 0.961 (0.024); 15/20; 1.1% | 0.996 (0.019); 11/20; 0.7% | 0.993 (0.038); 9/20; 1.6% |
| `chain_0.85` copy | 0.999 (0.005); 10/20; 0.0% | 0.781 (0.087); 17/20; 8.4% | 0.966 (0.074); 17/20; 7.1% | 0.995 (0.017); 12/20; 0.6% |
| `hidden_0.85` copy | 1.003 (0.009); 10/20; 0.1% | 0.987 (0.026); 13/20; 1.0% | 0.993 (0.017); 12/20; 0.7% | 0.996 (0.019); 12/20; 0.4% |
| `segments_0.85` forbidden | 1.068 (0.044); 5/20; 1.5% | 1.041 (0.028); 7/20; 1.3% | 1.022 (0.019); 8/20; 0.7% | 0.997 (0.022); 8/20; 0.5% |
| `hidden_0.85` forbidden | 1.006 (0.006); 8/20; 0.0% | 0.994 (0.009); 9/20; 0.3% | 1.001 (0.006); 10/20; 0.1% | 1.005 (0.007); 8/20; 0.0% |
| `chain_0.85` counting | 1.000 (0.003); 11/20; 0.1% | 0.964 (0.059); 16/20; 4.9% | 1.009 (0.013); 11/20; 0.6% | 1.007 (0.005); 8/20; 0.0% |
| `chain_0.6` dyck | 0.997 (0.004); 13/20; 0.0% | 1.005 (0.006); 8/20; 0.1% | 1.000 (0.004); 10/20; 0.1% | 0.995 (0.005); 9/20; 0.0% |

- **Hidden mass occurs under plain resampling on chain 0.85 with copy and with counting.** [verified] Mean 0.781
  and 0.964, 17 and 16 of 20 seeds below 1, one run carrying 8.4% and 4.9% of the total. Resampling off gives
  0.999 (0.005) and 1.000 (0.003) on the same cells.
- **Defensive resampling alone does not remove it on chain 0.85 with copy** (0.966 (0.074), 17 of 20 seeds
  below 1). Adding the calibrated twist at tau = 0.5 does (0.995 (0.017), largest run 0.6%).
- **tau = 0.5 is not uniformly better.** [verified] On chain 0.6 with copy it doubles the standard error (0.019
  to 0.038). [hypothesis] With an oracle denoiser the only twist error is the neglected coupling between masked
  positions, which is large on chain 0.85 and small on chain 0.6; flattening a twist that is already accurate
  only adds noise. The notes report the same direction on copy with resampling off (ratio 1.17).
- **Resampling off has the lowest or tied-lowest relative spread of Z_hat in six of the seven cells.** [verified]
  The exception is segments with forbidden motifs. This concerns the estimate of p(C) only, not sample quality.
- **The variance of log Z_hat is the wrong criterion for choosing tau.** [verified, `tail_criteria_check.json`]
  It picks the setting with the lowest true relative spread in 2 of 7 cells, and its worst pick is 16 times worse
  than the best (chain 0.85 copy, where it roughly doubles at tau = 0.5 although tau = 0.5 removes the hidden
  mass). Upper-tail criteria from 2,000-run pilots (relative spread of Z_hat, share of the top 1% of runs, Hill
  tail index; medians over 20 pilots) pick the best setting in 5 or 6 of 7 cells and are never more than 1.1
  times worse. [derived] The error of the mean is set by the upper tail of Z_hat; the log-variance is set mostly
  by the lower side, and the two agree only if Z_hat is close to log-normal.
- **Proposed rule for tau, not tested** (side project, deferred by agreement): pilot with resampling off, measure
  the share of final weight carried by particles that were far below their run's average at some step, and take
  the largest tau that keeps that share under a tolerance. A truth-free cross-check: mean Z_hat with resampling
  on should agree with resampling off.

## 8b. Guards now in `compare.py` (2026-11-15)

- Every cell and arm has its own random stream (`stream_seed`: crc32 of base seed, model, constraint, N, arm).
- Arm `fixed 1/step, no resampling`, which cannot hide mass, and a per-cell `control_check` (mean Z_hat of the
  one-per-step arm with and without resampling).
- Per arm: `top1_share`, `hill_k`, and `z_se_unreliable` when the tail index exceeds 1/2.
- Per-run log(Z_hat / p(C)) of every arm in `<out>.logZ.npz`.
- Arm `random theta=0.0`: the policy's sets with positions taken in random order.
- At N = 1,024 the one-per-step control will not give a usable Z_hat with or without resampling; rely on step
  counts, r and the exact marginals there.

**Open observation: reveal order may matter even without parallelism.** [verified that these are the numbers;
interpretation is a hypothesis] On the hidden chain at N = 32 the policy at theta = 0 takes all 32 steps, yet
the variance of log Z_hat falls from 1.09 to 0.17 (copy) and 0.55 to 0.17 (forbidden motifs) and the mean
minimum effective sample size rises from 0.38 to 0.53 and 0.44 to 0.66. Only the order of reveals differs from
the control. Over all cells the policy at theta = 0 has the lower variance of log Z_hat in 11 of 16 (N = 32) and
9 of 16 (N = 64); this cannot separate order from step count, and the measure is the one shown above to be
weak. The rerun with the guarded script has the arms to separate them.

**The gap at theta = 0 on chains is float underflow in the kernel.** [verified] The fitted unscreened kernel
is exactly zero from distance 22 on chain 0.6 (true mutual information 1.7e-10 there), from 57 on chain 0.85
and from 53 on the hidden chain at N = 64, and never zero on chain 0.85 and the hidden chain at N = 32. sd(r)
is about 1e-5 to 1e-4 in exactly the cells where the kernel reaches zero and about 1e-15 elsewhere.

**N = 32, local** (`compare_N32.json`, 16 cells, 336 arms, 3,501 s on 3 workers, shared seed 3): steps at
theta = 0 are 5.0 to 6.5 on segments, 7.9 to 22.0 on chain 0.6, 8.1 to 11.4 on chain 0.85, 32.0 on the hidden
chain. The control is within 2 s.e. of 1 in every cell.

## 9. Runs and machines

- **Cluster queue.** `mit_normal` gave start estimates of days on 2026-11-14; `mit_preemptable` started within
  minutes. `sbatch --test-only` gives the estimate before submitting. Maintenance on Tuesday 2026-11-17.
- **Local N = 32 run.** `compare.py --N 32 --nrun 2000 --nrun-mixed 500 --workers 3 --resume`, 2026-11-15 00:12
  to 01:10. Done.
- **Rerun with the guarded script.** Cluster job `a1d8bedb`, submitted 2026-11-15 02:05: N = 64 (no mixed arms)
  then N = 32 (mixed arms), 32 CPUs, resumable. Replaces the shared-seed files for analysis.
- **Pre-emption.** `mit_preemptable` pre-empted a 32-CPU job after about 27 min; jobs there must be resumable.
  `sbatch --test-only` overestimates the wait on that partition by hours.
- **Laptop.** 8 cores, 16 GB. Six workers pushed compressed memory to about 7.9 GB and load to about 12. Three
  workers lower the load but not the memory. The session's Python kernel was replaced several times at gaps in
  the conversation; cause unknown. `compare.py --resume` restarts from the saved cells.

## 10. Next steps

1. Analyse the guarded rerun at N = 32 and 64: net account at theta = 0, theta > 0 trade-offs, mixed term against its
   reference arm, priority scores, pilot size, Var(r)/2 against measured error.
2. Reveal order without parallelism: control against `random theta=0.0` against `policy theta=0.0`.
3. Long-chain code: a route that applies a counting automaton to a group of positions and leaves the rest free,
   its exact ground truth, copy on a subset of pairs, an anticorrelated chain, particle chunking in
   `CouplingPolicy`. Tests against enumeration.
4. Long-chain run, after the user's go-ahead.
5. Open checks: cross-entropy cap on co-revealed sets (model entropy, constrained entropy and cross-entropy
   compared); predicted against measured r from accumulated kernel loads; separator-first ordering.
6. Owed: three proofs (resampling at unequal progress under rollback; unbiasedness with an adaptive proposal;
   accuracy of the first-order effective-coupling formula); the fix to Definition 5.1 of the notes (predictions
   must not depend on the mask set); the `TorchChart` gradient; strip and grid oracles.
