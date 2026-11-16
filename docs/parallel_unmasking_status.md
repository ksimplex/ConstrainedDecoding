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
- **Rerun with the guarded script.** Cluster job `a1d8bedb`, 2026-11-15 02:07 to 03:00. Done; section 10.
- **Pre-emption.** `mit_preemptable` pre-empted a 32-CPU job after about 27 min; jobs there must be resumable.
  `sbatch --test-only` overestimates the wait on that partition by hours.
- **Laptop.** 8 cores, 16 GB. Six workers pushed compressed memory to about 7.9 GB and load to about 12. Three
  workers lower the load but not the memory. The session's Python kernel was replaced several times at gaps in
  the conversation; cause unknown. `compare.py --resume` restarts from the saved cells.

## 10. Rerun with guards (N = 32 and 64, cluster job `a1d8bedb`, 2026-11-15)

`compare2_N32.json`, `compare2_N64.json`: 16 cells each, 368 and 272 arms, independent streams, no invalid samples.
Two flags (chain 0.6 copy at N = 32, control on against off, recorded gap -4.8 jackknife s.e., -3.4 plain s.e.;
chain 0.85 counting at N = 64, policy theta = 0 at 0.904) did not reproduce with four new base seeds (8,000 runs per
arm: 0.994 (0.017) against 0.991 (0.014); 0.992 (0.028)). [verified] The Z-ratio s.e. in `compare.py` is now the
plain per-run one.

**Net account at theta = 0** (`net_account_theta0.json`). Cost per unit precision of Z_hat = steps x (relative
spread of Z_hat over runs)^2; entries are control / policy, above 1 means the policy is cheaper. [verified: read
from the JSON; single 2,000-run draws]

| constraint | N | segments | chain 0.6 | chain 0.85 | hidden chain |
|---|---|---|---|---|---|
| forbidden | 32 | 5.8 | 2.4 | 3.8 | 2.3 |
| forbidden | 64 | 4.3 | 4.8 | 3.6 | 3.3 |
| counting | 32 | 4.9 | 1.1 | 4.8 | 0.5 |
| counting | 64 | 12.9 | 2.1 | 8.9 | 0.4 |
| copy | 32 | 539.4 | 3.9 | 272.5 | 7.4 |
| copy | 64 | 9936.6 | 64.4 | 80.0 | 7.6 |
| dyck | 32 | 10.2 | 2.3 | 11.6 | 3.5 |
| dyck | 64 | 18.9 | 3.5 | 19.6 | 5.3 |

The policy is cheaper in 15 of 16 cells at both lengths (median 4.3 at N = 32,
6.4 at N = 64); the loss is the hidden chain with counting at both lengths. The largest copy
entries come from controls with heavy tails and are not reliable as sizes. Six of the eight counting cells gain.

**Order.** [verified that these are the numbers; mechanisms are hypotheses]
- Priority order ("bound": largest constraint distortion first) lowers the spread of Z_hat against the control in
  12 of 16 cells at both lengths, but raises it on counting, including on the hidden chain where no positions are
  co-revealed (0.35 to 0.47 at N = 32), so that part is an order effect alone.
- On forbidden motifs the same rule is the bottleneck for steps: chain 0.6, N = 64, 26.6 steps against 10.2 for the
  same sets in random order. It reveals next to existing reveals and creates little separation.
- The mixed-path term (N = 32, 500 runs) does not clearly reduce the counting noise and costs about 97 route passes
  per step (`mixed_vs_ref_N32.json`). Not pursued in that form.

## 11. Long chains (cluster job `24696121`, 2026-11-15 03:10 to 04:03)

`long_N128.json` ... `long_N1024.json`: 4 models (chain 0.6, chain 0.85, hidden chain, anticorrelated chain
`repel_0.8`) x 3 constraints (window: at least k of a contiguous tenth; copy on a tenth of the positions, spread or
contiguous) x 10 arms; 500 runs x 4 particles up to N = 256, 300 at 512, 200 at 1,024. 480 arms, no invalid samples.
Summary table: `long_chain_summary.json`.

**Steps at theta = 0 ("bound" priority).** [verified]

| model | constraint | N = 128 | N = 256 | N = 512 | N = 1024 |
|---|---|---|---|---|---|
| `chain_0.6` | window | 11.0 | 11.3 | 12.2 | 12.3 |
| `chain_0.6` | copy_spread | 9.7 | 10.5 | 10.4 | 10.5 |
| `chain_0.6` | copy_block | 13.5 | 18.0 | 20.8 | 21.1 |
| `chain_0.85` | window | 13.2 | 13.7 | 13.4 | 14.1 |
| `chain_0.85` | copy_spread | 11.5 | 13.2 | 14.3 | 12.2 |
| `chain_0.85` | copy_block | 13.7 | 13.0 | 17.5 | 20.7 |
| `hidden_0.85` | window | 70.5 | 87.2 | 107.5 | 93.0 |
| `hidden_0.85` | copy_spread | 77.3 | 86.3 | 110.9 | 107.4 |
| `hidden_0.85` | copy_block | 75.4 | 96.9 | 99.7 | 100.1 |
| `repel_0.8` | window | 7.9 | 8.4 | 9.5 | 9.2 |
| `repel_0.8` | copy_spread | 7.9 | 8.1 | 8.3 | 8.5 |
| `repel_0.8` | copy_block | 8.6 | 8.9 | 9.0 | 8.7 |

| N | step gain, median (min to max) | wall-time gain, median (min to max) | marginal error below control | max sd(r) |
|---|---|---|---|---|
| 128 | 10.4 (1.7 to 16.3) | 4.0 (0.9 to 8.3) | 12 of 12 | 1.4e-04 |
| 256 | 19.6 (2.6 to 31.7) | 6.0 (1.0 to 12.0) | 11 of 12 | 1.1e-04 |
| 512 | 37.0 (4.6 to 62.0) | 11.2 (2.6 to 17.8) | 11 of 12 | 1.7e-04 |
| 1024 | 77.7 (9.5 to 120.4) | 19.7 (4.7 to 30.5) | 12 of 12 | 2.4e-04 |

- Steps are flat in N, so the gain grows in proportion to N. [verified] [derived] Distance gives independence of N:
  with dependence range l, about N / l positions go together from the first step; separators then act within a
  range (about log l steps with perfect screening, about l without).
- Wall times are single runs on a shared node and include the policy's overhead.
- At N = 1,024 the one-per-step control with resampling is not a usable reference (marginal error 0.14 to 0.57 on
  the Markov chains, Z_hat off by large factors): hidden mass, as predicted. On the window cells the control
  without resampling has marginal error about 0.03 and the policy at theta = 0, run with resampling, about 0.09.
  Whether resampling is the cause for the policy is being tested (`hidden_mass_check.py`, job `36c53d08`).
- At fixed theta > 0 the spread of r grows about as sqrt(N) (chain 0.6 window, theta = 0.003: 0.29, 0.42, 0.60, 0.78).
  [verified] Marginal and adjacent-pair error do not detect this: co-revealed positions are far apart, so the
  error is in long-range joint structure. r is the instrument.
- On the hidden chain "theta = 0" means "the fitted kernel is exactly zero", which happens only where it underflows
  (about 50 positions; true mutual information about 1e-8 nats). The useful regime is a small positive tolerance.

## 12. Decisions and design changes, 2026-11-15 (later)

- **Whole-sequence error budget** B (nats) is the user-facing knob. Implemented statelessly as a per-position rate:
  a position joins a step's set only while the average load of the set stays <= B / N (`CouplingPolicy(rate=...)`).
  [derived] In KL the error of revealing a set together is the sum, in the order chosen, of each position's mutual
  information with those before it; under sub-additivity this is at most the summed kernel load.
- **Graded screening kernel**: coupling by distance and by the number of revealed positions in between
  (`fit_distance_kernel(count_edges=COUNT_EDGES)`), so that a run of revealed tokens screens more than one.
  On the hidden chain at N = 1,024 the pooled "at least one in between" curve is already 0.29 (d = 2) to 0.02
  (d = 45) of the unscreened one. [verified]
- **Constraint-blind priority scores** (`separator`, `sum`, `weighted`) and, as baselines, least-coupled-first
  (DEMASK's selection order) and a fixed dilated schedule (DUS). The priority must not depend on the constraint's
  type; design on the four core constraints, evaluate unchanged on the stress constraints and the long-chain ones.
- **Model geometry is a prior, not removed**: the kernel is pooled by distance; any user-supplied distance can
  replace it; the r audit checks it per model.
- **Metric**: distance to the constrained law, not benchmark accuracy. The goal is "an off-the-shelf diffusion model
  plus constraints": no training, one audit per model and constraint family.
- **Scope**: improving on DEMASK for unconstrained decoding is a possible separate paper; ideas that help here are
  used here, evaluated in the constrained setting only.
- Dry run of the budget sweep (N = 128, window, 150 runs; indicative only): on the hidden chain the spread of r is
  within about 15% of sqrt(2 x load) at budgets 0.01 to 1 nat; steps 15, 13, 10, 8, 7 against 46 at theta = 0;
  the dilated schedule with base 2 takes 7 steps with a spread of 1.47. On the Markov chain the separator score and
  the dilated schedule both take 7 steps with zero gap. Full sweep: job `b4e15799`.

## 13. Related work (read 2026-11-15)

`docs/parallel_unmasking_related_work.md` (five readers, full texts; DEMASK and DUS method sections also read
directly). DEMASK (arXiv:2604.02560) has our greedy threshold rule in structure (pairwise expected-shift matrix,
cumulative budget, positions before values) for unconstrained decoding, with a learned dependence head. PUNT
(arXiv:2510.21961) tests dependence online; its sets depend on sampled values. DUS (arXiv:2506.19037) is a fixed
dilated coarse-to-fine schedule under a fast-mixing assumption and names long-range constraints as its limitation.
None handles hard constraints, importance weights, or the effect of reveal order on weights. SIRS (arXiv:2610.22238)
suggests a random-weight correction for theta > 0 [hypothesis, untested].

## 14. Hidden-mass check at N = 512 and 1,024 (cluster job `36c53d08`, 2026-11-15) [verified]

`results/parallel_unmasking/hidden_mass_check.json`; 8 cells. Median marginal error (mean total variation between
the weighted particle marginals and the exact constrained marginals):

| Schedule | Resampling off | Defensive | Defensive, tau = 0.5 | Defensive, K = 16 | Off, K = 16 |
|---|---|---|---|---|---|
| One per step | 0.031 | 0.247 | 0.138 | 0.144 | n/a |
| Constraint-first (`bound`), theta = 0 | 0.031 | 0.066 | 0.081 | 0.045 | 0.016 |
| Separator, theta = 0 | 0.028 | 0.054 | 0.053 | 0.028 | 0.014 |

- Resampling off is the best setting for every schedule; the calibrated twist does not help on these cells.
- Particles that at some step held under 10% of their run's weight carry 74% to 99% of the final weight.
- Without resampling the parallel schedules match one-per-step accuracy with 6 to 100 denoiser calls against 1,024.
- **Decision:** experiments run with resampling off and more particles until a remedy is found. This is a statement
  about this twist on these (rare, long-range) constraints at K = 4 to 16, not about resampling in general.

## 15. Priority order at N = 64 (`order_N64.json`, `order_N64_least.json`, local) [verified]

theta = 0, 16 cells, 2,000 runs of 4 particles, defensive resampling. Cost gain = (64 x control spread^2) /
(steps x spread^2), where spread is sd / mean of Z_hat over runs and the control is one per step. "Lowest" is the
lowest cost gain over the 16 cells (above 1 is a gain).

| Order | Median steps | Spread / control | Median cost gain | Lowest | Cells with gain |
|---|---|---|---|---|---|
| `separator` | 7.0 | 0.55 | 14.7 | 0.90 | 15 |
| `sum` | 7.7 | 0.57 | 13.5 | 1.54 | 16 |
| `weighted` | 12.3 | 0.53 | 10.3 | 2.24 | 16 |
| `bound` (constraint-first) | 14.2 | 0.58 | 10.0 | 0.73 | 15 |
| `random` | 11.4 | 0.88 | 5.7 | 1.22 | 16 |
| `confident` (lowest entropy) | 29.7 | 0.62 | 3.4 | 0.28 | 13 |
| `least_random` (least-coupled-first, random start) | 15.6 | 0.93 | 2.2 | 0.80 | 14 |
| `least` (least-coupled-first, left-most start; DEMASK's order) | 38.0 | 1.08 | 2.1 | 0.10 | 11 |

- `separator` and `sum` are on par here. `separator`'s one loss (hidden chain, forbidden motif, 0.90) has a bootstrap
  90% interval of 0.63 to 1.19; `sum`'s spread is lower in 10 of 16 cells with a median difference of 2%. No evidence
  at this length that `sum` is better; the case for a constraint term is at long range (section 16).
- Least-coupled-first ties (exact zeros of the oracle kernel) are broken left-most, so it removes one token from the
  left edge of every gap per step and never bisects: 22 steps on `chain_0.6` and 57 on `chain_0.85`, the kernel
  ranges. A learned dependence matrix has no exact ties, so this is not a fair verdict on DEMASK; `least_tie`
  (random tie-break) was added for that (overnight run).
- One-per-step controls, resampling on against off (spread of Z_hat): on the forbidden motif resampling helps in 3 of
  4 models (3.1 against 8.5, 2.9 against 5.5, 2.1 against 2.8); on counting, copy and Dyck it hurts in all 12 cells,
  by factors of 1.1 to 7.6. [verified] Reading: a local constraint's low weight is permanent, a long-range
  constraint's is a provisional product-of-marginals guess. [hypothesis; `weight_trust.py` tests it]

## 16. Budget sweep at N = 256 and 1,024 (cluster job `b4e15799`, 2026-11-15) [verified]

`results/parallel_unmasking/budget_N256.json`, `budget_N1024.json`, `budget_sweep_summary.json`; figure
`docs/parallel_unmasking_budget_tradeoff.png`. 12 cells per length, 30 arms, 200 runs of 4 particles at 1,024 (300 at
256), defensive resampling (so the weight columns are confounded by section 14; steps and r are not).

Separator score, graded kernel, N = 1,024: steps (spread of r):

| Model | theta = 0 | B = 0.01 | 0.03 | 0.1 | 0.3 | 1.0 | Dilated base 2 | Dilated base 4 | Fixed 8/step |
|---|---|---|---|---|---|---|---|---|---|
| `chain_0.6` | 6 to 7 (0) | 5 to 6 (0.04 to 0.06) | 5 to 6 (0.06 to 0.11) | 5 to 6 (0.13 to 0.19) | 4 to 6 (0.11 to 0.40) | 4 to 5 (0.67 to 0.97) | 10 (0.00) | 5 (14.1 to 15.7) | 128 (1.7 to 2.7) |
| `chain_0.85` | 8 to 9 (0) | 7 to 8 (0.02 to 0.03) | 6 to 8 (0.01 to 0.04) | 6 to 8 (0.01 to 0.10) | 6 to 8 (0.10 to 0.19) | 6 to 7 (0.03 to 0.40) | 10 (0.00) | 5 (25.6 to 26.9) | 128 (1.9 to 3.4) |
| `hidden_0.85` | 62 to 68 (0) | 20 (0.11 to 0.12) | 17 (0.17 to 0.20) | 15 (0.33 to 0.34) | 13 (0.57 to 0.68) | 10 (1.11 to 1.28) | 10 (3.95 to 4.45) | 5 (8.5 to 9.3) | 128 (1.1 to 1.3) |
| `repel_0.8` | 5 to 6 (0) | 4 (0.06 to 0.07) | 4 (0.11 to 0.15) | 4 (0.23 to 0.29) | 3 to 4 (0.43 to 0.47) | 3 (1.11 to 1.27) | 10 (0.00) | 5 (14.7 to 15.7) | 128 (1.7 to 1.9) |

- N = 256, hidden chain: 53 to 55 steps at theta = 0; 16 to 17, 14, 12, 9 to 10, 7 to 8 under the five budgets; dilated
  base 2 takes 8 steps with a spread of r of 2.1.
- Predicted against measured: sd(r) / sqrt(2 x load) has median 0.84 at 1,024 (10% to 90%: 0.43 to 1.03, max 1.22)
  and 0.91 at 256 (0.65 to 1.08, max 1.47). The load never exceeds the budget (max 0.98 and 0.94 of it; median 0.57
  and 0.45).
- Graded against two-bin kernel: no difference on Markov and anticorrelated chains; on the hidden chain 20 against
  27 steps at B = 0.01, 15 against 19 at 0.1, 10 against 12 at 1.0 (N = 1,024), at equal spread of r.
- Order under a budget (median steps over three constraints, N = 1,024, B = 0.03): `chain_0.6` separator 5.0, bound
  8.9, random 9.0, least_random 11.0; hidden chain 17.0, 20.3, 20.5, 26.5. Least-coupled-first takes the most steps
  on `chain_0.85` and the hidden chain at every budget and both lengths, and on `chain_0.6` except at B = 1.0, N = 256;
  on the anticorrelated chain the four orders are within 4 to 8 steps and random takes as many or more.
- Weights (confounded by resampling): at 1,024 the separator order has worse spread of Z_hat and marginal error than
  constraint-first (`chain_0.6` 3.71 / 0.137 against 1.06 / 0.043). On `copy_spread` the gap is structural:
  constraint-first reveals both partners first and the weights become constant (spread 0.00 against 10 to 14).
- Where the gain over the dilated schedule comes from (hidden chain, N = 1,024, window, 150 paths, local): the
  dilated schedule reveals 2, 2, 4, 8, 16, 32, 64, 128, 256, 512 tokens; its first six steps have zero gap and its
  last two contribute spreads of 2.3 and 3.7. The budgeted schedule (1 nat) reveals 63 to 171 per step with a spread
  of 0.3 to 0.5 at every step. [verified]
- Reveal pattern (one path, N = 256, B = 0.03) [verified]: on `chain_0.85` 9 anchors about 26 apart, then every gap
  is bisected each step (9, 10, 20, 40, 80, 97 tokens; 6 steps). On the hidden chain every reveal in the first five
  steps is an isolated token; the count per step stays at 11 to 26; single tokens bisect down to a comb by step 7 and
  the holes fill in over six more steps (14 steps). Thick runs appear only as a by-product at the end.

## 17. Resampling frequency (local, 2026-11-15) [verified]

Defensive resampling, trigger at half the particles. One per step: 2.8 to 8.5 resampling events per run at N = 64
(4% to 13% of steps) and 5.6 to 12.8 at N = 512 (1% to 3%). Separator schedule at theta = 0: 1.2 to 4.4 per run.

## 18. Decisions, 2026-11-15 (morning)

- `sum` is not established as the default score; `separator`, `sum`, `weighted`, `lex` (constraint distortion first,
  separator score as tie-break) and `bound` are carried through the constraint-interaction study.
- Design-set long-range constraint: two windows sharing one count (`two_window`), plus Dyck at long lengths. Parity
  stays held out as a false-alarm check.
- New code: `lex`, `least_tie` priorities; `ThickSeparatorSchedule` (hand-built thick separators, to test the
  "thick separators, then recurse" picture against the greedy rule); `resample_power` (partial resampling, unbiased);
  arm sets `interact`, `resample`, `thick`; `weight_trust.py`.
- Remote runs wait for approval; `experiments/parallel_unmasking/jobs/interact_N1024.sbatch` is drafted, not submitted.

## 19. Thick separators against the greedy rule (local, 2026-11-15) [verified]

`results/parallel_unmasking/thick_N256.json`, `thick_N256_small.json`; figure
`docs/parallel_unmasking_thick_vs_greedy.png`. `ThickSeparatorSchedule(width, spacing)`: separators `width` tokens
thick every `spacing` positions, built one token per step, then the same inside every stretch. N = 256, window
constraint, 500 runs of 4 particles, no resampling. Greedy = separator score, graded kernel, whole-sequence budget.

Hidden chain, steps (spread of r):

| Greedy rule | Best thick schedule at the same steps |
|---|---|
| 7 (1.15), 9 (0.62) | width 2, spacing 16: 8 (0.85) |
| 11.7 (0.34) | width 3, spacing 16: 10 (0.48); width 4, spacing 16: 12 (0.27) |
| 14 (0.185) | width 4, spacing 24: 14 (0.196) |
| 16 (0.102) | width 4, spacing 32: 16 (0.142) |
| 21 (0.031), 26 (0.009) | width 6: 18 (0.057), 20 (0.054), 24 (0.030) |

- At its best spacing the hand-built schedule is on the greedy rule's curve: ratio of spreads at equal steps 1.01,
  0.97, 0.84, 1.06, 1.39 at 8, 10, 12, 14, 16 steps. Other spacings are 1.5 to 4 times worse. Neither dominates.
- The gap is set by the separator width, hardly by the spacing: about 2.1 (width 1), 0.85 (2), 0.33 to 0.48 (3),
  0.14 to 0.27 (4), 0.03 to 0.06 (6). Width 1 is recursive bisection and is as poor as the dilated schedule (2.1).
- Markov chains: width 1 with spacing at the kernel range is exact or nearly so in one step fewer than the greedy
  rule at theta = 0 (`chain_0.85`: 7 steps, spread 0.000 to 0.001, against 8; `chain_0.6`: 5 steps, 0.002, against 6).
  The greedy rule under budgets of 1e-4 to 1e-3 reaches the same step counts with spreads of 0.002 to 0.014.
- The rough count made beforehand ("thick-first costs more steps") was wrong. [verified]

## 20. Do intermediate weights predict final weights? (`weight_trust.py`, local, 2026-11-15) [verified]

`results/parallel_unmasking/weight_trust_N64.json`, `weight_trust_summary.json`; figure
`docs/parallel_unmasking_resampling_variance_ratio.png`. N = 64, 16 cells, 4,000 single particles per cell, no
resampling. rho = E[w_T^2 / w_t] E[w_t] / E[w_T^2]: the second moment of the final weights if ancestors were drawn
by the weights at step t (one event, many-particle limit), relative to not resampling. One per step, median over four
models, at 10 / 25 / 50 / 75 / 90% of the steps:

| Constraint | rho | Regression slope of final on current log-weight |
|---|---|---|
| Forbidden motif | 1.79, 0.21, 0.046, 0.027, 0.024 | 0.98, 0.94, 0.96, 0.98, 1.00 |
| Copy | 1,420, 10,300, 1.95, 0.145, 0.142 | 0.62, 0.67, 0.86, 0.98, 1.00 |
| Dyck | 11.1, 7.9, 0.86, 0.30, 0.24 | 0.41, 0.51, 0.76, 0.94, 0.99 |
| Counting | 5.6, 103, 12.5, 0.88, 0.62 | 0.07, 0.21, 0.51, 0.86, 0.98 |

- The local constraint's weights are permanent (slope near 1) and resampling pays from a quarter of the way in. For
  counting and Dyck early weight differences mostly do not survive, and resampling adds variance until half to three
  quarters of the way. Copy looks predictable in the bulk (slope 0.6 to 0.9) but a few particles with low early
  weight carry the final mass, so early resampling is the most harmful of all: the rank correlation misses it, rho
  does not. rho is heavy-tailed; values above 100 are orders of magnitude only.
- A partial exponent of 0.25 to 0.5 brings rho below 1 at 25% for copy (0.14 to 0.34 in three of four models) and
  Dyck (0.58 to 0.76), and does nothing for counting (about 1).
- This is the audit behind **gated resampling** (`resampling_gate` in `compare.py`, `resample_power` as a function of
  progress in `twisted_smc`): a pilot without resampling gives, per step, the exponent that minimises rho, and 0 where
  no exponent brings it under 0.8. Unbiased by construction (tests in `tests/test_resampling.py`).

## 21. Constraint-interaction study at N = 256 (local, 2026-11-15) [verified]

`results/parallel_unmasking/interact_N256.json`, `interact_N256_summary.json`. 4 models x (window, two_window,
copy_spread, copy_block), 100 runs of 16 particles, **no resampling**, graded kernel. Control = one per step with the
same particles. Medians over the 16 cells: steps | spread of Z_hat / control | marginal error / control.

| Order | theta = 0 | B = 0.03 | B = 0.3 |
|---|---|---|---|
| `separator` | 7.5, 0.76, 0.86 | 5.5, 0.79, 0.88 | 5.0, 1.02, 1.04 |
| `sum` | 7.6, 0.57, 0.77 | 6.0, 0.63, 0.79 | 5.6, 0.62, 0.83 |
| `weighted` | 13.9, 0.50, 0.70 | 9.1, 0.61, 0.78 | 8.0, 0.60, 0.78 |
| `lex` | 9.7, 0.55, 0.71 | 7.2, 0.67, 0.83 | 6.4, 0.65, 0.79 |
| `bound` | 15.1, 0.50, 0.77 | 9.9, 0.57, 0.75 | 8.0, 0.56, 0.74 |
| `random` | 13.2, 0.83, 0.91 | 9.7, 0.88, 0.93 | 8.3, 0.95, 0.97 |
| `least_tie` | 13.1, 1.19, 1.16 | 10.0, 1.12, 1.06 | 8.5, 1.15, 1.10 |

- `sum` takes about the steps of `separator` (ratio 1.14, 1.01, 1.12) with a lower spread in 12 of 16 cells at every
  budget; cost (steps x spread^2) is 1.4 to 1.7 times lower. This is the evidence for `sum` that N = 64 did not give.
- The difference is the long-range copy: spread ratio on `copy_spread` 0.10 to 0.14 (`sum`) against 0.69 to 1.36
  (`separator`); `bound`, `lex`, `weighted` reach 0.01 to 0.08 there but pay 1.2 to 2 times the steps overall.
- The schedule's gap does not depend on the constraint: spread of r under `sum` is 0.098 to 0.115 across the four
  constraints at B = 0.03 and 0.41 to 0.46 at B = 0.3. The weights do, as predicted.
- Cost gain over one per step (256 x control spread^2 / steps x spread^2), median (lowest cell): `sum` 91 (6.2),
  93 (29), 117 (23) at the three budgets; `separator` 40 (4.3), 65 (18), 41 (19); `least_tie` 12 (1.9), 19 (7.8), 20 (3.9).
- Unbiasedness: at theta = 0 the 108 arms have (Z_hat / p(C) - 1) / s.e. with mean 0.05, s.d. 0.99, largest 2.4.
  Under a budget Z_hat estimates the parallel law's p~(C): median ratio 0.986 at B = 0.03 and 0.952 at B = 0.3.

## 22. Fair least-coupled-first and `lex` at N = 64 (`order_N64_extra.json`) [verified]

Same protocol as section 15. `least_tie` (least-coupled-first with random tie-breaks): median 10.7 steps, spread /
control 0.94, cost gain 4.7, lowest 1.08, 16 of 16 cells with a gain: the level of `random` (11.4, 0.88, 5.7), above
the left-most tie-break (2.1 to 2.2) and `confident` (3.4), below `separator` (14.7). `lex`: 13.7 steps, 0.56, 10.9,
lowest 0.12: on the core constraints the constraint term is non-zero everywhere, so it behaves like `bound`.

## 23. Resampling against K, and gated resampling (N = 64, local, 2026-11-15) [verified]

`results/parallel_unmasking/resample_N64.json` (forbidden, copy; 6,400 particles per arm), `resample_N64_global.json`
(counting, Dyck; 3,200 per arm), `resample_N64_summary.json`. Models `chain_0.85`, `hidden_0.85`; K = 4, 16, 64 (and
256 for the first file) at a fixed total number of particles. Measure: relative standard error of the pooled Z_hat,
divided by its value without resampling (below 1 = resampling helps). Median [range] over K <= 64 and both models:

| Schedule | Constraint | Defensive | Plain | Systematic | Exponent 0.5 | Gated |
|---|---|---|---|---|---|---|
| One per step | Forbidden motif | 0.8 [0.5-1.2] | 1.2 [0.6-2.0] | 0.9 [0.6-2.2] | 0.8 [0.6-1.3] | 0.9 [0.5-1.1] |
| One per step | Copy | 4.5 [1.2-14.3] | 6.0 [1.7-10.9] | 3.7 [1.6-14.4] | 4.7 [1.3-12.8] | 1.6 [1.4-4.2] |
| One per step | Dyck | 1.8 [1.3-3.1] | 2.3 [1.4-3.3] | 3.2 [1.5-4.0] | 1.5 [1.0-2.2] | 1.2 [0.9-1.5] |
| One per step | Counting | 1.4 [1.0-5.7] | 2.2 [1.0-4.2] | 3.6 [1.2-5.6] | 1.6 [1.1-4.9] | 1.1 [0.9-1.4] |
| Separator, theta = 0 | Forbidden motif | 0.9 [0.6-1.6] | 1.0 [0.7-1.8] | 1.2 [0.6-2.1] | 0.8 [0.6-1.2] | 0.8 [0.5-1.0] |
| Separator, theta = 0 | Copy | 1.2 [1.0-1.5] | 1.8 [1.1-3.6] | 1.7 [1.2-3.2] | 1.1 [1.0-1.6] | 1.0 [0.9-1.1] |
| Separator, theta = 0 | Dyck | 1.0 [0.7-1.4] | 1.2 [1.0-4.4] | 1.3 [1.0-1.6] | 1.0 [0.8-1.3] | 0.9 [0.7-1.1] |
| Separator, theta = 0 | Counting | 1.0 [0.9-1.4] | 2.0 [0.9-7.4] | 1.6 [0.9-4.5] | 1.1 [1.0-1.7] | 1.0 [0.8-1.1] |

- **The harm is not a small-K effect.** On `chain_0.85` copy, one per step, defensive resampling is 12, 7, 14 and 5
  times worse than none at K = 4, 16, 64, 256, and plain resampling 8, 8, 11 and 31 times.
- **The parallel schedule removes most of it.** With a handful of steps there are 1 to 4 resampling events per run
  and the twist is accurate early; defensive resampling is then within 1.6 times of none in every cell.
- **Gated resampling is safe under the parallel schedule**: over 24 cell-by-K combinations it is never more than 1.1
  times worse than none, and better by over 1.25 times in 5 (median 0.96). Under one per step it is within 1.5 times
  in 20 of 24 (median 1.10, worst 4.2 on `hidden_0.85` copy at K = 16), against 11 of 24 for defensive.
- Where resampling helps (the local constraint) the gain is modest, a factor of 1.3 to 2.5 in standard error.
- Systematic resampling is no better than multinomial and often worse on the long-range constraints. A fixed
  exponent of 0.5 without the gate behaves like defensive resampling.
- The marginal error tells the same story with more noise: median ratio to no resampling 1.12 (gated, separator),
  1.28 (defensive, separator), 1.20 and 1.35 under one per step.
- Resampling events per run: one per step 2.8 to 12.8; separator schedule 1.2 to 4.4 (section 17).
- Reading: the gate's criterion is a one-event, many-particle statement. It is optimistic on `hidden_0.85` copy under
  one per step, where repeated events compound. A stricter threshold or a criterion for the remaining events is the
  obvious refinement. [hypothesis]

## 24. Thick separators at N = 1,024 (local, 2026-11-15) [verified]

`results/parallel_unmasking/thick_N1024.json`; hidden chain, window, 100 runs of 4 particles, no resampling. Greedy
rule: 10 steps (1.16), 13 (0.63), 15 (0.38), 17 (0.20), 20 (0.117), 25 (0.034), 29 (0.010), 61 at theta = 0. Best
thick schedule at each step count against the greedy curve at the same steps: 10: 0.98 against 1.16; 12: 0.54 against
0.77; 14: 0.41 against 0.49; 16: 0.31 against 0.27; 18: 0.13 against 0.17; 20: 0.108 against 0.114; 24: 0.057 against
0.043. Within about 30% either way, as at N = 256; the hand-built schedule is ahead at 10 to 14 steps. Its step count
does not grow with N but its gap does at fixed width and spacing (width 4, spacing 16: 0.27 at 256, 0.54 at 1,024),
and its two parameters were chosen after the fact. That it can be ahead says the greedy rule is not optimal in the
middle of the range. [verified] A separator score that values thickness is a lead. [hypothesis]

## 25. Constraint interaction at N = 1,024 (local, 2026-11-15) [verified]

`results/parallel_unmasking/interact_N1024_local.json`, `interact_N1024_local_summary.json`. `chain_0.85` and
`hidden_0.85` x (`copy_spread`, `two_window`), 50 runs of 16 particles, no resampling, no one-per-step control
(too slow locally). The full 12-cell version is drafted for the cluster (`jobs/interact_N1024.sbatch`, not submitted).

Spread-out copy (51 pairs), Z_hat / p(C) with the marginal error in brackets; * = tail guard flags the standard error:

| Model, budget | `separator` | `sum` | `weighted` | `lex` | `bound` | `random` | `least_tie` |
|---|---|---|---|---|---|---|---|
| `chain_0.85`, theta = 0 | 1.5e-10 (0.38)* | 1.19 (0.12)* | 1.07 (0.04) | 0.95 (0.02) | 1.03 (0.03) | 1.8e-3 (0.60)* | 1.4e-13 (0.60)* |
| `chain_0.85`, B = 0.03 | 2.0e-12 (0.56)* | 1.00 (0.14)* | 0.88 (0.09)* | 1.19 (0.06) | 0.94 (0.03) | 3.8e-5 (0.66)* | 6.7e-13 (0.43)* |
| `chain_0.85`, B = 0.3 | 1.1e-9 (0.58)* | 1.12 (0.17)* | 0.72 (0.22)* | 0.93 (0.09)* | 0.85 (0.10) | 2.7e-5 (0.57)* | 5.1e-14 (0.58)* |
| `hidden_0.85`, theta = 0 | 1.23 (0.16)* | 0.96 (0.03) | 1.11 (0.09)* | 0.96 (0.03) | 0.99 (0.03) | 0.97 (0.17)* | 0.51 (0.32)* |
| `hidden_0.85`, B = 0.03 | 0.64 (0.16)* | 1.02 (0.04) | 0.96 (0.03) | 0.93 (0.03) | 0.94 (0.04) | 2.28 (0.42)* | 0.23 (0.23)* |
| `hidden_0.85`, B = 0.3 | 0.82 (0.28)* | 0.87 (0.05) | 0.88 (0.03) | 0.89 (0.04) | 0.90 (0.02) | 0.60 (0.18)* | 0.23 (0.22)* |

- **On a long-range copy over a strongly correlated model, an order that ignores the constraint fails outright.**
  With `separator`, `random` or least-coupled-first the 800 particles find 1e-13 to 1e-3 of the constrained mass on
  `chain_0.85`: a copied token revealed after its neighbours must fit them, and almost never does. Orders with the
  constraint term (`sum`, `weighted`, `lex`, `bound`) reveal the partners before their neighbourhoods and recover
  p(C) to within the noise. The same cells at N = 256 show the onset (`separator` 0.67, others 0.90 to 1.20).
- Steps are not the price: on `copy_spread`, median over the six arms, `separator` 11.0 steps, `sum` 11.0, `lex`
  11.5, `bound` 13.7, `weighted` 14.0.
- `sum` works but with heavier tails than `lex` or `bound` on `chain_0.85` (marginal error 0.12 to 0.17 against 0.02
  to 0.10): at this length the constraint term deserves more weight than `sum` gives it.
- `two_window` does not separate the orders (spread of Z_hat 0.27 to 0.30 for the five scores; steps 10.5 for
  `separator` and `sum`, about 16 for `bound` and `weighted`).
- Unbiasedness at theta = 0 on the arms with a usable standard error (25 arms): t-statistics mean -0.06, s.d. 1.0,
  largest 2.1. Peak memory footprint per worker 0.2 to 0.7 GB.

## 26. Dyck at N = 256 (local, 2026-11-15) [verified, small sample]

`results/parallel_unmasking/interact_dyck_N256_seed0.json`: four models, 25 runs of 16 particles (400 particles; a
second seed was skipped for time), no resampling. Median steps / spread of Z_hat relative to one per step:
`separator` 9 / 0.95, `sum` 11 / 1.31, `random` 16 / 1.37, `least_tie` 15 / 1.76, `bound` 26 / 1.44, `lex` 26 / 1.54,
`weighted` 24 / 1.54 at theta = 0; under B = 0.03 all take 6 to 11 steps with ratios 1.06 to 1.51. Here the
constraint has a stake everywhere, and constraint-first orders cost three times the steps without better weights.
With 25 runs the spread ratios are indicative only.

**Where this leaves the score.** No single score is best everywhere: `separator` fails on long-range copy; `lex` and
`bound` are best there and cost 2 to 3 times the steps on Dyck and on the core constraints at N = 64; `sum` is never
the worst and is the working default. The open question is the relative weight of the two terms. [hypothesis: a
scale-free combination, for example ranks or each term divided by its maximum over masked positions]

## 27. Order or parallelism? The sequential smart-order control (local, 2026-11-15) [verified]

`results/parallel_unmasking/seqorder_N256_hidden085_summary.json` (`--arms seqorder`; `CouplingPolicy(max_set=1)` is one
token per step in priority order). `hidden_0.85`, N = 256, 100 runs of 16 particles, no resampling. The other hidden
chains (`hidden_0.85_n0.15`, `_n0.5`) were interrupted by a session restart and are partial.
Spread of Z_hat (denoiser calls): sequential | parallel at the exact threshold | parallel at B = 0.03.

| Constraint | Random (native) | `sum` | `lex` | `bound` | `separator` |
|---|---|---|---|---|---|
| `copy_spread` | 0.79 (256) | 0.04 (256), 0.12 (56), 0.18 (14) | 0.03, 0.19, 0.12 | 0.03, 0.08, 0.13 | 0.32, 0.56, 0.57 |
| `copy_block` | 0.45 | 0.19, 0.39, 0.27 | 0.09, 0.40, 0.28 | 0.10, 0.18, 0.14 | 0.29, 0.29, 0.29 |
| `two_window` | 0.33 | 0.14, 0.15, 0.15 | 0.15, 0.13, 0.15 | 0.15, 0.14, 0.18 | 0.29, 0.35, 0.27 |
| `window` | 0.21 | 0.13, 0.14, 0.17 | 0.12, 0.12, 0.15 | 0.12, 0.12, 0.15 | 0.21, 0.16, 0.21 |

- **A constraint-aware order is a fix to the base sequential sampler**: 1.7 to 28 times lower spread than random order
  at the same 256 calls for `sum`, `lex` and `bound` (unrounded ratios 1.66 to 27.6; `copy_spread` 21.8, 27.6, 24.7).
  The separator order alone gives 1.0 to 2.5. (Corrected 2026-11-16: an earlier version said "1.6 to 20", computed from
  the rounded table entries.)
- The file also holds a `random sequential` arm (random priority through the policy, one position per step): spread
  0.35, 0.21, 0.36, 0.20 on `copy_spread`, `copy_block`, `two_window`, `window`. It is not the plain random baseline:
  the policy's `forced_first` default reveals a forced copy partner next, which is itself a constraint-aware step. The
  baseline in the table is the native one-per-step sampler. [inferred from the code default, not isolated by a run]
- **Parallelism costs weights only on the copies**, and that cost is a selection flaw (section 31), not inherent.
- Smart parallel at 14 calls beats random sequential at 256 calls in all four cells. In calls x spread^2 the 14-call
  `sum` schedule is 9 to 16 times cheaper than `sum` sequential on three constraints and level on `copy_spread`.

## 28. First look at a real model: MDLM on OpenWebText (cluster job `ea73325e`, one L40S, 2026-11-15) [verified]

`results/parallel_unmasking/mdlm_N128.json`, `mdlm_N256.json`; script `experiments/parallel_unmasking/mdlm_first_look.py`.
130M parameters, GPT-2 tokens, 256 audited sequences at N = 128 and 128 at N = 256. The gap is defined per step as
log(sequential joint of the step's tokens in index order) - log(product of their conditionals); its mean is minus a KL
and is well defined although the model's conditionals are not order-consistent.

**Kernel (token-level coupling, nats).** Unscreened, by distance 1, 2, 4, 8, 16, 32, 64: N = 128: 1.41, 0.88, 0.55,
0.47, 0.54, 0.71, 0.86; N = 256: 1.55, 0.81, 0.32, 0.20, 0.19, 0.22, 0.41. With 3 to 4 revealed tokens in between: 0.05 to
0.15; with 17 or more: 0.005 to 0.03. **Dependence does not decay with distance; it decays with how much is revealed.**

**Unconstrained audit, N = 256** (calls, KL in nats, 99th percentile of the per-sequence gap):

| Schedule | Calls | KL | q99 |
|---|---|---|---|
| Separator, B = 1 | 142 | 1.11 +- 0.16 | 7.2 |
| Separator, B = 3 | 93 | 2.87 +- 0.27 | 12.9 |
| Random order, B = 3 | 76 | 2.12 +- 0.25 | 9.5 |
| Least-coupled-first (`least_tie`), B = 3 | 74 | 2.60 +- 0.26 | 10.9 |
| Separator, B = 10 | 78 | 9.35 +- 0.49 | 23.9 |
| Fixed 2 / 4 / 8 / 16 per step | 128 / 64 / 32 / 16 | 6.1 / 15.7 / 36.4 / 65.9 | 24.8 / 39.5 / 76.6 / 113 |
| Dilated, base 2 | 8 | 123 +- 3 | 208 |

- Budgets up to 0.3 nats allow no parallelism. About 3.4 times fewer calls cost 2 to 3 nats. At about 75 calls the gap
  is a third of fixed-size at 128 calls and a sixth of fixed-size at 64. N = 128: `least_tie` B = 3: 58 calls, KL 1.75;
  separator B = 3: 65 calls, 3.18; fixed 2 per step: 64 calls, 6.77. The separator order is not the best here.
- Measured KL / predicted load: 0.96 to 1.43 at N = 128 and 0.80 to 1.75 at N = 256 over the seven budgeted arms. The
  job printed a load three times too high (the schedule object was shared by three batches); the files carry the
  corrected value and the original.
- No budgeted arm below 56 calls was run: no comparison at 16 or 32 calls yet.

**Order inconsistency (N = 128, 96 sequences; log p about -454 nats).** A second random order differs from the first
by -0.4 +- 2.5 on average with a per-sequence s.d. of 24 nats; the separator order by -6.4 +- 2.7; left to right by
-1.1 +- 2.8. A schedule gap of 1 to 3 nats is small beside this; "exact" on this model is defined up to it.

**Constrained (N = 128, 48 runs of 8 particles, three token classes, no resampling).** Spread of Z_hat:

| Constraint | Random sequential | `sum` sequential | `lex` sequential | `lex`, B = 3 (55 to 58 calls) | `sum`, B = 3 (65 calls) |
|---|---|---|---|---|---|
| Window count | 0.96 | 0.89 | 0.54 | 0.51 | 1.01 |
| Two-window count | 0.90 | 1.07 | 0.64 | 0.67 | 1.07 |
| Class copy (6 pairs) | 1.05 | 0.90 | 0.55 | 0.55 | 0.67 |

- Constraint-first halves the spread and its parallel version keeps that at under half the calls.
- `sum` does nothing here: the separator term is in whole nats and swamps the constraint term (the scale problem).
- log Z_hat depends on the order by 0.3 to 1 nat (copy: -1.39 `lex`, -1.94 random, -2.35 `sum`).

## 29. A non-mean-field twist (local, 2026-11-15) [verified at N = 64; not yet in the main sampler]

`twist_error.py`, `twist_smc.py`; `results/parallel_unmasking/twist_error_N64.json`, `twist_smc_N64.json`.
The twist is the constraint's probability under a stand-in for the model. Product = independent positions (in use).
Chain = a first-order chain. `chain_ratio` builds it with no extra denoiser call from the current marginals and a
stationary adjacent-class kernel K: Q_i(a, b) proportional to mu_{i+1}(b) K(a, b) / sum_a' mu_i(a') K(a', b).

Spread over states of log psi_standin - log psi_true (nats), 25% revealed, 300 states from the constrained law, N = 64:

| Constraint (`hidden_0.85`) | Product | Chain, model's own pair conditionals | `chain_ratio` |
|---|---|---|---|
| Counting | 0.81 | 0.33 | 0.16 |
| Window | 1.09 | 0.39 | 0.24 |
| Two windows | 0.51 | 0.12 | 0.14 |
| Dyck | 1.13 | 0.52 | 0.34 |
| Forbidden motif | 0.98 | 0.76 | 0.65 |

On `chain_0.85` the exact-pair chain has error 0.000 (code check) and `chain_ratio` 0.31 to 0.39 against 0.95 to 2.03.

Sampler test (random-order sequential, three hidden chains x five constraints, 2,000 particles, positivity mix 5%):
- Resampling harm (spread with plain resampling / without): product median 1.57 (1.03 to 3.29; above 1.25 in 10 of 15
  cells); `chain_ratio` median 1.01 (0.80 to 1.19; none above 1.25).
- Spread without resampling relative to the product twist: `chain_ratio` median 0.89 (0.33 to 1.26); exact-pair chain
  0.54 (0.15 to 0.92).
- All arms unbiased against the exact p(C) (largest |Z - 1| / s.e. 3.7 over 90 estimates).

Plan: a route-generic lift. The chain is a set of pairwise factors between neighbours; lifting = multiplying the
constraint by them and summing exactly. Automata: product automaton (about x A). Grammars: intersection with the pair
automaton (up to x A^3). Factor graphs: when the combined graph stays narrow (aligned copy is a ladder); otherwise a
partial lift. Determinant-based routes (arborescences): not covered; the product twist remains the fallback.
[derived; needs a proof for the weighted, position-dependent case]. Order k costs A^k (automata), A^(3k) (charts).
Related: Discrete Copula Diffusion (Liu, Broadrick, Niepert & Van den Broeck, arXiv:2410.01949) keeps a diffusion
model's marginals and takes the dependence from a second model by I-projection; `chain_ratio` is one scaling pass of
that projection, used inside the twist so the target is unchanged. Iterating the projection is a lead. [hypothesis]

## 30. Gated resampling at N = 256 (local, 2026-11-15) [verified]

`results/parallel_unmasking/resample_N256.json`: `chain_0.85`, `hidden_0.85` x (window, copy_spread), K = 4, 16, 64,
3,200 particles. Relative s.e. of the pooled Z_hat over its value without resampling, 12 cases per schedule:

| Schedule | Defensive | Plain | Systematic | Exponent 0.5 | Gated |
|---|---|---|---|---|---|
| One per step: median (range) | 3.70 (0.23-49) | 2.27 (0.26-30) | 3.42 (0.16-44) | 4.93 (0.24-38) | 0.78 (0.24-1.45) |
| Separator: median (range) | 1.14 (0.24-2.7) | 1.56 (0.14-9.8) | 1.49 (0.13-30) | 1.36 (0.17-7.7) | 0.94 (0.19-1.14) |

Ungated resampling is 4 to 49 times worse on the window constraint and 2 to 5 times better on `copy_spread`, where the
weights are poor without it. The gate keeps the gain and removes the harm: never more than 1.5 times worse in 24
cases, better by over 1.25 times in 10.

## 31. Why parallel steps cost weights on the block copy: half-open pairs (local, 2026-11-15) [verified]

`hidden_0.85`, `copy_block`, N = 256. In every parallel step the two block tokens revealed together are from different
pairs (0.00 complete pairs, 2.00 half-open per step): the rule takes the best token of one block, then the best of
the other, and their forced partners stay masked beside fresh reveals. Variance of the log-weight after the first
step: 0.80 with two half-open pairs; 0.215 with one complete pair (`pull_forced`), identical to the sequential value
after the same two reveals (0.215), and likewise after the second pair (0.438 both). The gap in section 27 between
exact-threshold and budgeted parallel on this cell is real (bootstrap 90% intervals 0.35-0.44 and 0.24-0.30).
- At the exact threshold the pull-in did not engage: the kernel entry between partners 128 apart is 2e-10 nats, pilot
  noise that fails an exact zero test. `PULL_TOL = 1e-6` nats now applies to forced partners.
- Under a budget the pull-in engaged but 62% to 66% of the near-block positions were revealed before the block was
  complete (0% at the exact threshold). `guard=True` makes a position without a constraint stake count its coupling to
  still-masked positions with one as load. Test of both: `--arms fix` (`fix_N256.json`).
- **Result (2026-11-16) [verified, one model, four constraints, 100 runs x 16].** Spread of Z-hat, plain | +pull |
  +pull+guard (calls), against the sequential value:
  - `copy_block`, `lex`: sequential 0.09; theta = 0: 0.40 (57) | 0.12 (56) | 0.11 (56); B = 0.03: 0.28 (15) | 0.12 (20) | 0.10 (24).
  - `copy_block`, `sum`: sequential 0.19; theta = 0: 0.39 (54) | 0.23 (54) | 0.21 (60); B = 0.03: 0.27 (14) | 0.23 (14) | 0.20 (19).
  - `copy_spread`, `lex`: sequential 0.03; theta = 0: 0.19 (57) | 0.06 (56) | 0.04 (57); B = 0.03: 0.12 (15) | 0.07 (15) | 0.06 (15).
  - `copy_spread`, `sum`: sequential 0.04; theta = 0: 0.12 (56) | 0.08 (55) | 0.05 (56); B = 0.03: 0.18 (14) | 0.08 (14) | 0.07 (14).
  - `window`, `two_window`: 0.11 to 0.19 in every variant, sequential 0.12 to 0.15 (no change beyond noise); the guard
    adds about five calls under the budget (14-15 to 19-20).
  - Z-hat / p(C) 0.97 to 1.09 plain, 0.97 to 1.02 with the fixes; marginal TV on `copy_block` 2.2-3.1% plain, 1.4-1.9% fixed.
  - Reading: the parallel cost on the copies was the selection flaw. The pull-in carries most of the gain; the guard a
    little more at a cost in calls. Summary: `results/parallel_unmasking/fix_N256_summary.json`.
- `bound` breaks ties left to right and so pairs partners by accident; that is why it beat `lex` in parallel there.

## 32. Exposure: one criterion behind the order rules (2026-11-15) [derived, not yet checked numerically]

Exact: with the product twist and its locally optimal proposal the weight of a sequential step is
w = psi(new state) / psi(old state with the new token clamped): the revision of the twist caused by the other masked
positions' marginals moving. First order: log w ~ sum_j <mu^C_j / mu_j, change in mu_j>, and by Cauchy-Schwarz with the
local chi-square / KL relation sd(log w) <~ 2 sum_j sqrt(D_j M(i, j)), with D_j = KL(mu^C_j || mu_j) (the stake) and
M(i, j) the kernel. Exposure = stake x coupling over the positions that stay masked: one kernel application per step.
Constraint-first and the guard follow from keeping it low; the partner pull-in does not (it needs how the stake moves
after a reveal). To check: does cumulative exposure predict the log-weight variance across schedules, as load
predicts the gap? Each piece of the rule stays only if its ablation moves error at equal calls or calls at equal error.

## 33. Roadmap and decisions (2026-11-15, evening)

Paper framing (user): building on Dang & Ermon (constrained decoding with automata) and TWISTER (SMC for the exact
constrained law): (a) constraint families beyond automata; (b) a twist that is not mean-field; (c) a constraint-aware
unmasking schedule, sequential then parallel. Aim: constrained decoding that is useful and fast; gains in capacity,
accuracy and efficiency. Verification wanted: (i) the twist at length, on all families, across scales of dependence
and on a real diffusion language model; (ii) the sequential rule, then the parallel rule, each across models and
constraints with ablations and the accuracy / efficiency trade-off derived; (iii) a practical case (structured JSON on
a diffusion language model is the standard benchmark) with overall numbers. Proofs for major statements.
- Evaluate on hidden chains (several noise levels); Markov chains are a sanity check. Baseline = random sequential.
- Compare schedules with resampling off; resampling is studied separately.
- Prefer one justified, ablated criterion to a list of rules. End measures: error against the exact constrained law
  at equal denoiser calls, or calls to reach a given error; weight variance is the working proxy only.
- Twist changes must not regress: exact regression with the new options off, unbiasedness, a frozen benchmark,
  opt-in until it passes.
- Kernel: no per-layout or per-region patches; pool by maximum; report the audit distribution (tail frequencies,
  both directions); no worst-case bound. Arborescence routes are out of scope for the lift.
- A real language model's dependence is long-range early on; plan for a modest parallel gain there.
- Remote runs need approval; the 1,024-position job waits for the selection fix.

## 34. Next steps (2026-11-15, evening)

1. Selection: read `fix_N256.json`; check exposure against measured log-weight variance; settle a scale-free score;
   held-out constraints; ablations.
2. Refactor before the wrapper: split `compare.py` into a small library and per-study scripts; separate scoring from
   admission in `CouplingPolicy`; exact-regression and unbiasedness tests for every option; a README for the folder.
3. Chain twist in the main sampler: pair-lift wrapper for automata, positivity mix, no-regression benchmark at 64, 256
   and 1,024 with and without the gate; then grammars with a measured cost; then the partial lift for copy.
4. Kernel keyed on how much is revealed; pooling by maximum; the stress oracle with a persistent global latent.
5. Real model: budgets down to 16 calls; the chain twist; then an instruction-following model for structured JSON.
6. Cluster (after 1): interaction study and sequential control at 1,024 on the hidden chains
   (`jobs/interact_N1024.sbatch`, drafted). Finish the two interrupted hidden chains of section 27.
7. Proofs: unbiasedness with a state-dependent order; the closure proposition for the lift; the exposure bound; the
   three owed earlier. Notes and plan document.

## 35. Refactor (2026-11-16) [verified]

- **Reference frozen first.** `tests/test_pu_regression.py` runs every arm set on small cells (296 arms, 15 cells,
  about 95 s) and compares each field with `tests/golden/pu_compare.json`, made from the code before any change.
- **Policy.** `CouplingPolicy.reveal` is now `step_context` -> `score` -> `admit`. Scoring only orders candidates;
  admission holds the threshold / budget / rate rule (`_fits`), the guard (`_guard_load`) and the pull-in (`_pull`).
  `priority` may be a function of the step context, so a new order needs no change to the class.
- **Comparison.** `compare.py` is the command line only. The library is `experiments/parallel_unmasking/pulib/`:
  `cells` (models, constraints, seeds), `arms` (one registered function per study), `runner`, `metrics`, `gate`.
  Command lines, arm names and random streams are unchanged; the other scripts import through `compare` as before.
- **Checks.** Frozen reference reproduced exactly after each of the two steps. Two full-scale arms of `fix_N256`
  (`lex B=0.03 +pull`, `+pull+guard`; `copy_block`, 100 runs x 16) reproduced bit for bit, 32 numeric fields each.
  All 14 non-biology test files pass (363 tests), 25 of them new (`tests/test_policy_split.py`).
- **Found on the way.** The memory watcher started one thread per cell and never stopped it; it now stops with the
  cell. The full test suite with torch's default threads drove the laptop's load average above 60; run it with
  `OMP_NUM_THREADS=2` (README).
- **Not done here.** `twist_error.py` and `twist_smc.py` are unchanged; they become the test oracle of the chain-twist
  route wrapper when that is written.
- `forced_first` is on by default, so the `random sequential` arm of section 27 reveals forced partners first
  (measured in section 36b: on the full copy at N = 64 it lowers Var(log w_T) from 17.0 to 4.8).

## 36a. Suites, standard report and results log (2026-11-16) [verified]

- **Suites** (`pulib/suites.py`): `core64` (4 models x 4 whole-sequence constraints), `long256` (4 models x 4 long
  constraints), `hidden256` (three hidden chains x 4 long constraints: the evaluation set), `long1024`, `smoke`.
  `compare.py --suite NAME`.
- **Provenance** in every new output: commit, dirty flag, torch version, platform, start time.
- **Standard summary** (`pulib/summary.py`, `report.py`): calls, spread of Z-hat with a 90% bootstrap interval over
  runs, Z-hat / p(C), marginal TV, spread of r, cost = calls x spread^2; ratios to a baseline arm; cell-by-cell
  comparison of two outputs or two arms, a difference being called clear when the intervals are disjoint.
- **Log** `results/parallel_unmasking/log.jsonl`: one line per (run, cell, arm) with settings and commit;
  `report.py --query field=value ...` lists every logged setting of a cell. Backfilled from 22 earlier outputs
  (3,387 results, no commit recorded for those); their per-run files are now in the repo.
- **Checked against earlier reports.** The summary reproduces the section-31 values (`copy_block`, `lex`: 0.40, 0.12,
  0.11 at theta = 0; sequential 0.09). The same arm run in three studies (`fix`, `interact`, `seqorder`) has identical
  numbers, as the named random streams imply.
- **What the intervals add.** On `copy_block` with `lex`, pull-in against plain is clear (0.12 [0.11, 0.14] against
  0.40 [0.33, 0.46]); the guard on top of the pull-in is not resolved (0.11 [0.10, 0.12] against 0.12 [0.11, 0.14]).
  The bootstrap interval is too narrow under heavy tails; read it with `z_se_unreliable`.

## 36b. Exposure against measured weight variance (2026-11-16) [verified at N = 64; N = 256 not run]

`exposure_check.py`: hidden chain (noise 0.3), N = 64, forbidden / counting / copy / dyck, 15 schedules (6 one token per
step, 9 parallel), 1,000 single particles each, no resampling. A probe records, per particle and step, the log-weight
increment and the exposure of the set just chosen. Var(log w_T) is in nats^2 (lower is better).

- **Step level: exposure sets the scale of the increment but is not a bound with constant 2.** Ratio sd(increment) /
  exposure, median over exposure bins, by schedule: forbidden 0.47-1.52, counting 1.37-1.64, copy 0.44-0.89,
  dyck 1.14-3.16. Largest bin: 3.3, 6.8, 1.9, 5.8. The claim of section 32 (sd at most 2 x exposure) is **refuted**
  as stated on dyck and in the top bins of counting.
- **Run level: cumulative exposure does not rank schedules.** Spearman correlation of Var(log w_T) with
  E[sum of squared exposure] over the 15 schedules: +0.51, +0.07, -0.55, +0.45 (exposure added in quadrature over the
  positions of a step); +0.49, +0.81, +0.49, +0.72 (added coherently, which mostly tracks the size of the sets).
  On copy the constraint-first orders have the higher exposure (18.7 against 14.1) and the lower variance
  (1.8-2.0 against 4.5-5.1).
- **Why the step-to-run argument fails: increments of different steps cancel.** Sum over steps of Var(increment)
  divided by Var(log w_T): forbidden 1.1-3.6, copy 1.3-3.3, dyck 4.7-7.4, counting 9.4-19.7. For the global
  constraints almost all step-level fluctuation is undone later. This is the same fact as section 22 (intermediate
  weights do not predict final ones) seen from the other side.
- **Where the order matters, and by how much** (one token per step, Var(log w_T)): forbidden: `bound` 1.4, `lex` 1.5,
  `sum` 3.1, random 11.9, `separator` 24.4 (worse than random). Copy: `lex` 1.8, `bound` 2.0, `sum` 4.5, random with
  forced-first 4.8, `separator` 5.1, random without forced-first 17.0. Counting 0.34-0.51 and dyck 1.6-1.7 (2.5 for
  random without forced-first): no order effect beyond noise.
- **`forced_first` is a large part of the order fix on copy** (17.0 -> 4.8): the inference of section 27 is now
  measured, at N = 64 on the full copy.
- **An exact identity replaces the heuristic** [verified for one-token steps: 12 cases (4 schedules x 3
  constraints, 400 particles each), differences -0.15 to +0.17 nats against standard errors 0.12-0.34; 11 within one
  s.e., the largest 1.14 s.e. (counting, random with forced-first). The s.e. combines the two terms as if independent;
  per-particle differences were not saved. Corrected 2026-11-16: an earlier version said "all within one s.e."]. With the locally optimal proposal the weight is the model's path
  probability over the proposal's, so log W_T = - sum_t log(q_t / p_t), and
      KL(sampler's law || exact constrained law) = E[ sum_t D at the revealed position ] - log(1 / p(C)).
  The cumulative stake paid at reveal, in excess of the fixed total log(1 / p(C)), is the sampler's KL error. It is
  also -E[log(W / p(C))], which every run already gives. Measured KL in nats (lower is better): forbidden: random
  2.6, `bound` 0.77, `separator` 4.1, parallel `lex` + pull + guard 0.78; copy: random 4.6, forced-first 2.0, `bound`
  0.81, `separator` 1.8, parallel `lex` + pull + guard 0.67; counting 0.13-0.20 for all.
  For a parallel step the identity needs the joint KL of the set, not the sum of the positions' stakes. The same
  run included one parallel schedule (`lex theta=0.0 +pull+guard`), where the sum of marginal stakes misses by
  -0.01 (forbidden), -0.24 (counting) and -9.79 nats (copy, where pulled partners are drawn jointly): as expected,
  and not part of the check. The joint form is not yet tested.
- **Consequences.** (1) Exposure is dropped as the unifying criterion; it can stay as a per-step scale. (2) The
  objective for the order is now exact: pay as little stake as possible beyond log(1 / p(C)). By the chain rule the
  excess is the sum over steps of the error of the proposal's marginal at the revealed position, so the order should
  reveal where the product twist's constrained marginal is accurate. A plausible reading of why constraint-first
  wins on forbidden and copy [not tested]: a large D that comes from *revealed* context is exact, a constraint effect
  that arrives through masked neighbours is where the product twist errs. (3) The chain twist gives an estimate of
  exactly that error (the gap between its constrained marginal and the product one), so the two contributions meet.
- **Not done.** The N = 256 run was stopped: the probe's kernel sums run over the full pilot range and would have
  taken hours. Rerun with the pilot noise floor cut (entries below 1e-6), on the `hidden256` suite.
- Files: `results/parallel_unmasking/exposure_N64.json`, `exposure_N64_summary.json`, `exposure_identity_N64.json`.

## 36c. The schedule on every route family (2026-11-16) [verified]

The policy needs one thing from a route: exact constrained marginals, obtained by differentiating `logZ`
(`constrained_marginals`); the pull-in and the mixed term use the same query with clamps.

- **Survey against enumeration** (`tests/test_route_marginals.py`, nothing revealed and two positions revealed).
  Before the fix the query was right on automata, automaton products, the closed-form copy, the subset wrapper and
  factor graphs, and returned NaN on charts (Dyck-1, Dyck-2, automaton x grammar) and on the lifted
  automaton x factor-graph product.
- **Cause and fix.** `torch.logsumexp` has a NaN gradient where every input is -inf, which charts and padded factor
  tables contain by construction. `torch_routes/numerics.logsumexp_safe` gives gradient 0 there; without autograd it
  is `torch.logsumexp` itself, so sampling and weights are unchanged (frozen reference reproduced).
- **After the fix** all ten route types match enumeration to 1e-5 (masked wrapper included).
- **The policy through a chart.** Dyck-1 at N = 12 on a Markov chain, `lex`, theta = 0, 1,500 runs x 2: automaton
  Z-hat / p(C) = 0.992 +- 0.016, 8.12 calls; chart 0.997 +- 0.016, 8.13 calls.
- **Cost of the chart query** (2 CPU threads, per particle): N = 32: 3.8 ms (Dyck-1), 20 ms (Dyck-2); N = 64: 56 ms,
  57 ms. A 64-position grammar suite goes to the cluster; 32 runs locally.
- **Still missing for a grammar study** (experiments, not the method): an exact truth for a chain model under a
  grammar, and for a factor graph. Not covered at all: the spanning-tree route (excluded by decision). On an
  ambiguous grammar the marginals are derivation-weighted, like the twist; the schedule stays legal.

## 36d. Exact answers for grammar constraints (2026-11-16) [verified]

- **`models.GrammarTruth`**: exact log p(C) and per-position marginals for a chain model (Markov or hidden) under a
  grammar in Chomsky normal form, by an inside pass over the product of the grammar and the hidden chain
  (O(N^3 |R| H^3); marginals by autograd, practical to N of about 128).
- **Checked** (`tests/test_grammar_truth.py`): Dyck-1 as a grammar equals Dyck-1 as an automaton; Dyck-2 equals the
  bounded-depth Dyck-2 automaton at N = 12; Dyck-2 with atoms equals a bounded-depth automaton at N = 14 and
  brute-force enumeration at N = 7 (all to 1e-9, Markov and hidden chain); the three grammars are unambiguous (one
  parse per accepted string, by enumeration); the sampler through the chart is unbiased for the new p(C).
- **New constraints** in `pulib/cells.py`: `dyck_chart` (Dyck-1 through the chart, the control for `dyck`), `dyck2`
  (two bracket types, every position constrained: the stress case), `dyck2_atoms` (two bracket types and a free atom:
  the indicative case; grammar `compile.DYCK2_ATOMS`). **New models** `chain_0.85_atoms`, `hidden_0.85_atoms`: 5 symbols,
  60% atoms, 10% each bracket. **New suites** `grammar32`, `atoms32` (local), `grammar64`, `atoms64` (cluster).
- **log p(C) in nats** (closer to 0 = less constraining), at N = 32 | 64 | 128
  (`results/parallel_unmasking/grammar_exact_answers.json`):
  - `dyck` = `dyck_chart`: `chain_0.85` -4.93 | -5.85 | -6.84; `hidden_0.85` -4.90 | -5.82 | -6.80.
  - `dyck2`: `chain_0.85` -7.44 | -10.66 | -16.49; `hidden_0.85` -12.01 | -19.85 | -34.90.
  - `dyck2_atoms`: `chain_0.85_atoms` -2.39 | -4.28 | -7.75; `hidden_0.85_atoms` -7.23 | -11.90 | -20.21
    (on the generic 5-symbol chains, where atoms are 20%: -5.29 | -8.80 | -14.21 and -9.69 | -15.85 | -27.16).
- **Smoke run** (N = 32, 24 runs x 8, pilot 300; not a result): Z-hat / p(C) within its s.e. of 1 in every arm; Dyck-1
  through the automaton and through the chart agree within the intervals (`lex` sequential: spread 0.25 and 0.24;
  `lex` B = 0.03: 0.32 and 0.34 at 10.8 and 10.9 calls). The chart cell took 115 s against 3 s for the automaton.
- Still without an exact truth: factor graphs under a chain.

## 36e. Speed pass (2026-11-16) [verified: frozen reference reproduced exactly, all test files pass]

Profiled on `hidden_0.85` x `window`, N = 256, 400 particles, one CPU thread, and on `dyck2_atoms` at N = 32.

- **Done (exact, local changes).**
  1. Kernel sums: the count of revealed positions was re-binned for every distance on every call; the kernel now
     keeps a lookup from count to value (`DistanceKernel._by_count`).
  2. The gap diagnostic asked the denoiser for the conditionals at a state the sampler had just evaluated; the
     oracle keeps its last answer and the diagnostic is evaluated before the denoiser moves on (one denoiser pass
     per step saved when there is no resampling).
  3. (Earlier today) the memory watcher thread is stopped with its cell.
- **Effect.** One token per step (256 calls): 67.6 s -> 48.4 s. Parallel, `lex` B = 0.03 with pull-in and guard
  (19 calls): 13.6 s -> 10.1 s.
- **Where the time goes now** (one token per step, 48 s): kernel sums 30%, constrained marginals 18%, route logZ 17%,
  route sampling 14%, denoiser 12%. Chart cell (44 s for 192 particles, 24 calls): inside pass 53%, its backward
  pass 27%.
- **Not done, with reasons.**
  - Reusing one route forward pass for the twist, the proposal and the marginals (the same input is evaluated three
    times per step): needs a cache threaded through the sampler; estimated gain about 12% on chart cells, 2-3% on
    automata. Worth doing when chart runs start.
  - A noise floor on the pilot kernel: its range is 145-157 positions with any entry above zero and 37 above 1e-6
    nats; cutting below 1e-6 makes a kernel sum 3-4 times cheaper. It changes which positions the exact threshold
    admits, so it is a method change, not a speed fix.
  - Charts on a GPU: the chart code is batched torch; the 64-position grammar suites go to the cluster GPU.
  - Vectorising the chart's per-nonterminal reduction loop: a real rewrite of a core route; only if GPU runs show it
    matters.

## 36. Next steps (2026-11-16; replaces section 34)

Done since section 34: the selection-fix run (section 31) and the refactor (section 35). One departure from item 2
of section 34: the studies are registered functions behind one command line, not separate scripts, so the existing
job scripts and arm names keep working.

1. Selection: exposure is refuted as the criterion (section 36b). Test the replacement: an order that minimises the
   stake paid beyond log(1 / p(C)), i.e. reveals where the proposal's marginal is accurate; measure KL = -E[log(W / p(C))]
   as the end measure on `hidden256`; then a scale-free score (`sum` fails on MDLM, `lex` costs steps on Dyck),
   held-out constraints, ablations of order, guard, pull-in, forced-first and tie-break.
1a. A real grammar through the chart route (requested 2026-11-16). Prerequisites: constrained marginals from the chart
   (done, section 36c) and an exact truth for a chain model under a grammar (done, section 36d). Then: Dyck-1 through the chart in the same
   cells as through the automaton (should agree within noise); pure Dyck-2 as the stress case; Dyck-2 with a free
   atom symbol (the nesting skeleton of JSON) as the indicative case. Chart cost and the cost of the chain twist on
   a chart (up to the cube of the number of classes) measured separately. Before the 1,024-position cluster job.
2. Chain twist in the main sampler: pair-lift wrapper for automata, positivity mix, no-regression benchmark at 64, 256
   and 1,024 with and without the gate; then grammars with a measured cost; then the partial lift for copy. With many
   classes: full class by default, likely classes kept exactly and the rest merged, loss reported against the number
   of groups.
3. Kernel keyed on how much is revealed; pooling by maximum; the stress oracle with a persistent global latent.
4. Real model: budgets down to 16 calls; the chain twist; then an instruction-following model for structured JSON
   (character-level automaton lifted to tokens; count the classes of one realistic schema first).
5. Cluster (needs approval): interaction study and sequential control at 1,024 on the hidden chains
   (`jobs/interact_N1024.sbatch`, drafted). Finish the two interrupted hidden chains of section 27.
6. Proofs: unbiasedness with a state-dependent order; the closure proposition for the lift; the exposure bound; that a
   coarse-memory chain with exact conditionals is never further in KL than the product; the three owed earlier.
