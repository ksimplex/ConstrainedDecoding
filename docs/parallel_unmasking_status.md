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

## 36f. Chain twist in the main sampler (2026-11-16) [verified for automaton constraints; study in progress]

- **`decode/chain_twist.py`**: `ChainTwist(wdfa, K, eps, active)`, the first-order chain twist of section 29 for automaton
  constraints (whole sequence or a subset of positions), with `log_chain`, `log_psi` (mixture with a share eps of the
  product twist) and `marginals` (constrained marginals under the chain law, by autograd). `fit_pair_kernel` fits K
  on model samples. `twisted_smc(..., twist=...)` uses it as the intermediate twist in the weights; the proposal and
  the terminal twist are unchanged, so without resampling the weights are those of the product twist.
- **Tests** (`tests/test_chain_twist.py`, 6): value and marginals equal brute-force enumeration of the chain law
  (also on a subset of positions, and for states that already break the constraint); equal to the reference of
  `twist_error.py` to 1e-6; eps = 1 is the product twist; in the sampler the classes are identical and log Z-hat
  agrees to 1e-4 without resampling, and Z-hat is unbiased with resampling at every step. Frozen reference
  unchanged with the option off.
- **Not done**: the chain-twisted proposal (the proposal is still the product twist's); grammars and copy; the
  no-regression benchmark at 256 and 1,024.
- **Study `--arms twist`** (`hidden_0.85`, N = 64, 200 runs x 16, 5 automaton constraints x 3 schedules x 2 twists x
  3 resampling settings = 90 arms; complete). Spread of Z-hat = sd / mean over runs (lower is better); "ratio" = spread
  with resampling / spread without (above 1 = resampling hurts).
  - **Random one token per step** (the base sampler), product twist off | defensive | plain, then chain twist:
    - `dyck`: 0.31 | 0.43 | 0.91, then 0.32 | 0.28 | 0.30 (plain ratio 2.97 -> 0.94; events per run 3.4 -> 1.2).
    - `window`: 0.19 | 0.40 | 1.05, then 0.22 | 0.21 | 0.21 (5.53 -> 0.99; events 3.0 -> 0.2).
    - `two_window`: 0.34 | 0.42 | 0.47, then 0.32 | 0.29 | 0.32 (1.36 -> 1.01; events 2.1 -> 0.8).
    - `counting`: 0.16 | 0.17 | 0.15, then 0.16 | 0.14 | 0.15 (resampling never fires at K = 16: no information).
    - `forbidden`: 0.67 | 0.68 | 0.66, then 0.52 | 0.54 | 0.51 (ratios 0.99 and 0.99; the product/chain gap is
      heavy-tail noise, the two "off" arms have the same law).
  - **Over all 15 (constraint, schedule) settings**, ratio for plain resampling: product twist median 1.06, worst 5.53,
    above 1.25 in 4 of 15; chain twist median 0.99, worst 1.13, above 1.25 in none. Defensive: product worst 2.11,
    chain worst 1.17.
  - With the constraint-first order (`lex`, sequential or B = 0.03 with pull-in) resampling fires rarely (0-2.2 events
    per run) and there is little harm to remove: product 0.88-1.26, chain 0.80-1.13.
  - **Unbiasedness**: on the one-token-per-step arms (exact schedules), (Z-hat / p(C) - 1) / s.e. has largest absolute
    value 2.51 over the 20 chain-twist arms with resampling and 2.98 over the 20 product-twist ones. Under the budget
    Z-hat estimates that schedule's own hit rate (0.90-0.98 of p(C) here), as before.
  - **Reading.** The chain twist does what section 29 found in the reference sampler, now in the main one: it makes
    resampling safe (never more than 1.13 times worse than none, against up to 5.5 times) and gains nothing by itself
    without resampling. In these cells resampling with the chain twist is roughly neutral, not a gain; whether it
    becomes a gain needs longer sequences or fewer particles, where degeneracy bites.
  - Files: `results/parallel_unmasking/twist_N64.json` (+ `.logZ.npz`), `twist_N64_summary.json`; logged.

## 36g. Is the chain-product disagreement an estimate of the product twist's error? (2026-11-16) [verified at N = 64]

`error_estimate_check.py`: three hidden chains (noise 0.15, 0.3, 0.5), N = 64, counting / dyck / forbidden / window /
two_window, 60 states per cell from the exact constrained law at reveal fractions 0.1-0.75. Per masked position:
true error eps = KL(product twist's constrained marginal || exact conditional) and the estimate
est = KL(product's || chain twist's). All in nats; `ratio` = pilot chain as in the sampler, `exact` = the model's own
pair conditionals.

- **Identity** [derived]: eps = est + KL(chain || exact) + sum (q0 - q1) log(q1 / pi); to leading order
  |sqrt(eps) - sqrt(est)| <= sqrt(chain's own error). At the level of unconstrained joints the errors add exactly
  (Pythagoras for nested exponential families), so there the disagreement is a lower bound; not so for constrained
  per-position marginals.
- **Sanity**: on the Markov chain the exact-pair chain's error is zero (ratio 6e-8).
- **The chain twist is only moderately better per position** (root-KL error relative to the product's, median over the
  costlier half): counting 0.85, dyck 0.58, forbidden 0.63, window 0.48, two_window 0.44 (`exact`: 0.53, 0.54, 0.32,
  0.30, 0.33). The whole-state figures of section 29 (0.2-0.3) were more favourable.
- **Calibration** (median est / eps): 0.48, 0.25, 1.62, 1.09, 0.68. **Tail**, share of the costlier half
  underestimated more than tenfold: 0.23, 0.34, 0.08, 0.09, 0.12 (more than a hundredfold: 0.04-0.08);
  overestimated more than tenfold: 0.02, 0.01, 0.17, 0.09, 0.05.
- **Ranking within a state** (Spearman): 0.12, 0.47, 0.72, 0.58, 0.60 (`exact`: 0.55-0.86). The costliest tenth of
  positions is ranked above the estimate's median in 77%, 94%, 91%, 98%, 97% of cases.
- **True error at the position each rule would reveal next**, positions that are not forced, nats x 1000:

  | Constraint | Ideal | Lowest estimate | Lowest entropy | Highest stake | Separator | Random |
  |---|---|---|---|---|---|---|
  | counting | 0.13 | 2.57 | 2.02 | 5.80 | 9.11 | 3.21 |
  | dyck | 0.08 | 5.87 | 5.32 | 29.5 | 23.7 | 13.8 |
  | forbidden | 0.84 | 3.45 | 3.95 | 13.3 | 59.7 | 28.6 |
  | window | 0.00 | 0.29 | 2.10 | 16.3 | 9.89 | 4.67 |
  | two_window | 0.06 | 1.92 | 4.70 | 23.3 | 30.3 | 13.2 |

- **Reading.** (1) The estimate is one-sided: a high value marks a costly position, a low value certifies nothing;
  on the global constraint (counting) both twists miss the long-range dependence and it carries little information.
  Unsafe as an admission test against an absolute budget; usable as a ranking signal. (2) Lowest entropy of the
  constrained marginal (the old `confident` order) does about as well on three constraints without any chain twist;
  its earlier weakness was steps (29.7 against 7-14), not accuracy. (3) Highest stake is worse than random at one
  step on four of five, yet constraint-first did well in full runs on the windows: one-step error is not the whole
  story, and "cheapest first" needs a full-run test. (4) The ideal pick is far better than the best practical rule on every constraint: by a factor of 15 (counting), 68 (dyck), 4 (forbidden), 947 (window), 32 (two_window). On `window` the ideal error is essentially zero (3e-7 nats; the table shows it as 0.00). (Corrected 2026-11-16: an earlier version said "4-70 times", which left out `window`.)
- **Why counting is the hard case** [rough calculation, not measured]: a first-order chain fitted on adjacent
  tokens predicts dependence decaying as the adjacent correlation to the power of the distance; with a hidden state
  the true decay is slower, and at noise 0.3 the chain captures about a quarter of the covariance summed over all
  distances. A constraint that sums over all positions feels all of the shortfall. A chain over a small latent state
  (same marginal matching) is the natural richer stand-in; not built.
- Files: `results/parallel_unmasking/error_estimate_N64.json`, `error_estimate_N64_summary.json`.

## 36h. Second-order chain twist: error against cost (2026-11-16) [verified, reference implementation only]

`twist_order.py`: product, first-order and second-order chain stand-ins against the exact twist, in the pooled-table
version (`ratio`: pairs K, triples K2 from 20,000 model samples, marginal-matched, no denoiser call) and with the
model's own conditionals (`exact`). Three hidden chains x five automaton constraints at N = 64 (120 states per cell,
four reveal fractions); `hidden_0.85` x forbidden / window / two_window at N = 256 (60 states). Not in the sampler.

- **Sanity**: on the Markov chain both exact chains have zero error (5e-15 nats whole-state).
- **Spread of the whole-state log-twist error** (nats, lower is better; what moves the weights), N = 64,
  product | order 1 pooled | order 2 pooled | order 1 exact | order 2 exact:
  counting 0.62 | 0.14 | 0.10 | 0.22 | 0.10; dyck 0.80 | 0.23 | 0.14 | 0.32 | 0.15; forbidden 0.65 | 0.39 | 0.17 | 0.47 | 0.19;
  window 0.69 | 0.16 | 0.10 | 0.22 | 0.09; two_window 0.41 | 0.13 | 0.11 | 0.09 | 0.02.
  At N = 256: forbidden 1.42 | 0.96 | 0.37 | 1.12 | 0.48; window 1.65 | 0.45 | 0.15 | 0.61 | 0.24; two_window 1.29 | 0.33 | 0.14 | 0.42 | 0.16.
  Order 2 over order 1 (pooled): 1.1-2.3 times lower at 64, 2.4-3.0 times lower at 256.
- **Mean KL per position** (nats x 1000; the error a proposal built on the stand-in would add at a reveal; costlier half
  of positions), N = 64, same order of columns: counting 6.0 | 6.6 | 4.7 | 1.6 | 0.5; dyck 24.5 | 8.6 | 3.7 | 6.5 | 1.9;
  forbidden 53.7 | 12.1 | 5.4 | 6.2 | 2.6; window 36.2 | 8.1 | 4.4 | 6.4 | 2.0; two_window 59.0 | 10.5 | 3.8 | 9.9 | 1.5.
  Order 1 pooled cuts the product's by 2.8-5.6 times except on counting (no gain); order 2 pooled cuts a further
  1.4-2.7 times; exact conditionals reach 12-39 times below the product at order 2.
- The median position gains less than the mean (per-position root-KL relative to the product, median: order 1 pooled
  0.58-1.02, order 2 pooled 0.39-0.85): the chains remove the largest errors first.
- **Cost** (ms per state, one CPU thread, three classes): N = 64: product 0.04, order 1 0.09, order 2 0.19;
  N = 256: 0.14, 0.43, 0.77. Order 2 costs about twice order 1 here; in general a factor of the class count.
- Files: `results/parallel_unmasking/twist_order_N64.json`, `twist_order_N256.json`, `twist_order_summary.json`.

## 36i. Order study: cheapest-first to separator-first (cluster job `91729edd`, 2026-11-16) [verified at N = 64]

Setting: 3 hidden chains (noise 0.15, 0.3, 0.5) x 5 automaton constraints, 64 positions, 100 runs x 16 particles, 43 arms
per cell, 645 results, no errors. Code commit `7bb07b0`. Files: `results/parallel_unmasking/axis_N64.json`,
`axis_N64.logZ.npz`, `axis_N64_summary.json`; logged to `log.jsonl`.

Orders: `cheapest` ranks by the true error of the next reveal using the exact twist (the ideal greedy reference, oracle
models only); `estimate` ranks by the chain-against-product disagreement; `confident` is lowest entropy; `lex` is
constraint-first; `sum` is stake plus coupling; `separator` is highest coupling; `random` is uniform.

**Check of the exact arms.** The 210 sequential arms should estimate p(C) without bias. 2.9% are more than two
standard errors from the truth (chance level); the largest deviation is 3.38 standard errors (`confident`, Dyck, noise 0.15).

**Sequential orders, resampling off.** KL(sampler || exact constrained law) per particle in nats, lower is better.
Geometric mean over the 3 chains. All arms use 64 calls. Largest standard error of any entry before averaging: 0.2.

| constraint | mode | cheapest | estimate | confident | lex | sum | separator | random |
|---|---|---|---|---|---|---|---|---|
| counting | sequential | 0.152 | 0.189 | 0.415 | 0.180 | 0.101 | 0.126 | 0.159 |
| dyck | sequential | 0.232 | 0.383 | 0.668 | 0.506 | 0.525 | 0.511 | 0.561 |
| forbidden | sequential | 0.548 | 0.595 | 0.540 | 0.589 | 1.009 | 3.806 | 2.416 |
| window | sequential | 0.051 | 0.444 | 0.061 | 0.073 | 0.069 | 0.314 | 0.351 |
| two_window | sequential | 0.258 | 0.716 | 0.145 | 0.085 | 0.098 | 0.464 | 0.658 |

Mean rank over the 15 cells (1 = lowest KL): cheapest 2.40, estimate 4.87, confident 4.07, lex 3.27, sum 2.87, separator 4.80, random 5.73.
Times lowest: cheapest 5, estimate 0, confident 3, lex 4, sum 3, separator 0, random 0.

Differences per chain (noise 0.3, 0.15, 0.5), in standard errors of the difference:

| constraint | comparison | noise 0.3 | noise 0.15 | noise 0.5 |
|---|---|---|---|---|
| two_window | cheapest against lex | 0.269 against 0.090 (+8.1 s.e.) | 0.461 against 0.212 (+8.2 s.e.) | 0.139 against 0.032 (+7.4 s.e.) |
| two_window | cheapest against confident | 0.269 against 0.161 (+4.5 s.e.) | 0.461 against 0.301 (+5.0 s.e.) | 0.139 against 0.063 (+4.9 s.e.) |
| window | estimate against cheapest | 0.504 against 0.051 (+16.9 s.e.) | 0.399 against 0.105 (+10.3 s.e.) | 0.435 against 0.025 (+16.8 s.e.) |
| window | estimate against random | 0.504 against 0.363 (+4.0 s.e.) | 0.399 against 0.570 (-4.2 s.e.) | 0.435 against 0.210 (+7.8 s.e.) |
| window | cheapest against lex | 0.051 against 0.070 (-1.4 s.e.) | 0.105 against 0.247 (-6.2 s.e.) | 0.025 against 0.023 (+0.3 s.e.) |
| dyck | cheapest against estimate | 0.254 against 0.424 (-5.5 s.e.) | 0.559 against 0.727 (-3.8 s.e.) | 0.089 against 0.182 (-5.1 s.e.) |
| dyck | cheapest against lex | 0.254 against 0.611 (-9.4 s.e.) | 0.559 against 1.138 (-10.1 s.e.) | 0.089 against 0.186 (-5.1 s.e.) |
| counting | cheapest against sum | 0.176 against 0.122 (+2.6 s.e.) | 0.324 against 0.197 (+4.4 s.e.) | 0.062 against 0.043 (+1.6 s.e.) |
| counting | cheapest against random | 0.176 against 0.179 (-0.1 s.e.) | 0.324 against 0.336 (-0.4 s.e.) | 0.062 against 0.067 (-0.4 s.e.) |
| counting | confident against random | 0.508 against 0.179 (+11.0 s.e.) | 0.842 against 0.336 (+11.8 s.e.) | 0.168 against 0.067 (+5.8 s.e.) |
| forbidden | cheapest against confident | 0.615 against 0.576 (+0.9 s.e.) | 1.020 against 1.011 (+0.2 s.e.) | 0.262 against 0.270 (-0.3 s.e.) |
| forbidden | cheapest against random | 0.615 against 2.662 (-23.4 s.e.) | 1.020 against 5.092 (-29.7 s.e.) | 0.262 against 1.040 (-16.5 s.e.) |
| forbidden | separator against random | 4.248 against 2.662 (+11.1 s.e.) | 8.477 against 5.092 (+14.1 s.e.) | 1.532 against 1.040 (+6.7 s.e.) |

Readings:
- No order is lowest on every constraint.
- Ideal greedy cheapest-first is not the optimum. It is lowest on Dyck and tied lowest on the single window and on
  forbidden, but on two windows constraint-first is 2 to 4 times lower on every chain (7 to 8 standard errors), and on
  counting it is no better than random and worse than `sum` (resolved on two of the three chains). This is the myopia
  expected of a greedy rule, now measured.
- The estimate order is never lowest. On the single window it is 4 to 17 times worse than the ideal order across the
  chains (0.399 to 0.504 against 0.025 to 0.105), and about 3 times worse on two windows.
- Lowest entropy is the worst order on counting, and among the best on forbidden and the windows.
- Separator-first is the worst order on forbidden (worse than random) and second lowest on counting.

**Why the estimate order fails on windows** (`experiments/parallel_unmasking/order_attribution.py`, local, noise 0.3,
400 particles). The sampler's KL equals the sum over steps of the true error at the revealed position, so it can be
split by where the revealed position lies: inside the window, near it (within 6 positions, one correlation length), or
far. `true` and `est` are the summed true error and the summed estimate at the same positions, in nats per particle.
`step` is the mean step at which positions of that group are revealed.

| constraint | order | KL (sum of steps) | true inside | true near | true far | est inside | est near | est far | step inside | step near | step far | cluster KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| window | estimate | 0.467 | 0.060 | 0.326 | 0.081 | 0.033 | 0.015 | 0.000 | 57 | 49 | 22 | 0.504 ± 0.025 |
| window | cheapest | 0.053 | 0.019 | 0.014 | 0.019 | 0.046 | 0.028 | 0.139 | 49 | 51 | 23 | 0.051 ± 0.008 |
| window | lex | 0.074 | 0.074 | 0.000 | 0.000 | 0.072 | 0.016 | 0.041 | 8 | 36 | 35 | 0.070 ± 0.011 |
| window | random | 0.371 | 0.086 | 0.262 | 0.024 | 0.133 | 0.052 | 0.049 | 29 | 32 | 32 | 0.363 ± 0.025 |
| two_window | estimate | 0.861 | 0.084 | 0.619 | 0.158 | 0.050 | 0.033 | 0.000 | 54 | 42 | 18 | 0.835 ± 0.035 |
| two_window | cheapest | 0.285 | 0.079 | 0.119 | 0.087 | 0.061 | 0.078 | 0.098 | 42 | 40 | 23 | 0.269 ± 0.019 |
| two_window | lex | 0.087 | 0.087 | 0.000 | 0.000 | 0.064 | 0.031 | 0.032 | 7 | 36 | 34 | 0.090 ± 0.011 |
| two_window | random | 0.741 | 0.105 | 0.577 | 0.059 | 0.163 | 0.145 | 0.034 | 28 | 32 | 32 | 0.756 ± 0.033 |

- The estimate order reveals the far and near positions before the window, because the estimate rates them almost
  free: 0.000 summed over the far positions and 0.015 over the near ones on the single window, against true errors of
  0.081 and 0.326. Most of its KL is paid at the near positions.
- Revealing the window late is not the cause in itself: the ideal order also reveals it late and has the lowest KL.
- Constraint-first pays only inside the window. Once the window is resolved every other position is free.
- This is the failure anticipated in section 36g: where the product and chain twists are wrong in the same way, their
  disagreement is small and the estimate understates the error.
- An earlier hypothesis, that the estimate is exactly zero everywhere outside the window, was wrong: its largest value
  outside the window at the fully masked state is 0.056.
- Check of the attribution: the step-sum agrees with the cluster's weight-based KL in all 8 cases. The local
  weight-based KL (not tabulated) agrees in 7; for `lex` on the single window it is 0.011 ± 0.019 against the
  cluster's 0.070 ± 0.011, 2.7 standard errors apart, taken to be a fluctuation of a heavy-tailed mean at 400 particles.

**Parallel modes.** Denoiser calls per particle, lower is faster, resampling off:

| constraint | mode | cheapest | estimate | confident | lex | sum | separator | random |
|---|---|---|---|---|---|---|---|---|
| counting | B=0.03 +pull | 15.2 | 14.7 | 17.7 | 12.7 | 10.8 | 11.0 | 12.7 |
| counting | B=0.3 +pull | 10.2 | 9.6 | 10.1 | 7.6 | 6.7 | 6.6 | 7.9 |
| dyck | B=0.03 +pull | 17.8 | 14.8 | 16.1 | 11.7 | 10.1 | 10.0 | 11.5 |
| dyck | B=0.3 +pull | 9.6 | 8.9 | 8.5 | 6.8 | 5.7 | 5.9 | 6.8 |
| forbidden | B=0.03 +pull | 19.4 | 19.5 | 19.0 | 20.7 | 12.1 | 10.9 | 12.8 |
| forbidden | B=0.3 +pull | 13.2 | 13.2 | 11.0 | 12.4 | 7.9 | 6.7 | 8.0 |
| window | B=0.03 +pull | 16.7 | 17.1 | 20.1 | 11.4 | 10.8 | 10.7 | 12.7 |
| window | B=0.3 +pull | 10.0 | 10.0 | 11.4 | 6.8 | 6.7 | 6.7 | 8.0 |
| two_window | B=0.03 +pull | 17.2 | 18.2 | 19.4 | 11.5 | 11.1 | 10.7 | 12.7 |
| two_window | B=0.3 +pull | 9.9 | 10.4 | 11.9 | 6.8 | 6.7 | 6.6 | 8.0 |

Cost = calls x spread^2, lower is better, where spread is the sd/mean of the estimate of p(C) over runs. Resampling off:

| constraint | mode | cheapest | estimate | confident | lex | sum | separator | random |
|---|---|---|---|---|---|---|---|---|
| counting | sequential | 1.28 | 1.47 | 6.75 | 4.62 | 0.59 | 0.57 | 1.37 |
| counting | B=0.03 +pull | 1.04 | 0.85 | 0.85 | 0.37 | 0.28 | 0.32 | 0.51 |
| counting | B=0.3 +pull | 0.50 | 0.37 | 0.35 | 0.21 | 0.19 | 0.16 | 0.31 |
| dyck | sequential | 1.96 | 4.29 | 5.51 | 3.41 | 2.77 | 4.11 | 4.52 |
| dyck | B=0.03 +pull | 1.51 | 1.76 | 1.35 | 0.77 | 0.84 | 0.62 | 0.70 |
| dyck | B=0.3 +pull | 0.71 | 0.74 | 0.93 | 0.48 | 0.41 | 0.39 | 0.62 |
| forbidden | sequential | 5.19 | 5.39 | 5.51 | 4.74 | 11.83 | 26.19 | 20.78 |
| forbidden | B=0.03 +pull | 2.63 | 3.54 | 2.83 | 4.03 | 2.91 | 4.22 | 4.16 |
| forbidden | B=0.3 +pull | 3.54 | 2.59 | 3.07 | 3.09 | 2.71 | 3.08 | 4.22 |
| window | sequential | 0.39 | 3.48 | 0.46 | 0.51 | 0.48 | 1.47 | 2.30 |
| window | B=0.03 +pull | 0.07 | 0.99 | 0.20 | 0.11 | 0.13 | 0.34 | 0.45 |
| window | B=0.3 +pull | 0.07 | 0.29 | 0.17 | 0.14 | 0.12 | 0.23 | 0.30 |
| two_window | sequential | 2.08 | 6.67 | 1.47 | 0.51 | 0.62 | 2.80 | 5.23 |
| two_window | B=0.03 +pull | 0.28 | 1.07 | 0.12 | 0.08 | 0.08 | 1.15 | 1.20 |
| two_window | B=0.3 +pull | 0.06 | 1.04 | 0.17 | 0.08 | 0.09 | 0.60 | 0.81 |

- Orders that rank by coupling (`sum`, `separator`) need the fewest calls; `cheapest`, `estimate` and `confident` need
  1.3 to 2 times more at the same budget.
- The lowest-cost arm is a parallel arm in all 15 cells (`sum` in 6, `cheapest` in 5, `lex` in 2, `separator` and
  `estimate` in 1 each; B = 0.3 in 10, B = 0.03 in 5).
- **Caveat that changes how to read the cost table.** Cost measures spread around the arm's own mean. A budgeted
  parallel schedule is not exact, and the bias against the exact law is not in the cost:

| mode | arms | Z_hat / p(C): min | median | max | share more than 2 s.e. from 1 | median marginal TV error |
|---|---|---|---|---|---|---|
| sequential | 105 | 0.858 | 1.001 | 1.099 | 4% | 0.0158 |
| B=0.03 +pull | 105 | 0.815 | 0.995 | 1.083 | 14% | 0.0171 |
| B=0.3 +pull | 105 | 0.585 | 0.958 | 1.206 | 74% | 0.0210 |

  At B = 0.3 most arms sample a measurably different law (forbidden: 0.62 to 0.74 of the true p(C)). The marginal TV
  column is at 16 particles and is dominated by sampling noise, so it does not separate bias from noise. This study
  therefore does not answer "error against the exact law at equal calls" for the budgeted arms.

**Chain twist with defensive resampling against resampling off.** Ratio of spreads over the 315 matched pairs: median
0.98, 10th to 90th percentile 0.81 to 1.14, range 0.40 to 2.05. Below 1 means resampling helped.

| constraint | median ratio | min | max | resampling events per run |
|---|---|---|---|---|
| counting | 1.02 | 0.40 | 2.05 | 0.21 |
| dyck | 0.98 | 0.74 | 1.25 | 0.60 |
| forbidden | 0.88 | 0.60 | 1.14 | 1.80 |
| window | 1.00 | 0.78 | 1.19 | 0.08 |
| two_window | 1.02 | 0.75 | 1.32 | 0.27 |

Neutral at this length except on forbidden, where resampling fires most. Consistent with section 36f.

## 36j. Where the time of a step goes on MDLM (cluster job `1fc965f6`, one L40S, 2026-11-16) [verified: 12 cells, per-stage split; small pilot]

`experiments/parallel_unmasking/mdlm_timing.py`, commit `5c12f04`; file `results/parallel_unmasking/mdlm_timing_run1.json`.
Half-revealed state, three token classes, two constraints: a count on a window of a tenth of the positions and a count
on the whole sequence. Median of 7 repetitions, milliseconds for the whole batch, lower is better. The last three
columns are in denoiser calls at the same batch size.

| N | particles | constraint | denoiser | class projection | kernel sum | route logZ | route marginals | reveal, sequential | reveal, parallel | chain twist | chain twist (calls) | step, sequential (calls) | step, parallel (calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 8 | window | 9.0 | 0.9 | 5.6 | 0.7 | 3.6 | 10.7 | 48.5 | 7.4 | 0.82 | 2.67 | 6.86 |
| 128 | 8 | whole_sequence | 9.0 | 0.9 | 5.6 | 6.3 | 32.1 | 38.4 | 191.2 | 10.1 | 1.12 | 8.73 | 25.69 |
| 128 | 64 | window | 30.0 | 7.6 | 5.6 | 0.7 | 3.5 | 11.1 | 61.1 | 7.4 | 0.25 | 1.74 | 3.40 |
| 128 | 64 | whole_sequence | 30.0 | 7.6 | 5.6 | 6.3 | 32.8 | 38.6 | 320.4 | 10.2 | 0.34 | 3.55 | 12.94 |
| 128 | 256 | window | 155.1 | 30.3 | 5.6 | 0.7 | 3.6 | 10.7 | 62.6 | 7.4 | 0.05 | 1.29 | 1.62 |
| 128 | 256 | whole_sequence | 155.1 | 30.3 | 5.6 | 6.3 | 32.4 | 38.7 | 323.9 | 10.2 | 0.07 | 1.64 | 3.48 |
| 256 | 8 | window | 10.7 | 1.9 | 11.1 | 1.4 | 6.8 | 19.4 | 108.8 | 14.4 | 1.36 | 3.60 | 12.00 |
| 256 | 8 | whole_sequence | 10.7 | 1.9 | 11.1 | 12.5 | 64.3 | 75.8 | 626.5 | 20.1 | 1.89 | 14.02 | 65.72 |
| 256 | 64 | window | 65.3 | 15.1 | 11.1 | 1.4 | 6.9 | 19.8 | 112.7 | 14.5 | 0.22 | 1.63 | 3.06 |
| 256 | 64 | whole_sequence | 65.3 | 15.1 | 11.1 | 12.6 | 64.6 | 77.7 | 637.6 | 20.3 | 0.31 | 3.35 | 11.92 |

- The run stopped at N = 256 with 256 particles: the full-vocabulary log-softmax of that batch needs 12.3 GiB on top of
  35 GiB held. That cell is not measured.
- Route, kernel and twist times do not grow with the number of particles; they are about 50 to 60 microseconds per
  position step, which is GPU per-operation overhead. The script passes the full particle batch to each timer
  (read from the code; a shape assertion is added in the rerun).
- The chain twist costs 0.05 to 0.34 of a denoiser call at 64 or more particles, and 0.8 to 1.9 at 8.
- The constraint-aware sequential reveal equals one constrained-marginals pass plus one kernel sum.
- The parallel reveal is 4.5 to 8.4 times dearer than the sequential one. From the code, the candidates are the
  forced-partner pull-in (up to 8 extra constrained-marginals passes per step) and an admission loop over all
  positions. Component times predict 38 + 8 x 32 = 294 ms at N = 128 on the whole-sequence constraint against 320 to
  324 measured. This is a prediction, not a stage measurement.
- This run did not record how many positions a parallel reveal admits, so its cost per revealed position is unknown.
- The run used rate 1/N for the parallel reveal, which is a whole-sequence budget of B = 1, looser than the studies'
  0.03 and 0.3.
- Rerun with per-stage timers, a no-pull-in variant and the admitted count (commit `7395e53`): the second and third
  submissions (nodes node4204 and node3403) hung at Python start-up or `import torch` before any timing code ran and
  were cancelled at their time limits; a 5-minute `timeout` did not end the third. A fresh Python from the same
  environment started in 2.7 s on the login node afterwards (user's check). A fourth submission with staged start-up
  diagnostics (job `48847543`, node3502) located the problem: a plain `ls` of a file under `~/orcd/pool` did not
  return in 120 s and was hard-killed, while the job's working directory under the home filesystem worked. So the
  pooled storage, which holds the environment and the model cache, was unreachable from three GPU nodes between 02:01
  and at least 05:03 (four GPU nodes in all; node-specific, see the cluster notes). No timing code ran in attempts 2
  to 5. The sixth, pinned to a node that passed a storage probe, completed; its results follow. Section 36l has the
  stage split on CPU.

**Per-stage rerun (job `b41c1480`, node3203, commit `7395e53`, all 12 cells; file `results/parallel_unmasking/mdlm_timing_stages.json`).**
Same settings as the first run, with the reveal timed by stage, a variant without the pull-in, and the positions admitted.
Milliseconds for the whole batch, lower is better; the last three columns are one sampler step in denoiser calls at the
same batch size.

| N | particles | constraint | denoiser | reveal, sequential | reveal, parallel | reveal, parallel, no pull-in | pull-in | pull-in passes | positions per step (with, without pull-in) | step, sequential (calls) | step, parallel (calls) | step, parallel, no pull-in (calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 8 | window | 9.1 | 10.6 | 48.1 | 25.1 | 15.5 | 4 | 2.0, 2.0 | 2.64 | 6.77 | 4.23 |
| 128 | 8 | whole_sequence | 9.1 | 38.3 | 188.6 | 52.8 | 127.5 | 4 | 2.0, 2.0 | 8.68 | 25.23 | 10.28 |
| 128 | 64 | window | 30.2 | 10.7 | 59.6 | 28.0 | 30.6 | 8 | 2.0, 2.0 | 1.71 | 3.33 | 2.29 |
| 128 | 64 | whole_sequence | 30.2 | 38.3 | 315.4 | 55.9 | 257.3 | 8 | 2.0, 2.0 | 3.53 | 12.70 | 4.11 |
| 128 | 256 | window | 155.7 | 10.7 | 62.0 | 30.5 | 30.5 | 8 | 2.0, 2.0 | 1.28 | 1.61 | 1.41 |
| 128 | 256 | whole_sequence | 155.7 | 38.3 | 317.9 | 58.3 | 257.6 | 8 | 2.0, 2.0 | 1.64 | 3.43 | 1.76 |
| 256 | 8 | window | 10.7 | 19.5 | 106.2 | 48.0 | 55.8 | 8 | 2.6, 2.6 | 3.60 | 11.70 | 6.26 |
| 256 | 8 | whole_sequence | 10.7 | 75.4 | 615.8 | 104.0 | 509.2 | 8 | 2.6, 2.6 | 13.89 | 64.36 | 16.56 |
| 256 | 64 | window | 65.4 | 19.5 | 110.3 | 52.3 | 56.3 | 8 | 2.6, 2.6 | 1.63 | 3.01 | 2.13 |
| 256 | 64 | whole_sequence | 65.4 | 75.5 | 623.6 | 108.6 | 513.5 | 8 | 2.6, 2.6 | 3.32 | 11.69 | 3.82 |
| 256 | 256 | window | 315.1 | 19.2 | 115.9 | 58.4 | 56.7 | 8 | 2.5, 2.5 | 1.27 | 1.58 | 1.39 |
| 256 | 256 | whole_sequence | 315.1 | 75.3 | 628.3 | 114.5 | 513.0 | 8 | 2.5, 2.5 | 1.62 | 3.38 | 1.75 |

- The pull-in is 32 to 83% of the parallel reveal's time. It runs up to 8 extra constrained-marginals passes per step.
- It admitted nothing in this state: positions per step are identical with and without it in all 12 cells.
- Without the pull-in the parallel reveal costs 1.4 to 3.0 times the sequential reveal; with it, 4.5 to 8.3 times.
- Limits of this measurement: one half-revealed state per cell, a pilot of only 8 traces, and rate 1/N (B = 1). The
  schedule admits 2.0 to 2.6 positions per step here, far fewer than a properly piloted schedule would, so the
  per-step costs are measured but the cost per revealed position is not representative.
- The N = 256, 256-particle cell that ran out of memory in the first run fits once nothing large is held while the
  denoiser is timed.

## 36k. A resampling pressure case: 256 positions, whole-sequence constraints (cluster job `a16667c2`, 2026-11-16) [verified; small samples]

Setting: `hidden_0.85`, 256 positions, forbidden, counting and Dyck; the native random order one token per step
(exact, 256 calls) and constraint-first under B = 0.03 with pull-in (15 to 26 calls); product against chain twist;
resampling off, defensive, plain. 50 runs of 16 particles and 12 runs of 64 particles. Code commit `5c12f04`.
Files: `results/parallel_unmasking/pressure_K16.json`, `pressure_K64.json`, `pressure_N256_summary.json`; logged.
Exact log p(C): forbidden -42.4, counting -3.98, Dyck -7.82.

Criterion stated before the run: poor accuracy without resampling under both twists, unreliable resampling under the
product twist, and a clear gain from resampling under the chain twist.

**Which statistic.** The usual spread (sd/mean of the estimate of p(C)) is not usable on forbidden: 19 of its 24
arms are flagged heavy-tailed (21 of 72 overall), and the two resampling-off arms, which are the same sampler under
different seeds, give spreads of 2.53 and 0.88 on one schedule. The table therefore reports the root-mean-square over
runs of log(estimate / p(C)) in nats, lower is better, with the mean ± its standard error in brackets (0 is ideal; an
unbiased estimator has mean at most 0). On this statistic the two replicate off arms agree.

16 particles, 50 runs:

| constraint | schedule | calls | twist | off | defensive | plain | resampling events (defensive) | ESS, off |
|---|---|---|---|---|---|---|---|---|
| forbidden | fixed 1/step | 256 | product | 3.32 (-2.29 ± 0.34) | 2.18 (-1.31 ± 0.25) | 2.68 (-1.78 ± 0.29) | 20.8 | 0.064 |
| forbidden | fixed 1/step | 256 | chain | 3.59 (-2.45 ± 0.37) | 1.39 (-0.71 ± 0.17) | 1.98 (-1.31 ± 0.21) | 17.4 | 0.065 |
| forbidden | lex B=0.03 +pull | 26 | product | 1.39 (-0.85 ± 0.16) | 1.20 (-0.79 ± 0.13) | 1.34 (-1.00 ± 0.13) | 6.5 | 0.095 |
| forbidden | lex B=0.03 +pull | 26 | chain | 1.35 (-1.00 ± 0.13) | 0.98 (-0.45 ± 0.12) | 1.03 (-0.41 ± 0.14) | 5.5 | 0.099 |
| counting | fixed 1/step | 256 | product | 0.16 (-0.00 ± 0.02) | 0.15 (-0.01 ± 0.02) | 0.15 (+0.01 ± 0.02) | 0.1 | 0.640 |
| counting | fixed 1/step | 256 | chain | 0.14 (-0.04 ± 0.02) | 0.14 (+0.01 ± 0.02) | 0.15 (-0.01 ± 0.02) | 0.0 | 0.759 |
| counting | lex B=0.03 +pull | 17 | product | 0.17 (-0.03 ± 0.02) | 0.19 (+0.02 ± 0.03) | 0.17 (+0.02 ± 0.02) | 0.2 | 0.583 |
| counting | lex B=0.03 +pull | 17 | chain | 0.17 (-0.03 ± 0.02) | 0.20 (-0.01 ± 0.03) | 0.21 (-0.01 ± 0.03) | 0.1 | 0.652 |
| dyck | fixed 1/step | 256 | product | 0.48 (-0.11 ± 0.07) | 0.68 (-0.16 ± 0.09) | 0.88 (-0.57 ± 0.09) | 6.4 | 0.174 |
| dyck | fixed 1/step | 256 | chain | 0.44 (-0.09 ± 0.06) | 0.39 (-0.02 ± 0.06) | 0.42 (-0.06 ± 0.06) | 2.3 | 0.292 |
| dyck | lex B=0.03 +pull | 16 | product | 0.39 (-0.03 ± 0.06) | 0.38 (-0.18 ± 0.05) | 0.59 (-0.28 ± 0.07) | 2.7 | 0.297 |
| dyck | lex B=0.03 +pull | 16 | chain | 0.37 (-0.07 ± 0.05) | 0.39 (-0.09 ± 0.05) | 0.33 (-0.08 ± 0.05) | 1.2 | 0.348 |

64 particles, 12 runs:

| constraint | schedule | calls | twist | off | defensive | plain | resampling events (defensive) | ESS, off |
|---|---|---|---|---|---|---|---|---|
| forbidden | fixed 1/step | 256 | product | 2.13 (-1.45 ± 0.47) | 1.35 (-0.93 ± 0.29) | 1.00 (-0.64 ± 0.23) | 24.8 | 0.019 |
| forbidden | fixed 1/step | 256 | chain | 2.38 (-0.69 ± 0.69) | 0.66 (-0.13 ± 0.20) | 1.05 (-0.58 ± 0.26) | 20.2 | 0.018 |
| forbidden | lex B=0.03 +pull | 26 | product | 0.84 (-0.38 ± 0.22) | 0.60 (-0.25 ± 0.16) | 0.79 (-0.51 ± 0.18) | 7.9 | 0.037 |
| forbidden | lex B=0.03 +pull | 26 | chain | 0.94 (-0.32 ± 0.27) | 0.37 (-0.08 ± 0.11) | 0.55 (-0.35 ± 0.13) | 6.4 | 0.040 |
| counting | fixed 1/step | 256 | product | 0.07 (-0.03 ± 0.02) | 0.10 (+0.00 ± 0.03) | 0.07 (+0.02 ± 0.02) | 0.0 | 0.657 |
| counting | fixed 1/step | 256 | chain | 0.08 (+0.02 ± 0.02) | 0.08 (+0.01 ± 0.02) | 0.07 (-0.02 ± 0.02) | 0.0 | 0.690 |
| counting | lex B=0.03 +pull | 17 | product | 0.08 (+0.02 ± 0.02) | 0.08 (-0.02 ± 0.02) | 0.14 (-0.08 ± 0.03) | 0.0 | 0.602 |
| counting | lex B=0.03 +pull | 17 | chain | 0.13 (+0.03 ± 0.04) | 0.10 (+0.00 ± 0.03) | 0.11 (-0.01 ± 0.03) | 0.0 | 0.595 |
| dyck | fixed 1/step | 256 | product | 0.21 (-0.00 ± 0.06) | 0.42 (-0.22 ± 0.11) | 0.95 (+0.26 ± 0.28) | 7.7 | 0.171 |
| dyck | fixed 1/step | 256 | chain | 0.18 (+0.01 ± 0.06) | 0.23 (+0.07 ± 0.07) | 0.14 (+0.03 ± 0.04) | 2.8 | 0.267 |
| dyck | lex B=0.03 +pull | 16 | product | 0.20 (-0.11 ± 0.05) | 0.23 (-0.09 ± 0.06) | 0.26 (-0.04 ± 0.08) | 3.6 | 0.298 |
| dyck | lex B=0.03 +pull | 16 | chain | 0.16 (-0.02 ± 0.05) | 0.18 (+0.01 ± 0.06) | 0.23 (-0.03 ± 0.07) | 1.3 | 0.333 |

Against the criterion:
- Pressure exists on forbidden only. Without resampling its effective sample size is 2 to 6% and the log error is
  2.1 to 3.6 nats at 256 calls. Counting is not a pressure case at this length (resampling almost never fires) and
  Dyck is mild (0.18 to 0.48 nats).
- Forbidden, one token per step: chain twist with defensive resampling has the lowest log error at both particle
  counts, 1.39 against 3.32 and 3.59 without resampling (16 particles) and 0.66 against 2.13 and 2.38 (64 particles).
  The product twist also gains there (2.18 and 1.35 with defensive resampling), so the chain's margin over the product
  is 1.6 to 2.0 times with defensive resampling. With plain resampling the two twists are level at 64 particles
  (1.05 and 1.00).
- Dyck: resampling under the product twist is harmful (0.68 and 0.88 against 0.44 to 0.48 at 16 particles; 0.42 and
  0.95 against 0.18 to 0.21 at 64). Under the chain twist it is neutral (0.39 and 0.42; 0.23 and 0.14).
- So the criterion is met in part: a pressure case exists, chain-twist resampling gives a 2.4 to 3.6-fold gain in it,
  and product-twist resampling is unsafe on Dyck. It is not met in the stronger sense that only the chain twist
  gains: on forbidden the product twist gains as well.
- The constraint-first parallel schedule at 26 calls has a lower log error on forbidden than the random order at 256
  calls even without resampling (1.35 to 1.39 against 3.32 to 3.59 at 16 particles). Its reference p(C) is slightly
  off because a budgeted schedule is not exact.
- Limits: one chain, 12 runs at 64 particles, and no arm with the chain twist in the proposal.

## 36l. Where the schedule's own time goes over a whole decode (local CPU, 2026-11-16) [verified on oracle cells; GPU split in section 36j]

`experiments/parallel_unmasking/reveal_stages.py`; file `results/parallel_unmasking/reveal_stages_local.json`.
`hidden_0.85`, 16 particles, pilot 2,000, constraint-first order, the study's arm settings (B is the whole-sequence
budget). Each policy is run from the fully masked state to completion and every reveal is timed by stage. The denoiser
is not timed: these are the schedule's own costs. Times are milliseconds per decode of the batch on 2 CPU threads,
lower is better. `steps` counts reveal steps until the last particle finishes, so it is a little above the per-particle
call counts of the studies.

| N | constraint | arm | steps | positions per step | pull-in passes per step | context | scoring | admission loop | pull-in | total | per step | pull-in share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | window | sequential lex | 64.0 | 1.00 | 0.00 | 115 | 146 | 17 | 0 | 279 | 4.4 | 0% |
| 64 | window | lex B=0.03 | 13.0 | 5.04 | 0.00 | 20 | 29 | 63 | 0 | 112 | 8.6 | 0% |
| 64 | window | lex B=0.03 +pull | 13.0 | 5.04 | 6.05 | 22 | 30 | 77 | 114 | 244 | 18.7 | 47% |
| 64 | window | lex B=0.3 | 8.3 | 7.82 | 0.00 | 14 | 19 | 43 | 0 | 76 | 9.1 | 0% |
| 64 | window | lex B=0.3 +pull | 8.3 | 7.82 | 5.68 | 14 | 19 | 50 | 73 | 156 | 18.8 | 47% |
| 64 | counting | sequential lex | 64.0 | 1.00 | 0.00 | 642 | 149 | 18 | 0 | 808 | 12.6 | 0% |
| 64 | counting | lex B=0.03 | 14.7 | 4.48 | 0.00 | 143 | 34 | 71 | 0 | 248 | 16.9 | 0% |
| 64 | counting | lex B=0.03 +pull | 14.7 | 4.48 | 6.57 | 145 | 34 | 85 | 857 | 1121 | 76.5 | 76% |
| 64 | counting | lex B=0.3 | 9.7 | 6.80 | 0.00 | 94 | 23 | 49 | 0 | 166 | 17.2 | 0% |
| 64 | counting | lex B=0.3 +pull | 9.7 | 6.80 | 6.48 | 97 | 23 | 59 | 583 | 761 | 78.8 | 77% |
| 64 | forbidden | sequential lex | 64.0 | 1.00 | 0.00 | 560 | 149 | 17 | 0 | 725 | 11.3 | 0% |
| 64 | forbidden | lex B=0.03 | 24.7 | 2.64 | 0.00 | 217 | 58 | 99 | 0 | 374 | 15.2 | 0% |
| 64 | forbidden | lex B=0.03 +pull | 24.7 | 2.64 | 5.73 | 222 | 60 | 127 | 1207 | 1616 | 65.5 | 75% |
| 64 | forbidden | lex B=0.3 | 14.0 | 4.66 | 0.00 | 119 | 32 | 61 | 0 | 213 | 15.2 | 0% |
| 64 | forbidden | lex B=0.3 +pull | 14.0 | 4.66 | 5.90 | 121 | 33 | 75 | 707 | 935 | 66.8 | 76% |
| 256 | window | sequential lex | 256.0 | 1.00 | 0.00 | 1243 | 2217 | 168 | 0 | 3628 | 14.2 | 0% |
| 256 | window | lex B=0.03 | 18.5 | 14.04 | 0.00 | 72 | 137 | 300 | 0 | 509 | 27.5 | 0% |
| 256 | window | lex B=0.03 +pull | 18.5 | 14.04 | 5.73 | 72 | 136 | 333 | 345 | 885 | 47.8 | 39% |
| 256 | counting | sequential lex | 256.0 | 1.00 | 0.00 | 20554 | 1975 | 144 | 0 | 22674 | 88.6 | 0% |
| 256 | counting | lex B=0.03 | 19.0 | 13.60 | 0.00 | 1586 | 156 | 353 | 0 | 2095 | 110.2 | 0% |
| 256 | counting | lex B=0.03 +pull | 19.0 | 13.60 | 7.13 | 1464 | 143 | 372 | 6491 | 8470 | 445.8 | 77% |

- A sequential step is one constrained-marginals pass (context) plus one kernel sum (scoring).
- A parallel step without pull-in costs 1.2 to 2.1 times a sequential step; the extra is the admission loop, which
  visits every position. Over a whole decode its schedule time is 1.9 to 10.8 times lower than sequential, because
  there are fewer steps.
- The pull-in runs 5.7 to 7.1 extra constrained-marginals passes per step and takes 39 to 77% of the schedule's
  time. With it, a decode's schedule time is 1.7 to 4.6 times that without it.
- On these constraints (window, counting, forbidden) the pull-in admitted nothing: in all 8 pairs the number of steps
  and the positions per step are identical with and without it. The pull-in was built for constraints with forced
  partners (the block copy, section 31), where it lowered the spread from 0.40 to 0.12. Here there are none to find.
- With the pull-in, the parallel schedule's own time exceeds the sequential schedule's at 64 positions on counting and
  forbidden, and is below it at 256.
- This explains the GPU measurement of section 36j in kind (the parallel reveal is several times dearer per step and
  the candidate was the pull-in). The GPU run used a looser budget (rate 1/N, i.e. B = 1) and a small pilot, so its
  ratios are not the same quantity. The per-stage GPU rerun (end of section 36j) agrees in kind: the pull-in dominates and admits nothing.
- Not done: a pull-in that skips constraints or steps with nothing to pull. That is a design choice for the selection
  rule.

## 36m. How biased are the budgeted schedules against the one-token-per-step law? (2026-11-16) [verified at N = 64]

**Decision (user, 2026-11-16).** The target for parallel schedules is the law of the one-token-per-step sampler
conditioned on the constraint, taken as the denoiser's underlying joint. On a real denoiser, which is not
order-consistent, the reference is the average over uniformly random orders (each particle draws its own order);
a constraint-aware order is then an approximation whose error is the model's order inconsistency (zero on oracle
models; to be measured on MDLM). A budgeted parallel schedule is then an
approximation, its bias counts as error, and the budget is the tolerance.

From the order study (section 36i), resampling off. The sampler is exact for the model run under the same schedule,
so the mean of the estimate of p(C) differs from the exact p(C) only through the dependence among positions revealed
together. Pooled over 3 chains and 7 orders (21 arms each), inverse-variance weighted; 0 is unbiased.

| budget | constraint | mean calls | log(estimate / p(C)), nats | in s.e. | as a multiple of the budget | arms beyond 2 s.e. |
|---|---|---|---|---|---|---|
| none (sequential) | counting | 64.0 | -0.003 ± 0.003 | -1.1 |  | 0% |
| none (sequential) | dyck | 64.0 | -0.002 ± 0.004 | -0.4 |  | 5% |
| none (sequential) | forbidden | 64.0 | +0.006 ± 0.007 | +0.9 |  | 0% |
| none (sequential) | window | 64.0 | +0.001 ± 0.002 | +0.7 |  | 10% |
| none (sequential) | two_window | 64.0 | -0.005 ± 0.003 | -1.8 |  | 5% |
| B = 0.03 | counting | 13.6 | +0.016 ± 0.004 | +4.4 | 0.5 | 14% |
| B = 0.03 | dyck | 13.2 | -0.005 ± 0.005 | -1.0 | 0.2 | 14% |
| B = 0.03 | forbidden | 16.4 | -0.053 ± 0.009 | -6.2 | 1.8 | 24% |
| B = 0.03 | window | 14.3 | -0.005 ± 0.002 | -2.6 | 0.2 | 14% |
| B = 0.03 | two_window | 14.5 | -0.007 ± 0.002 | -3.0 | 0.2 | 5% |
| B = 0.3 | counting | 8.5 | +0.066 ± 0.003 | +19.0 | 0.2 | 67% |
| B = 0.3 | dyck | 7.6 | -0.016 ± 0.005 | -3.1 | 0.1 | 76% |
| B = 0.3 | forbidden | 10.5 | -0.387 ± 0.011 | -36.3 | 1.3 | 100% |
| B = 0.3 | window | 8.6 | -0.068 ± 0.003 | -24.0 | 0.2 | 76% |
| B = 0.3 | two_window | 8.7 | -0.031 ± 0.003 | -11.9 | 0.1 | 52% |

- The sequential arms are unbiased within noise, as they must be.
- At B = 0.03 the bias is at most 1.6% on four constraints and 5.3% on forbidden, at 13 to 16 calls instead of 64.
- At B = 0.3 it is 2 to 7% on four constraints and 0.39 nats on forbidden (the estimate is 0.68 of p(C)), at 8 to 10 calls.
- In nats the bias is 0.1 to 0.5 of the budget, except on forbidden, where it is 1.3 to 1.8 times the budget.
- Pooling hides biases of opposite sign between orders: on Dyck at B = 0.3 individual orders are off by +14% and -14%
  (section 36i), which pools to -1.6%.
- Why forbidden exceeds its budget is examined below: the conditioned sequences are unusually regular, so they
  realise several times the model's average dependence.

**Where the bias comes from (added the same day).** Each run records, per particle, the gap r = log of the true
joint over the schedule's law. Since the sampler is exact for the schedule's law, the bias is
log(estimate / p(C)) = -log E[exp r] over the weighted particles, which is about -(mean of r + half its variance).
Pooled over chains and orders:

| budget | constraint | mean gap | half variance | predicted log bias | measured |
|---|---|---|---|---|---|
| B = 0.03 | counting | -0.036 | 0.017 | +0.020 | +0.012 |
| B = 0.03 | dyck | -0.008 | 0.016 | -0.009 | -0.008 |
| B = 0.03 | forbidden | +0.048 | 0.006 | -0.054 | -0.051 |
| B = 0.03 | window | -0.005 | 0.013 | -0.008 | -0.004 |
| B = 0.03 | two_window | -0.007 | 0.014 | -0.007 | -0.005 |
| B = 0.3 | counting | -0.270 | 0.204 | +0.066 | +0.055 |
| B = 0.3 | dyck | -0.181 | 0.202 | -0.020 | -0.009 |
| B = 0.3 | forbidden | +0.275 | 0.092 | -0.367 | -0.381 |
| B = 0.3 | window | -0.100 | 0.170 | -0.070 | -0.071 |
| B = 0.3 | two_window | -0.157 | 0.192 | -0.035 | -0.035 |

- The identity closes to about 0.01 nats on the pooled values (per arm: median 0.006 to 0.065, largest 0.16).
- With no constraint, mean and half-variance cancel exactly. On four constraints they still nearly cancel. On
  forbidden the mean is positive: sequences that satisfy the constraint are more probable under the true joint than
  under the schedule's law, so the parallel sampler under-produces them. The excess is a shift of the mean, not spread.
- Constraint-induced linkage among co-revealed positions cannot bias the estimate by itself [derived]: with an
  independent model the schedule's law equals the joint for any schedule. It costs weight variance and dead particles.

**Motif test (local, `hidden_0.85`, constraint-first order, 100 runs x 16 particles, one arm per cell;
`results/parallel_unmasking/motif_N64.json`).** The standard forbidden cell bans two motifs of length 2. Variants:
the same kind of motif at lengths 3 and 4, and each length-2 motif alone.

| motifs banned | log 1/p(C) | calls, B = 0.3 | log bias, B = 0.3 | mean gap | half variance | shift = mean + half variance | log bias, B = 0.03 | sequential (should be 0) |
|---|---|---|---|---|---|---|---|---|
| 00, 12 | 10.54 | 12.9 | -0.294 ± 0.055 | +0.265 | 0.076 | +0.341 | -0.071 ± 0.032 | -0.001 ± 0.031 |
| 000, 121 | 3.67 | 11.0 | -0.150 ± 0.051 | +0.014 | 0.176 | +0.190 | -0.009 ± 0.041 | +0.089 ± 0.041 |
| 0000, 1212 | 1.95 | 10.8 | -0.110 ± 0.030 | -0.135 | 0.191 | +0.056 | -0.025 ± 0.026 | +0.072 ± 0.031 |
| 00 | 3.12 | 8.4 | -0.254 ± 0.028 | +0.072 | 0.170 | +0.242 | -0.024 ± 0.019 | +0.014 ± 0.021 |
| 12 | 4.25 | 10.2 | -0.161 ± 0.019 | -0.033 | 0.203 | +0.170 | +0.004 ± 0.022 | +0.028 ± 0.021 |

- The bias at B = 0.3 falls with motif length: -0.29, -0.15, -0.11. Only length 2 against length 4 is resolved
  (2.9 standard errors).
- Rarity is not the explanation: banning the repeat "00" alone is as biased as the standard cell (-0.25 against
  -0.29, not resolved apart) although it is far less rare (3.1 against 10.5 nats). It is more biased than banning the
  switch "12" alone (2.7 standard errors).
- The length trend does not separate explanations: a longer motif links more positions but less hard (user), and it
  is also less rare. Rarity alone is ruled out by the repeat motif. The regularity measurement below gives the
  mechanism for the standard cell.
- Caution: two of the five sequential controls sit 2.2 and 2.3 standard errors from zero, where they must be zero, so
  these standard errors are somewhat too small.

**Are the conditioned sequences unusually regular? Yes** (`experiments/parallel_unmasking/regularity_check.py`,
`results/parallel_unmasking/regularity_N64.json`; 40,000 exact draws from p(. | C) and from the model, `hidden_0.85`,
64 positions, 3 symbols). "Model dependence realised" is the model's pointwise log dependence
log p(a, b) / (p(a) p(b)) for a pair of tokens at the given distance, averaged over the conditioned sequences and
divided by its average under the model (the mutual information: 0.0601, 0.0168, 0.0013 nats at distances 4, 8, 16).

| motifs banned | log 1/p(C) | allowed set, entropy as a share of log 3 | symbol frequencies | adjacent agreement | mean run length | agreement at distance 8 | at 16 | model dependence realised, as a multiple of its average, at distance 4 | at 8 | at 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| none (the model) | 0.00 | 1.000 | 0.30, 0.30, 0.40 | 0.612 | 2.51 | 0.426 | 0.363 | 1.0 | 1.0 | 1.0 |
| 00, 12 | 10.54 | 0.742 | 0.12, 0.32, 0.57 | 0.743 | 3.72 | 0.628 | 0.544 | 2.8 | 4.8 | 13.3 |
| 000, 121 | 3.67 | 0.939 | 0.13, 0.26, 0.61 | 0.643 | 2.72 | 0.515 | 0.475 | 1.3 | 2.1 | 6.9 |
| 0000, 1212 | 1.95 | 0.982 | 0.15, 0.36, 0.49 | 0.616 | 2.54 | 0.467 | 0.415 | 1.1 | 1.6 | 4.0 |
| 00 | 3.12 | 0.916 | 0.09, 0.39, 0.52 | 0.629 | 2.62 | 0.517 | 0.462 | 1.6 | 2.6 | 7.5 |
| 12 | 4.25 | 0.878 | 0.53, 0.20, 0.28 | 0.670 | 2.94 | 0.486 | 0.420 | 1.9 | 2.8 | 7.7 |

- With 2 symbols and one switch banned, only n + 1 = 65 of 2^64 sequences are allowed, all of them two runs. With 3
  symbols and 2 of 9 pairs banned the allowed set is still exponentially large (entropy 0.74 of log 3 per position).
- The conditioned law is nevertheless far more regular than the model, because the model's weights concentrate it on
  the run-like part of the allowed set: symbol 0 falls from 30% to 12%, adjacent agreement rises from 0.61 to 0.74
  although one repeat is banned, the mean run lengthens from 2.5 to 3.7, and tokens 16 apart agree 54% of the time
  against 36%.
- On those sequences the model's dependence between tokens 4 to 16 apart is 2.8 to 13.3 times its average. The
  schedule's kernel is that average, so its budget under-counts the dependence among positions revealed together.
  This is the mean shift of the gap seen above.
- The variants order the same way for motif length (standard cell most, lengths 3 and 4 least), matching their biases.
  The two single motifs do not: the switch alone shows more excess than the repeat alone but less bias. So this
  context-free measure (pairs with nothing revealed around them) is not the whole account.
- Correction: an earlier version of this section said the "unusually regular sequences" explanation was not
  supported and that banning a repeat does not make sequences more run-like. Neither had been measured; both are
  contradicted here (banning "00" alone raises agreement at distance 8 from 0.43 to 0.52).

**Decision (user, 2026-11-16).** Treat this as a known failure mode and do not pursue it now: a 3-symbol alphabet with
length-2 motifs that oppose the model's agreement pressure is not a realistic setting; realistic motifs are longer and
alphabets larger. Warning sign to keep: a constraint that opposes the model's strongest local dependence, so that
conditioning makes sequences much more regular than the model's own samples. Not built: a kernel measured on
constrained samples, or a budget rescaled by the constraint's effect on long-range agreement.

Relative RMS error of the estimate of p(C) at equal total denoiser calls, constraint-first order, geometric mean over
chains; lower is better. Error squared = bias squared (noise-corrected) + spread squared scaled to the number of
particles the budget buys. First three columns: the calls that 16 sequential particles cost; last three: 16 times that.

| constraint | sequential | B = 0.03 | B = 0.3 | sequential, 16x calls | B = 0.03, 16x | B = 0.3, 16x |
|---|---|---|---|---|---|---|
| counting | 0.269 | 0.085 | 0.091 | 0.067 | 0.037 | 0.055 |
| dyck | 0.231 | 0.110 | 0.158 | 0.058 | 0.030 | 0.117 |
| forbidden | 0.272 | 0.265 | 0.406 | 0.068 | 0.087 | 0.325 |
| window | 0.089 | 0.043 | 0.109 | 0.022 | 0.014 | 0.097 |
| two_window | 0.090 | 0.037 | 0.051 | 0.022 | 0.014 | 0.037 |

- B = 0.03 is at or below the sequential error on four of five constraints at both budgets; on forbidden it is level
  at the smaller budget and worse at the larger.
- B = 0.3 is worse than sequential at the larger budget on four of five constraints (counting is the exception).
- This is the estimate of p(C) only. Sample quality needs enough particles that noise sits below the bias; not run.
- Consequence for reporting: `cost` (calls x spread squared) ignores this bias and must not be used alone to rank
  budgeted arms.

## 36n. Selection as planning on pair costs: formulation and first check (2026-11-16) [formulation derived; check verified at N = 64]

**Objective (user and assistant, 2026-11-16).** Maximum accuracy at a fixed compute budget. With K particles and T
steps the compute is K T, and the variance of the estimate is about (exp(kappa E) - 1) / K, where E is the KL
divergence of one particle's path from the target and kappa is about 2.4 here (section 36i). At a fixed budget the
quantity to minimise is therefore T (exp(kappa E) - 1), which for small E is T x E: steps times total error. No price
per step is tuned. The exchange rate is implied: saving a step is worth an extra error of up to E / T, the current
average error per step (for large E, about 1 / (kappa T)). Since T is the sum over reveals of 1 / (positions revealed
in that step), the compute can be charged per reveal as its share of a call; the errors themselves add and are not
divided. The co-reveal bias is not reduced by more particles, so it is budgeted against the target accuracy and is
the one place the operating point enters. [derived; untested]

**Total error is additive** (section 36b; rechecked in 36i): for one token per step, KL(sampler || target) is the
expected sum of the per-step errors, for any order that depends only on what is revealed. In `_propose`
(`twistroute/decode/twist.py`) the classes of all positions revealed in a step are read from one joint draw of the
route, so the constraint's own coupling within a step is handled by the sampler, and the reveal set is chosen without
seeing that draw.

**Pair picture [derived at leading order].** Revealing i while j is masked costs
c(i before j) = 1/2 Var over the class a at i of E[rho_j(x_j) | x_i = a], with rho_j the constrained over the model
marginal at j. It is zero if j has no stake or is independent of i. The two mechanisms are two properties of c:
asymmetry (an unconstrained position revealed before a constrained one it depends on pays; the reverse is free), and
screening (c shrinks when something between the pair is revealed, which is a separator's value). A forced position
has zero cost against every partner. Pairs revealed together pay the model's dependence, which is the bias.

**First check** (`experiments/parallel_unmasking/pair_cost_check.py`, `results/parallel_unmasking/pair_cost_N64.json`):
`hidden_0.85`, 64 positions, 60 exact constrained states at four reveal fractions, every unforced masked position
(about 9,000 per constraint). Unlike section 36g, positions are not restricted to those carrying a stake and their
immediate neighbours. `pair` is the sum of c over masked j. `lookahead` re-evaluates the product twist at the
successor state. `exact update` takes the model's marginals after the reveal from the model (tests the derivation);
`kernel update` multiplies the current marginals by the model's unconditional pair ratio at that distance, with no
screening by revealed tokens (a first cut at what a pilot kernel would give).

Rank correlation with the true cost within a state, 1 is the same ranking:

| constraint | pair, exact update | lookahead, exact update | pair, kernel update | chain twist (section 36g) |
|---|---|---|---|---|
| counting | 0.21 | 0.22 | 0.23 | 0.06 |
| dyck | 0.42 | 0.37 | 0.38 | 0.46 |
| forbidden | 0.83 | 0.94 | 0.62 | 0.80 |
| window | 0.92 | 0.98 | 0.88 | 0.28 |
| two_window | 0.86 | 0.93 | 0.79 | 0.27 |

Median of estimate over true cost on the costlier half of positions, 1 is the right size:

| constraint | pair, exact update | lookahead, exact update | pair, kernel update | chain twist |
|---|---|---|---|---|
| counting | 1.31 | 12.48 | 2.94 | 0.45 |
| dyck | 0.40 | 2.94 | 1.03 | 0.23 |
| forbidden | 0.28 | 2.74 | 1.80 | 1.88 |
| window | 0.72 | 3.69 | 6753.48 | 0.72 |
| two_window | 0.75 | 2.09 | 110.48 | 0.20 |

True cost at the position each rule would reveal next, nats x 1000, lower is better:

| constraint | ideal | pair, exact update | lookahead, exact update | pair, kernel update | chain twist | lowest entropy | random |
|---|---|---|---|---|---|---|---|
| counting | 0.10 | 4.40 | 4.62 | 3.85 | 3.08 | 2.72 | 3.28 |
| dyck | 0.06 | 10.03 | 15.39 | 10.60 | 5.92 | 9.26 | 16.15 |
| forbidden | 0.90 | 2.01 | 2.28 | 6.35 | 6.04 | 4.55 | 36.24 |
| window | 0.00 | 0.00 | 0.00 | 0.00 | 0.26 | 3.81 | 8.28 |
| two_window | 0.12 | 0.56 | 0.45 | 0.66 | 3.02 | 5.26 | 15.22 |

Windows by distance from the window (inside, within 6, farther), mean cost in nats x 1000:

| constraint | revealed | true: inside, near, far | pair, exact update | pair, kernel update | chain twist |
|---|---|---|---|---|---|
| window | 10% | 20.2, 68.7, 2.2 | 155.2, 65.9, 2.0 | 192.3, 94.4, 4.2 | 29.5, 9.1, 1.4 |
| window | 25% | 14.1, 27.2, 0.4 | 68.4, 22.7, 0.3 | 108.3, 47.7, 2.0 | 19.9, 6.2, 1.8 |
| window | 50% | 5.8, 3.9, 0.0 | 6.9, 2.8, 0.0 | 13.1, 9.9, 0.4 | 12.3, 3.5, 1.3 |
| window | 75% | 1.3, 1.0, 0.0 | 0.4, 1.1, 0.0 | 1.6, 6.4, 0.3 | 4.7, 0.8, 0.6 |
| two_window | 10% | 21.9, 62.8, 6.1 | 92.8, 49.8, 4.1 | 107.4, 63.9, 7.5 | 20.5, 9.4, 1.4 |
| two_window | 25% | 19.6, 34.8, 1.7 | 46.0, 23.8, 1.0 | 67.9, 44.8, 5.1 | 22.2, 7.8, 1.8 |
| two_window | 50% | 10.1, 7.4, 0.2 | 11.0, 5.2, 0.1 | 21.1, 13.4, 1.7 | 13.3, 4.0, 1.2 |
| two_window | 75% | 2.3, 0.6, 0.0 | 2.5, 0.5, 0.0 | 22.0, 5.1, 0.8 | 10.3, 0.9, 0.5 |

Median of the kernel-update pair estimate over the true cost, by fraction revealed:

| constraint | 10% revealed | 25% | 50% | 75% |
|---|---|---|---|---|
| counting | 3.88 | 2.69 | 2.03 | 2.23 |
| dyck | 0.60 | 0.97 | 1.57 | 1.71 |
| forbidden | 0.26 | 0.68 | 2.00 | 10.77 |
| window | 1.86 | 6.89 | 36.98 | 68996.92 |
| two_window | 1.02 | 1.92 | 9.13 | 1095.65 |

- The asymmetric cost is predicted well: for unconstrained positions near a window the pair estimate with the exact
  update is 0.68 to 1.06 of the true mean across the four fractions and both window constraints. The chain twist's
  estimate there is 4 to 8 times too small at 10 to 25% revealed.
- For positions inside the window the pair estimate is 2 to 8 times too large at 10 to 25% revealed and about right
  at 50%. It uses the
  constraint ratio at j before i's own class is fixed; the ratio conditional on that class is the correction to try.
- Ranking within a state is good on the local constraints (0.83 to 0.98) and poor on the global ones (0.2 to 0.4), as
  expected of a pairwise account. On counting its next pick is worse than a random one; on Dyck it is better than random but worse than the chain
  twist's.
- The lookahead estimate ranks slightly better but is 2 to 12 times too large: re-evaluating a mean-field twist after
  updating every marginal counts shared dependence several times.
- The size is off by a constraint-dependent factor (0.28 on forbidden, 1.3 on counting), so sums over a plan are not
  yet comparable across constraints.
- The kernel update without screening is usable when little is revealed and fails progressively after that. A kernel
  that conditions on what lies between is needed before this can run without the model's exact update.
- Open: on counting the ideal pick is about 30 times cheaper than a random one and no estimator finds it.

**Two fixes and a diagnostic (later the same day; `results/parallel_unmasking/pair_cost2_N64.json`).** The constraint
ratio at j is now also taken after position i is fixed at the candidate class ("conditioned ratio"; one route pass per
position and class), and the kernel update is screened: the log pair ratio is scaled by
sqrt(kernel given what is revealed in between / unscreened kernel), from the pilot kernel.

Rank correlation with the true cost within a state:

| constraint | pair, exact update | pair with conditioned ratio, exact update | pair with conditioned ratio, screened kernel | lookahead, exact update | chain twist |
|---|---|---|---|---|---|
| counting | 0.21 | 0.20 | 0.18 | 0.22 | 0.06 |
| dyck | 0.42 | 0.43 | 0.40 | 0.37 | 0.46 |
| forbidden | 0.83 | 0.95 | 0.90 | 0.94 | 0.80 |
| window | 0.92 | 0.92 | 0.89 | 0.98 | 0.28 |
| two_window | 0.86 | 0.86 | 0.81 | 0.93 | 0.27 |

Median of estimate over true cost on the costlier half of positions:

| constraint | pair, exact update | pair with conditioned ratio, exact update | pair with conditioned ratio, screened kernel | lookahead, exact update | chain twist |
|---|---|---|---|---|---|
| counting | 1.31 | 1.30 | 1.90 | 12.48 | 0.45 |
| dyck | 0.40 | 0.44 | 0.66 | 2.94 | 0.23 |
| forbidden | 0.28 | 0.75 | 2.08 | 2.74 | 1.88 |
| window | 0.72 | 0.77 | 12.50 | 3.69 | 0.72 |
| two_window | 0.75 | 0.77 | 3.23 | 2.09 | 0.20 |

True cost at the next pick, nats x 1000:

| constraint | ideal | pair, exact update | conditioned ratio, exact update | conditioned ratio, screened kernel | chain twist | lowest entropy | random |
|---|---|---|---|---|---|---|---|
| counting | 0.10 | 4.40 | 4.54 | 4.06 | 3.08 | 2.72 | 3.28 |
| dyck | 0.06 | 10.03 | 11.09 | 11.37 | 5.92 | 9.26 | 16.15 |
| forbidden | 0.90 | 2.01 | 2.56 | 2.48 | 6.04 | 4.55 | 36.24 |
| window | 0.00 | 0.00 | 0.00 | 0.00 | 0.26 | 3.81 | 8.28 |
| two_window | 0.12 | 0.56 | 0.56 | 0.59 | 3.02 | 5.26 | 15.22 |

Windows by distance from the window (inside, within 6, farther), mean cost in nats x 1000:

| constraint | revealed | true: inside, near, far | conditioned ratio, exact update | conditioned ratio, screened kernel |
|---|---|---|---|---|
| window | 10% | 20.2, 68.7, 2.2 | 164.7, 65.9, 2.0 | 192.7, 75.8, 2.3 |
| window | 25% | 14.1, 27.2, 0.4 | 74.1, 22.7, 0.3 | 104.7, 30.5, 0.3 |
| window | 50% | 5.8, 3.9, 0.0 | 9.7, 2.8, 0.0 | 13.4, 4.4, 0.0 |
| window | 75% | 1.3, 1.0, 0.0 | 0.9, 1.1, 0.0 | 1.6, 1.2, 0.0 |
| two_window | 10% | 21.9, 62.8, 6.1 | 93.7, 49.8, 4.1 | 107.1, 56.3, 4.4 |
| two_window | 25% | 19.6, 34.8, 1.7 | 47.2, 23.8, 1.0 | 65.3, 30.1, 1.4 |
| two_window | 50% | 10.1, 7.4, 0.2 | 10.7, 5.2, 0.1 | 17.3, 6.2, 0.1 |
| two_window | 75% | 2.3, 0.6, 0.0 | 2.3, 0.5, 0.0 | 14.5, 0.9, 0.0 |

- The conditioned ratio fixes forbidden: rank 0.83 to 0.95, size 0.28 to 0.75, tenfold underestimates 11% to 0%.
- It does not fix the overestimate inside the window, so that explanation was wrong (see the collective term below).
- With the screened pilot kernel and no denoiser call, mean costs near the window are 0.84 to 1.13 of the truth at
  10 to 50% revealed (up to 1.6 at 75%, where the costs are about 1e-3 nats). The next pick costs 0.00 on the window
  (ideal 0.00), 0.59 on two windows (ideal 0.12, random 15.2) and 2.48 on forbidden (ideal 0.90, random 36.2).
  The median ratio is still large late in a decode, where the true costs are below 1e-3 nats.
- Counting and Dyck are unchanged.
- Is the cost a common over- or under-reaction of the twist? Fitted slope of the true log tilt on the product twist's
  (1 = the twist's tilt is right), median [quartiles]; interquartile range within a state:

| constraint | 10% revealed | 25% | 50% | 75% |
|---|---|---|---|---|
| counting | 0.97 [0.62, 1.34]; 0.67 | 0.90 [0.55, 1.33]; 0.69 | 0.86 [0.66, 1.22]; 0.49 | 0.95 [0.83, 1.14]; 0.28 |
| dyck | 1.65 [0.43, 2.40]; 1.68 | 1.23 [0.44, 1.82]; 1.30 | 1.01 [0.73, 1.44]; 0.62 | 0.98 [0.82, 1.19]; 0.29 |
| forbidden | 1.45 [1.04, 2.08]; 1.02 | 1.17 [0.75, 1.51]; 0.75 | 1.03 [0.39, 1.14]; 0.69 | 1.01 [0.49, 1.08]; 0.84 |
| window | 1.30 [1.16, 1.39]; 0.21 | 1.25 [1.09, 1.41]; 0.22 | 1.24 [1.08, 1.42]; 0.22 | 1.08 [1.01, 1.21]; 0.20 |
| two_window | 1.20 [1.01, 1.41]; 0.36 | 1.20 [0.91, 1.44]; 0.52 | 1.17 [0.97, 1.40]; 0.32 | 1.05 [1.00, 1.17]; 0.12 |

  On the windows the factor is consistent (1.05 to 1.3 with a narrow range: the twist under-reacts). On counting
  it varies widely between positions of one state, so no single factor explains the cost there.

## 36o. The missing term on a sum constraint: mean shift and spread together (2026-11-16) [verified on counting at N = 64]

`experiments/parallel_unmasking/collective_check.py`, `results/parallel_unmasking/collective_N64.json`; counting
(exactly k of 64 positions are class 0), `hidden_0.85`, 60 states, exact one-step updates.

**Derivation.** Revealing i as class a leaves the other masked positions to supply k_a zeros. The exact conditional is
pi_i(a) proportional to p_i(a) P(S = k_a | x_i = a), S the number of zeros among the others. The product twist uses an
independent sum at the current marginals, which is wrong in two ways with opposite signs: the mean (a zero at i raises
the others' expected count, by position i's influence) and the spread (the true count is more dispersed; the ratio of
variances, VIF, minus 1 is the variance-weighted average influence over positions). To leading order the error in the
log odds at i is proportional to (influence of i - average influence): positions with typical coupling are cheap, and
both weakly and strongly coupled ones are dear. `both` predicts pi_i with a normal count using the shifted mean and
the inflated variance; `closed form` is 1/2 q_i(0)(1 - q_i(0)) (slope x (influence_i - (VIF - 1)))^2.

| revealed | spread ratio (VIF) | influence, median [10%, 90%] | rank: mean only | spread only | both | closed form | size, both | next pick x1000: ideal | both | closed form | random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10% | 4.21 | 3.10 [1.35, 4.56] | 0.17 | 0.09 | 0.23 | 0.35 | 0.61 | 0.05 | 2.18 | 1.03 | 2.06 |
| 25% | 2.74 | 1.55 [0.41, 3.11] | 0.32 | 0.22 | 0.57 | 0.43 | 0.61 | 0.21 | 1.34 | 1.26 | 4.59 |
| 50% | 1.64 | 0.48 [0.05, 1.44] | 0.32 | 0.24 | 0.68 | 0.51 | 0.72 | 0.17 | 0.65 | 0.70 | 2.78 |
| 75% | 1.18 | 0.10 [0.00, 0.48] | 0.32 | 0.25 | 0.45 | 0.53 | 0.70 | 0.15 | 0.81 | 0.60 | 2.59 |

- The independent sum understates the count's variance by a factor of 4.2 when 10% is revealed and 1.2 at 75%.
- Neither error alone ranks positions (0.09 to 0.32, like the pair account's 0.21); together they reach 0.57 to 0.68
  in the middle of a decode. Early (10% revealed) the account is still weak.
- The mean shift alone is 23 times too large early: this is the overshoot of the lookahead estimate in section 36n,
  and of the pair estimate inside a window.
- The next pick under the closed form is 2 to 4 times cheaper than a random one at every fraction, and still 4 to 20
  times dearer than the ideal.
- Reading for the two mechanisms: on a sum constraint the dominant saving from revealing well-connected positions
  early is that it lowers the spread ratio for every later reveal. That fits the order study (section 36i), where
  `sum` and `separator` were the best orders on counting and lowest entropy the worst.
- General form this points to: the cost of revealing i is KL(q_i || the tilt under a twist that is right to second
  order, in mean and in spread, evaluated at the successor). The pair term is its mean part.
- Not done: the same with kernel inputs; Dyck (a running-sum constraint); a general second-order twist for automata.

## 36p. Order consistency of MDLM (cluster job `b1ffcca2`, one L40S on node3203, 2026-11-16) [measured; samples not inspected]

`experiments/parallel_unmasking/mdlm_order_consistency.py` at `cc24eb5`,
`results/parallel_unmasking/mdlm_order_consistency.json`. 32 sequences drawn from the model one token per step in a
random order; each scored under 16 further random orders, left-to-right and right-to-left. p_sigma(x) is the product
of the model's one-token conditionals along order sigma. All in nats per sequence, mean ± s.e. over sequences.

| tokens | -log p per token, other orders | sd of log p over random orders | gap to the random-order mixture | own order minus other orders | left-to-right minus random | right-to-left minus random |
|---|---|---|---|---|---|---|
| 32 | 3.16 | 9.7 ± 1.0 | 13.3 ± 1.5 | +38.1 ± 4.1 | +4.1 ± 1.4 | +1.3 ± 1.8 |
| 64 | 3.16 | 14.4 ± 1.4 | 19.3 ± 2.0 | +67.0 ± 6.8 | -0.1 ± 2.6 | -10.4 ± 3.1 |
| 128 | 3.74 | 16.2 ± 1.3 | 23.2 ± 1.9 | +102.1 ± 6.9 | +0.7 ± 3.7 | -19.9 ± 4.3 |

- Check of the script: on the exact hidden-chain model every quantity is zero (spread 1e-7 on the cluster).
- A sequence is 38, 67 and 102 nats more probable under the order that generated it than under another random
  order, about one nat per token. Averaged over samples, that difference is the KL divergence from the law of one
  order to the law of another. It is two orders of magnitude above the errors optimised elsewhere in this document.
- Among orders that did not generate the sequence, log p varies with a standard deviation of 10 to 16 nats.
- Left-to-right is level with a random order at 64 and 128 tokens and 4 nats higher at 32; right-to-left is lower by
  10 to 20 nats at 64 and 128 tokens.
- Caveats that could change the size: the sequences are unconditional samples at 32 to 128 tokens from a checkpoint
  trained at 1,024; they were not saved, so it is not known whether they are degenerate; the mean entropy along the
  generating order (about 2.1 nats per token) is low. No real text was scored.
- If it holds on realistic text, the choice of order matters far more on this model than any schedule error
  measured so far, and a constraint-aware order samples a law far from the random-order reference (section 36m).
- Next (needs a GPU): save the samples and per-step entropies, score real text and prompted samples, lengths to 1,024.

## 36q. Where along a decode each order pays: counting and Dyck (local, 2026-11-16) [verified at N = 64, one hidden chain]

Script `experiments/parallel_unmasking/counting_orders.py`; results `results/parallel_unmasking/order_study/{counting,dyck}_orders_N64.json`.
`hidden_0.85`, 64 positions, 300 particles, one token per step. KL is the divergence of the sampler's law from the exact
constrained law in nats (lower is better), as the sum over steps of the true error at the revealed position. The spread
ratio is Var(number of class-0 tokens still to come) / (the same variance if the masked positions were independent);
1 means the product twist's picture of the remaining count is right.

**counting** (log p(C) = -3.24)

| order | KL | paid in quarter 1 | 2 | 3 | 4 | spread ratio at step 0 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.180 | 0.041 | 0.063 | 0.052 | 0.025 | 5.85 | 3.81 | 2.77 | 1.71 | 1.24 |
| lex | 0.204 | 0.040 | 0.110 | 0.050 | 0.003 | 5.85 | 4.25 | 3.39 | 2.04 | 1.14 |
| separator | 0.136 | 0.043 | 0.065 | 0.026 | 0.002 | 5.85 | 3.36 | 2.29 | 1.37 | 1.05 |
| sum | 0.123 | 0.042 | 0.057 | 0.023 | 0.001 | 5.85 | 3.35 | 2.28 | 1.37 | 1.05 |
| cheapest | 0.191 | 0.003 | 0.023 | 0.054 | 0.111 | 5.85 | 4.11 | 3.47 | 2.75 | 2.09 |
| centred | 0.255 | 0.016 | 0.042 | 0.081 | 0.117 | 5.85 | 3.87 | 2.86 | 2.01 | 1.61 |

**dyck** (log p(C) = -5.82)

| order | KL | paid in quarter 1 | 2 | 3 | 4 | spread ratio at step 0 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.642 | 0.342 | 0.181 | 0.085 | 0.033 | 5.97 | 4.12 | 2.93 | 1.82 | 1.28 |
| lex | 0.586 | 0.218 | 0.146 | 0.129 | 0.094 | 5.97 | 5.52 | 5.17 | 4.04 | 2.12 |
| separator | 0.594 | 0.399 | 0.151 | 0.040 | 0.003 | 5.97 | 3.68 | 2.47 | 1.45 | 1.07 |
| sum | 0.599 | 0.403 | 0.152 | 0.040 | 0.004 | 5.97 | 3.74 | 2.56 | 1.55 | 1.12 |
| cheapest | 0.262 | 0.018 | 0.034 | 0.081 | 0.129 | 5.97 | 4.68 | 4.18 | 3.40 | 2.42 |

- `cheapest` is the ideal greedy order (lowest true error now, needs the exact twist); `centred` is greedy on the closed form of section 36o.
- Counting: `sum` and `separator` pay as much as random early and almost nothing in the last quarter, and they bring the
  spread ratio down fastest. The ideal greedy order pays almost nothing early and most of its error in the last quarter:
  it never lowers the spread ratio. On this constraint the whole gain is investing early.
- Dyck: the ideal greedy order has less than half the KL of every other order. `separator` and `sum` again shift the
  error to the first quarter, but the total is no lower than left-to-right (`lex`).
- So the two whole-sequence constraints disagree about greedy, and neither is explained by one fixed recipe.

## 36r. One step of planning on the present cost estimates (local, 2026-11-16) [verified at N = 32; 64 particles, noisy]

Script `experiments/parallel_unmasking/plan_check.py`; results `results/parallel_unmasking/order_study/plan_*_N32.json`.
An estimate c_j(x) of the cost of revealing j is used greedily (lowest c_i now) or with one step of planning (lowest
c_i(x) + E_a sum_j c_j(x with i = a)). Estimates: the closed form of section 36o on counting, the pair cost of section 36n
elsewhere; both use the model's exact one-step update. KL in nats, lower is better; 64 particles, so differences under
about 20% are not resolved.

| constraint | random | lex | separator | sum | ideal greedy | greedy on estimate | plan on estimate |
|---|---|---|---|---|---|---|---|
| counting | 0.200 | 0.188 | 0.115 | 0.103 | 0.124 | 0.134 | 0.159 |
| window | 0.410 | 0.078 | 0.373 | 0.095 | 0.044 | 0.675 | 0.077 |
| two_window | 0.801 | 0.090 | 0.582 | 0.115 | 0.229 | 1.286 | 0.146 |
| forbidden | 1.420 | 0.378 | 2.259 | 0.589 | 0.368 | 0.491 | 0.366 |

- Windows and forbidden: greedy use of the pair estimate is poor on the windows (9 and 14 times `lex`). One step of
  planning on the same estimate brings it level with `lex` on window and forbidden and to 1.6 times `lex` on two_window.
- Counting: planning on the closed form is no better than its greedy use, and both lose to `sum`.

## 36s. The ideal order, exactly (local, 2026-11-16) [verified against brute-force enumeration; 10 to 12 positions]

Script `experiments/parallel_unmasking/best_plan_exact.py`; results `results/parallel_unmasking/order_study/best_plan_*.json`.
At 10 positions with 3 symbols (12 with 2) every partially revealed state can be tabulated (about one million states).
With the true error eps_i(x) at every state, the best state-dependent order solves
V(x) = min_i [ eps_i(x) + sum_a q_i(a | x) V(x with i = a) ] in one backward pass, and the same pass gives the exact KL
of any other rule: no sampling noise. Check: the pass reproduces the KL of the left-to-right sampler computed by
enumerating all sequences, to six decimals, on counting, Dyck, window and forbidden at 6 to 8 positions.
`window_small`: at least 3 of the 4 middle positions are class 0; `two_window_small`: the same over two pairs.

Exact KL of the sampler from the constrained law, nats, lower is better.

| rule | counting, hidden_0.85, N=10 | counting, chain_0.85, N=10 | dyck, hidden_0.85, N=12 | dyck, chain_0.85, N=12 | window_small, hidden_0.85, N=10 | two_window_small, hidden_0.85, N=10 | forbidden, hidden_0.85, N=10 |
|---|---|---|---|---|---|---|---|
| best state-dependent order | 0.0232 | 0.2572 | 0.0748 | 0.7932 | 0.0239 | 0.0261 | 0.1630 |
| error now + exact cost of finishing greedily | 0.0299 | 0.2845 | 0.0844 | 0.9674 | 0.0240 | 0.0349 | 0.1691 |
| error now + exact cost of finishing in random order | 0.0269 | 0.3045 | 0.0848 | 0.9554 | 0.0240 | 0.0609 | 0.1814 |
| lookahead on true error, depth 1 | 0.0344 | 0.6946 | 0.0776 | 1.2104 | 0.0250 | 0.0305 | 0.1698 |
| lookahead on true error, depth 4 | 0.0255 | 0.5659 | 0.0748 | 1.1008 | 0.0239 | 0.0261 | 0.1642 |
| ideal greedy (lowest true error now) | 0.0422 | 0.9095 | 0.0996 | 1.2935 | 0.0301 | 0.0391 | 0.1800 |
| lex | 0.0498 | 0.9262 | 0.1028 | 1.4388 | 0.0645 | 0.0502 | 0.1772 |
| sum | 0.0417 | 0.8378 | 0.1362 | 1.0639 | 0.0589 | 0.0620 | 0.2039 |
| separator | 0.0449 | 0.7671 | 0.1614 | 1.0405 | 0.1338 | 0.3890 | 0.4041 |
| greedy on the estimate | 0.0403 | 0.8801 | 0.1759 | 1.8300 | 0.6368 | 0.0514 | 0.1992 |
| one-step plan on the estimate | 0.0537 | 1.0622 | 0.1188 | 1.4208 | 0.0293 | 0.0682 | 0.1933 |
| full-depth plan on the estimate | 0.0442 | 1.0237 | 0.1508 | 1.6320 | 0.0257 | 0.0306 | 0.1859 |
| most tilted first | 0.0491 | 1.0157 | 0.1028 | 1.4388 | 0.0645 | 0.0502 | 0.1772 |
| least tilted first | 0.0926 | 1.0606 | 0.5937 | 2.8824 | 0.9095 | 1.1655 | 0.6959 |
| random | 0.0653 | 0.9423 | 0.3447 | 2.8171 | 0.4315 | 0.5776 | 0.3942 |
| left to right | 0.1015 | 1.3631 | 0.4651 | 3.2454 | 0.4495 | 0.4890 | 0.1792 |
| worst | 0.1834 | 2.2601 | 0.7837 | 6.4061 | 0.9681 | 1.1897 | 0.7713 |

As a multiple of the optimal KL:

| rule | counting, hidden_0.85, N=10 | counting, chain_0.85, N=10 | dyck, hidden_0.85, N=12 | dyck, chain_0.85, N=12 | window_small, hidden_0.85, N=10 | two_window_small, hidden_0.85, N=10 | forbidden, hidden_0.85, N=10 |
|---|---|---|---|---|---|---|---|
| error now + exact cost of finishing greedily | 1.29 | 1.11 | 1.13 | 1.22 | 1.00 | 1.34 | 1.04 |
| error now + exact cost of finishing in random order | 1.16 | 1.18 | 1.13 | 1.20 | 1.00 | 2.33 | 1.11 |
| lookahead on true error, depth 1 | 1.48 | 2.70 | 1.04 | 1.53 | 1.04 | 1.17 | 1.04 |
| lookahead on true error, depth 4 | 1.10 | 2.20 | 1.00 | 1.39 | 1.00 | 1.00 | 1.01 |
| ideal greedy (lowest true error now) | 1.82 | 3.54 | 1.33 | 1.63 | 1.26 | 1.50 | 1.10 |
| lex | 2.15 | 3.60 | 1.37 | 1.81 | 2.70 | 1.92 | 1.09 |
| sum | 1.80 | 3.26 | 1.82 | 1.34 | 2.46 | 2.38 | 1.25 |
| separator | 1.93 | 2.98 | 2.16 | 1.31 | 5.59 | 14.90 | 2.48 |
| greedy on the estimate | 1.73 | 3.42 | 2.35 | 2.31 | 26.60 | 1.97 | 1.22 |
| one-step plan on the estimate | 2.31 | 4.13 | 1.59 | 1.79 | 1.22 | 2.61 | 1.19 |
| full-depth plan on the estimate | 1.90 | 3.98 | 2.02 | 2.06 | 1.07 | 1.17 | 1.14 |
| most tilted first | 2.11 | 3.95 | 1.37 | 1.81 | 2.70 | 1.92 | 1.09 |
| least tilted first | 3.99 | 4.12 | 7.94 | 3.63 | 37.99 | 44.65 | 4.27 |
| random | 2.81 | 3.66 | 4.61 | 3.55 | 18.02 | 22.13 | 2.42 |
| left to right | 4.37 | 5.30 | 6.22 | 4.09 | 18.77 | 18.74 | 1.10 |
| worst | 7.90 | 8.79 | 10.48 | 8.08 | 40.43 | 45.58 | 4.73 |

The estimate is the closed form on counting and the pair cost elsewhere (exact one-step update).

What the optimal order does, by step (expectation under its own sampler). First line: the share of steps where its pick is
also the cheapest now. Second: error it pays at each step. Third: error the ideal greedy order pays at each step.

```
counting, hidden_0.85
  cheapest now   1.00 0.00 0.30 0.34 0.33 0.46 0.77 0.91 1.00 1.00
  optimal pays   0.0002 0.0026 0.0035 0.0046 0.0057 0.0030 0.0015 0.0016 0.0005 0.0000
  greedy pays    0.0002 0.0008 0.0016 0.0033 0.0069 0.0053 0.0107 0.0121 0.0013 0.0000
counting, chain_0.85
  cheapest now   0.00 0.70 0.02 0.64 0.11 0.32 0.76 0.97 1.00 1.00
  optimal pays   0.1016 0.0061 0.0230 0.0126 0.0955 0.0052 0.0129 0.0002 0.0000 0.0000
  greedy pays    0.0214 0.0049 0.0174 0.0300 0.0462 0.0781 0.2021 0.1147 0.3949 0.0000
dyck, hidden_0.85
  cheapest now   1.00 1.00 0.00 0.00 0.59 0.55 0.21 0.62 0.26 0.89 1.00 1.00
  optimal pays   0.0000 0.0000 0.0110 0.0180 0.0090 0.0101 0.0116 0.0068 0.0054 0.0018 0.0012 0.0000
  greedy pays    0.0000 0.0000 0.0110 0.0087 0.0223 0.0118 0.0135 0.0116 0.0134 0.0064 0.0009 0.0000
dyck, chain_0.85
  cheapest now   1.00 0.00 0.56 0.56 0.75 0.26 0.08 0.11 0.38 1.00 1.00 1.00
  optimal pays   0.0000 0.6200 0.0005 0.0288 0.0395 0.0647 0.0345 0.0038 0.0014 0.0000 0.0000 0.0000
  greedy pays    0.0000 0.0000 0.1225 0.0864 0.1131 0.0712 0.1114 0.0738 0.2000 0.1041 0.4112 0.0000
window_small, hidden_0.85
  cheapest now   1.00 1.00 0.38 1.00 1.00 1.00 1.00 1.00 1.00 1.00
  optimal pays   0.0109 0.0096 0.0035 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
  greedy pays    0.0109 0.0096 0.0002 0.0002 0.0004 0.0032 0.0030 0.0024 0.0002 0.0000
two_window_small, hidden_0.85
  cheapest now   1.00 1.00 0.35 1.00 1.00 1.00 1.00 1.00 1.00 1.00
  optimal pays   0.0087 0.0007 0.0166 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
  greedy pays    0.0087 0.0007 0.0002 0.0049 0.0098 0.0068 0.0052 0.0028 0.0000 0.0000
forbidden, hidden_0.85
  cheapest now   0.00 0.26 0.42 0.22 0.35 0.33 0.35 0.54 1.00 1.00
  optimal pays   0.0887 0.0139 0.0052 0.0226 0.0078 0.0108 0.0081 0.0041 0.0019 0.0000
  greedy pays    0.0657 0.0181 0.0271 0.0144 0.0165 0.0128 0.0131 0.0080 0.0044 0.0000
```

- The room: random order is 2.8, 3.7, 4.6, 3.6, 18.0, 22.1, 2.4 times the optimal KL in the seven cells.
- The optimal order is not greedy. It pays more than the cheapest available reveal in the early and middle steps and
  less than the greedy order late. On counting with the Markov chain its first reveal is the dearest position on offer
  (0.102 nats; the cheapest is 0.021), and the greedy order later pays 0.39 nats in a single step.
- The horizon is long. Lookahead on the true error to depth 4 is still 2.2 and 1.4 times optimal on the two Markov-chain
  cells. Scoring a reveal by its error now plus the exact cost of *finishing* from the state it leaves (greedily or in
  random order) is within 1.34 times optimal in every cell when the continuation is greedy.
- No fixed recipe is close everywhere: the best of `lex`, `sum`, `separator` is 1.80, 2.98, 1.37, 1.31, 2.46, 1.92, 1.09 times optimal.
- Planning on the present estimates, even to full depth, is not reliably better than their greedy use: the estimates are
  not accurate at the hypothetical states a plan visits. The exception is the small window, where the greedy use fails.

## 36t. The order problem needs no exact twist: total tilt, fixed orders, planning in a surrogate (local, 2026-11-16) [identity proved and verified; planning verified at 10 to 12 positions only]

Script `experiments/parallel_unmasking/surrogate_plan.py`; results `results/parallel_unmasking/order_study/surrogate_*.json`.

**Identity.** One token per step; the order may depend on the revealed state; the model is order-consistent, so
p(x) = prod_t m_{i_t}(a_t | x_t) along any such order; the proposal is the product twist's constrained marginal q_i, so
every sample satisfies C. Then, with s the sampler's law,

    KL(s || p(. | C)) = E_s[log s(x) - log p(x)] + log p(C) = log p(C) + E_s[ sum_t KL(q_{i_t} || m_{i_t}) ].

- KL(q_i || m_i), the *tilt*, is how far the proposal moves the model's marginal at the revealed position. Both q_i and
  m_i are in hand when the position is chosen, so the cost of the present step is known exactly, without the exact twist.
- log p(C) does not depend on the order. The best order is the one with the least expected total tilt.
- Relation to the true error: eps_i(x) = tilt_i(x) + log psi(x) - E_{a ~ q_i} log psi(x with i = a), with psi the exact
  twist; the psi terms telescope to log p(C). The two costs have the same best plan and different greedy orders: greedy on
  eps assumes the rest of the decode is exact, greedy on tilt assumes it is free. Least-tilted-first is 3.6 to 45 times optimal.
- The expected total tilt is minus the expected log weight of a particle (the evidence lower bound read step by step).
  What is new here is its use as the planning objective, and that it makes the order problem a statement about the
  model's conditionals and the route alone.
- Checked in all seven cells: total tilt + log p(C) equals the KL to six decimals for the optimal and the random order,
  and the best plan for total tilt has the optimal KL.

**Fixed orders and surrogates.** Exact KL as a multiple of the optimal KL (lower is better):

| cell | random | ideal greedy | best fixed order | plan in 1st-order chain | plan in 2nd-order chain | fixed order found in 2nd-order chain | true tilt now + 2nd-order chain afterwards |
|---|---|---|---|---|---|---|---|
| counting, hidden_0.85, N=10 | 2.81 | 1.82 | 1.42 | 2.19 | 1.38 | 1.62 | 3.01 |
| counting, chain_0.85, N=10 | 3.66 | 3.54 | 1.49 | 1.00 | 1.00 | 1.49 | 1.15 |
| dyck, hidden_0.85, N=12 | 4.61 | 1.33 | 1.46 | 2.02 | 1.24 | 1.65 | 3.08 |
| dyck, chain_0.85, N=12 | 3.55 | 1.63 | 1.12 | 1.00 | 1.00 | 1.12 | 1.00 |
| window_small, hidden_0.85, N=10 | 18.02 | 1.26 | 1.00 | 3.88 | 1.22 | 1.00 | 9.05 |
| two_window_small, hidden_0.85, N=10 | 22.13 | 1.50 | 1.00 | 2.62 | 1.34 | 1.34 | 2.64 |
| forbidden, hidden_0.85, N=10 | 2.42 | 1.10 | 1.10 | 1.16 | 1.06 | 1.10 | 1.60 |

- *Best fixed order*: the best state-independent permutation found by local search (swaps and insertions, 13 starts) with
  exact evaluation. *Plan in a surrogate*: a first- or second-order Markov chain is fitted to 2,000 unconditional pilot
  samples; the best plan is computed inside it (its own exact twist, no further calls to the model) and then scored
  exactly on the true model.
- The plan from the second-order surrogate is 1.00 to 1.38 times optimal and beats the ideal greedy order, which needs the
  exact twist, in 7 of 7 cells. It also beats the best of `lex`, `sum`, `separator` in 7 of 7.
- A first-order surrogate is not enough on the hidden chain (2.0 to 3.9 times optimal on four of its five cells): the
  surrogate must carry the dependence beyond neighbours. Where the family contains the truth (Markov chain) it is 1.00.
- A fixed order recovers the whole gain on the windows, most of it on forbidden and Dyck with the Markov chain, and
  part of it on counting and on Dyck with the hidden chain (1.4 to 1.5 times optimal). The best fixed orders found:
  - counting, hidden_0.85: [7, 6, 0, 1, 2, 8, 3, 4, 5, 9]
  - counting, chain_0.85: [2, 4, 6, 8, 0, 7, 9, 3, 1, 5]
  - dyck, hidden_0.85: [11, 0, 5, 6, 10, 4, 9, 1, 2, 8, 7, 3]
  - dyck, chain_0.85: [11, 5, 0, 7, 4, 8, 3, 9, 2, 6, 10, 1]
  - window_small, hidden_0.85: [6, 5, 3, 4, 8, 2, 1, 0, 9, 7]
  - two_window_small, hidden_0.85: [1, 2, 8, 7, 3, 9, 5, 0, 6, 4]
  - forbidden, hidden_0.85: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  On the Markov chain under counting it is every second position first: after those the rest are independent and the
  product twist is exact. On the hidden chain it is contiguous runs. The windows go first, then outwards. There is no
  common pattern, which is the case for computing the order rather than prescribing it.
- Mixing does not work: the true tilt now plus the surrogate's expected tilt afterwards is worse than the surrogate's own
  plan in all seven cells (by 0.4% on Dyck with the Markov chain, by 1.15 to 7 times elsewhere). The tilt totals are several nats and the differences that matter are hundredths; an error in
  the surrogate's log p(C | x) does not cancel unless both terms come from the same model.
- Limits. 10 to 12 positions is under two correlation lengths. The backward pass over all states does not exist at 64
  positions and beyond, so a planner that scales is still to be built (fixed order optimised inside the surrogate;
  rollouts inside the surrogate; a fitted value). Alphabets of 2 and 3. The pilot is the same 2,000 samples the kernel uses.

## 36u. Calibrated twists on the resampling pressure case (cluster job `4b0e67e1`, 2026-11-16) [verified; 50 runs at 16 particles, 12 at 64]

Arm set `calibrate` of `compare.py`; results `results/parallel_unmasking/calibrate_K{16,64}.json`, summary `calibrate_N256_summary.json`.
`hidden_0.85`, 256 positions, forbidden and Dyck. Calibration (`twist_temp` tau) flattens the marginals of the masked
positions inside the intermediate twist; tau = 1 is no calibration. RMS of log(Z_hat / p(C)) in nats, lower is better,
with its jackknife standard error; 16 particles | 64 particles. The uncalibrated rows reproduce section 36k (same seeds).

**forbidden, one token per step (256 steps)**

| twist | resampling | tau = 1 | tau = 0.7 | tau = 0.5 |
|---|---|---|---|---|
| product | off | 3.32 ± 0.26 \| 2.13 ± 0.24 | | |
| product | defensive | 2.18 ± 0.22 \| 1.35 ± 0.18 | 1.39 ± 0.15 \| 0.56 ± 0.07 | 1.92 ± 0.21 \| 1.03 ± 0.18 |
| product | plain | 2.68 ± 0.19 \| 1.00 ± 0.13 | 1.22 ± 0.17 \| 0.77 ± 0.17 | 1.97 ± 0.18 \| 0.96 ± 0.14 |
| chain | defensive | 1.39 ± 0.12 \| 0.66 ± 0.10 | 1.51 ± 0.19 \| 0.65 ± 0.10 | 1.81 ± 0.20 \| 1.41 ± 0.20 |
| chain | plain | 1.98 ± 0.17 \| 1.05 ± 0.21 | 1.32 ± 0.14 \| 0.83 ± 0.18 | 1.87 ± 0.18 \| 1.45 ± 0.31 |

**forbidden, budgeted, lex, B = 0.03 (26 steps)**

| twist | resampling | tau = 1 | tau = 0.7 | tau = 0.5 |
|---|---|---|---|---|
| product | off | 1.39 ± 0.10 \| 0.84 ± 0.12 | | |
| product | defensive | 1.20 ± 0.11 \| 0.60 ± 0.10 | 0.96 ± 0.10 \| 0.47 ± 0.06 | 0.89 ± 0.08 \| 0.38 ± 0.06 |
| product | plain | 1.34 ± 0.13 \| 0.79 ± 0.08 | 0.98 ± 0.11 \| 0.49 ± 0.14 | 1.06 ± 0.10 \| 0.55 ± 0.16 |
| chain | defensive | 0.98 ± 0.09 \| 0.37 ± 0.06 | 0.87 ± 0.08 \| 0.38 ± 0.06 | 0.80 ± 0.10 \| 0.59 ± 0.12 |
| chain | plain | 1.03 ± 0.10 \| 0.55 ± 0.08 | 1.02 ± 0.08 \| 0.57 ± 0.10 | 0.96 ± 0.07 \| 0.59 ± 0.14 |

**dyck, one token per step (256 steps)**

| twist | resampling | tau = 1 | tau = 0.7 | tau = 0.5 |
|---|---|---|---|---|
| product | off | 0.48 ± 0.06 \| 0.21 ± 0.05 | | |
| product | defensive | 0.68 ± 0.09 \| 0.42 ± 0.05 | 0.44 ± 0.04 \| 0.16 ± 0.04 | 0.49 ± 0.05 \| 0.14 ± 0.03 |
| product | plain | 0.88 ± 0.08 \| 0.95 ± 0.39 | 0.43 ± 0.04 \| 0.28 ± 0.04 | 0.44 ± 0.05 \| 0.26 ± 0.03 |
| chain | defensive | 0.39 ± 0.03 \| 0.23 ± 0.06 | 0.37 ± 0.03 \| 0.18 ± 0.02 | 0.49 ± 0.05 \| 0.17 ± 0.05 |
| chain | plain | 0.42 ± 0.06 \| 0.14 ± 0.02 | 0.38 ± 0.04 \| 0.28 ± 0.03 | 0.33 ± 0.03 \| 0.22 ± 0.05 |

**dyck, budgeted, lex, B = 0.03 (16 steps)**

| twist | resampling | tau = 1 | tau = 0.7 | tau = 0.5 |
|---|---|---|---|---|
| product | off | 0.39 ± 0.04 \| 0.20 ± 0.03 | | |
| product | defensive | 0.38 ± 0.04 \| 0.23 ± 0.04 | 0.41 ± 0.04 \| 0.13 ± 0.03 | 0.33 ± 0.03 \| 0.18 ± 0.03 |
| product | plain | 0.59 ± 0.05 \| 0.26 ± 0.05 | 0.37 ± 0.04 \| 0.23 ± 0.05 | 0.46 ± 0.04 \| 0.23 ± 0.06 |
| chain | defensive | 0.39 ± 0.04 \| 0.18 ± 0.03 | 0.44 ± 0.04 \| 0.21 ± 0.05 | 0.36 ± 0.04 \| 0.15 ± 0.03 |
| chain | plain | 0.33 ± 0.03 \| 0.23 ± 0.07 | 0.42 ± 0.06 \| 0.13 ± 0.03 | 0.47 ± 0.08 \| 0.16 ± 0.04 |

- Forbidden, one token per step: calibrating the product twist at tau = 0.7 brings it level with the uncalibrated chain
  twist under defensive resampling (1.39 against 1.39 at 16 particles, 0.56 against 0.66 at 64), from 2.18 and 1.35 uncalibrated.
  tau = 0.5 flattens too far for both twists. The chain twist gains nothing from calibration under defensive resampling;
  under plain resampling tau = 0.7 removes its loss.
- Dyck, one token per step: calibration removes the harm that resampling does with the product twist (0.68 and 0.88
  uncalibrated, 0.43 to 0.49 calibrated, 0.48 with resampling off).
- Budgeted schedule: every calibrated arm and the chain twist are within about two standard errors of each other;
  the uncalibrated product twist is the worst on forbidden.
- Reading: on this case calibration and the chain twist are substitutes as intermediate twists. Neither changes the
  proposal, so neither changes the per-particle KL (10.4 nats on forbidden here); section 36w is about the proposal.

## 36v. Order consistency of MDLM, follow-up (cluster job `8fabd084`, one L40S on node3203, 2026-11-16) [measured; 6 to 32 sequences per length]

`experiments/parallel_unmasking/mdlm_order_consistency.py`; results `results/parallel_unmasking/mdlm_consistency/`.
As section 36p, with real prose (six public-domain passages cut into chunks), lengths to 1,024, and the samples saved.
All quantities in nats. *sd over orders*: standard deviation of log p of one sequence across random reveal orders
(0 for an order-consistent model). *Gap to mixture*: log of the average probability over orders minus the average log.
*Own order*: log p under the order that generated a sample minus the average over other orders.

| source | tokens | sequences | random orders | -log p per token | sd over orders | per token | gap to mixture | left-to-right | right-to-left | own order | distinct-token share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real text | 32 | 29 | 16 | 5.32 | 16.0 ± 1.1 | 0.500 | 21.1 ± 1.8 | +11.4 ± 2.6 | -2.1 ± 2.8 |  | 0.81 |
| real text | 64 | 12 | 16 | 4.52 | 18.2 ± 2.0 | 0.284 | 29.6 ± 3.7 | +11.7 ± 4.9 | -5.8 ± 3.7 |  | 0.70 |
| real text | 128 | 6 | 16 | 3.97 | 15.3 ± 0.7 | 0.120 | 25.2 ± 3.0 | +3.1 ± 8.8 | -13.2 ± 6.0 |  | 0.62 |
| model samples | 32 | 32 | 16 | 3.16 | 9.7 ± 1.0 | 0.302 | 13.3 ± 1.5 | +4.1 ± 1.4 | +1.3 ± 1.8 | +38 ± 4 | 0.37 |
| model samples | 64 | 32 | 16 | 3.16 | 14.4 ± 1.4 | 0.225 | 19.3 ± 2.0 | -0.1 ± 2.6 | -10.4 ± 3.1 | +67 ± 7 | 0.37 |
| model samples | 128 | 32 | 16 | 3.74 | 16.2 ± 1.3 | 0.127 | 23.2 ± 1.9 | +0.7 ± 3.7 | -19.9 ± 4.3 | +102 ± 7 | 0.48 |
| model samples | 256 | 32 | 16 | 4.44 | 19.3 ± 1.0 | 0.075 | 30.8 ± 2.0 | -1.8 ± 3.6 | -22.1 ± 4.0 | +169 ± 8 | 0.58 |
| model samples | 512 | 8 | 8 | 5.34 | 16.9 ± 1.2 | 0.033 | 23.4 ± 2.1 | -12.9 ± 7.0 | -21.8 ± 6.3 | +270 ± 23 | 0.57 |
| model samples | 1024 | 8 | 4 | 6.56 | 24.4 ± 2.2 | 0.024 | 22.7 ± 3.5 | -19.6 ± 5.0 | -41.9 ± 13.2 | +380 ± 19 | 0.59 |

- The inconsistency is real: on prose that no order generated, log p varies across orders with a standard deviation of
  15 to 18 nats at 32 to 128 tokens.
- It does not grow with length. On model samples the standard deviation is 10 to 24 nats from 32 to 1,024 tokens, so
  per token it falls from 0.30 to 0.024.
- The short samples of section 36p were degenerate, as suspected: at 32 and 64 tokens many are one token repeated
  (distinct-token share 0.37). From 128 tokens the samples read as text (0.48 to 0.59; prose is 0.62 to 0.81).
- A sample is far more probable under its own order: 38 to 380 nats, 1.2 falling to 0.37 nats per token. Entropy along
  the generating order falls through the decode at every length (6.7 to 5.7 nats per step at 1,024 tokens).
- Left-to-right scores prose 11 nats higher than a random order at 32 and 64 tokens; right-to-left is lower at every
  length for samples of 64 tokens and more.
- Consequence for the target (random-order average, user decision of 2026-11-16): one order's law differs from the
  average by tens of nats in log-probability on a typical sequence, at any length. Against that, the schedule errors
  measured in this document are 0.01 to 1 nat. A constraint-aware order on this model changes the sampled law by
  far more than any of the errors being optimised, so the order rule and the target have to be settled together.

## 36w. The chain twist as the proposal (local, 2026-11-16) [exact at 10 to 12 positions; sampled at 64 with 300 particles, one hidden chain]

Scripts `surrogate_plan.py --proposal chain` (exact) and `counting_orders.py --proposal chain` (sampled); results
`results/parallel_unmasking/order_study/chainprop_*.json` and `prop_*_N64.json`.
In the sampler the chain twist is an intermediate potential only: the revealed token is still drawn from the product
twist's constrained marginal, so the per-particle KL is unchanged (section 36f). Here the token is drawn from the chain
twist's constrained marginal instead (`ChainTwist.marginals`, pair table from 2,000 pilot samples). The identity of
section 36t holds for any proposal that stays inside the constraint. Checks: the chain proposal puts at most 1e-15 on a
value with no completion; the tables match brute-force enumeration of the left-to-right sampler to six decimals for both
proposals in five cells; total tilt + log p(C) equals the KL to six decimals in all seven.

**Exact, 10 to 12 positions.** KL of the sampler from the constrained law, nats, lower is better: product proposal -> chain proposal.

| cell | random | ideal greedy | best fixed order | optimal | plan in 2nd-order surrogate | best fixed order under the chain proposal |
|---|---|---|---|---|---|---|
| counting, hidden_0.85, N=10 | 0.0653 -> 0.0549 | 0.0422 -> 0.0161 | 0.0329 -> 0.0145 | 0.0232 -> 0.0067 | 0.0321 -> 0.0180 | [2, 1, 3, 4, 5, 6, 7, 8, 9, 0] |
| counting, chain_0.85, N=10 | 0.9423 -> 0.3277 | 0.9095 -> 0.0273 | 0.3822 -> 0.0005 | 0.2572 -> 0.0005 | 0.2577 -> 0.0005 | [1, 2, 4, 5, 6, 7, 8, 0, 3, 9] |
| dyck, hidden_0.85, N=12 | 0.3447 -> 0.0778 | 0.0996 -> 0.0052 | 0.1092 -> 0.0095 | 0.0748 -> 0.0015 | 0.0924 -> 0.0145 | [0, 11, 5, 6, 2, 3, 1, 7, 8, 10, 9, 4] |
| dyck, chain_0.85, N=12 | 2.8171 -> 0.0891 | 1.2935 -> 0.0288 | 0.8856 -> 0.0000 | 0.7932 -> 0.0000 | 0.7935 -> 0.0000 | [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 10, 8] |
| window_small, hidden_0.85, N=10 | 0.4315 -> 0.1953 | 0.0301 -> 0.0098 | 0.0239 -> 0.0115 | 0.0239 -> 0.0098 | 0.0291 -> 0.0164 | [5, 3, 6, 4, 0, 1, 7, 8, 2, 9] |
| two_window_small, hidden_0.85, N=10 | 0.5776 -> 0.1682 | 0.0391 -> 0.0008 | 0.0261 -> 0.0011 | 0.0261 -> 0.0007 | 0.0351 -> 0.0010 | [1, 7, 8, 2, 3, 4, 5, 0, 6, 9] |
| forbidden, hidden_0.85, N=10 | 0.3942 -> 0.0561 | 0.1800 -> 0.0117 | 0.1792 -> 0.0223 | 0.1630 -> 0.0095 | 0.1734 -> 0.0168 | [6, 7, 9, 8, 5, 4, 3, 2, 1, 0] |

- The chain proposal lowers the KL under every rule in every cell. Its ideal greedy order beats the *optimal* order of
  the product proposal in all seven cells.
- Under the chain proposal the best fixed orders are close to contiguous, and on the Markov chain the error is
  essentially zero: the first-order chain twist is then exact for a contiguous masked stretch.
- Order still matters: random order is 3 to 200 times the ideal greedy KL under the chain proposal.

**Sampled, 64 positions, `hidden_0.85`, 300 particles.** KL as the sum over steps of the true error, nats, lower is better:
product proposal -> chain proposal. The estimate from the weights agrees to within about two of its standard errors (0.01 to 0.17).

| constraint | random | left to right | lex | sum | ideal greedy |
|---|---|---|---|---|---|
| counting | 0.180 -> 0.188 | 0.778 -> 0.208 | 0.204 -> 0.263 | 0.123 -> 0.146 | 0.191 -> 0.078 |
| dyck | 0.642 -> 0.298 | 1.291 -> 0.314 | 0.586 -> 0.288 | 0.599 -> 0.246 | 0.262 -> 0.103 |
| forbidden | 2.588 -> 0.419 | 0.586 -> 0.216 | 0.582 -> 0.216 | 1.171 -> 0.284 | 0.599 -> 0.096 |
| window | 0.389 -> 0.256 | 0.421 -> 0.254 | 0.076 -> 0.080 | 0.091 -> 0.078 | 0.050 -> 0.017 |
| two_window | 0.730 -> 0.450 | 0.733 -> 0.365 | 0.085 -> 0.102 | 0.115 -> 0.135 | 0.278 -> 0.032 |

- Forbidden and Dyck: the chain proposal lowers the KL under every order, by 2.7 to 6 times on forbidden and 2 to 4 times on Dyck.
- Counting and the windows: no gain under the fixed recipes (`lex`, `sum`), which already reveal the constrained positions
  in a good order for the product proposal; a gain of 1.5 to 2 times under random and left-to-right on the windows; none under random on counting (left-to-right there improves 3.7 times, to the level of random).
- With the ideal greedy order the chain proposal gives the lowest KL measured on every constraint: counting 0.078 against 0.123, dyck 0.103 against 0.262, forbidden 0.096 against 0.582, window 0.017 against 0.050, two_window 0.032 against 0.085 for the best product-proposal arm.
- Without the exact twist the best chain-proposal arm is counting 0.146 (product 0.123), dyck 0.246 (product 0.586), forbidden 0.216 (product 0.582), window 0.078 (product 0.076), two_window 0.102 (product 0.085): the gain that is
  reachable today is on forbidden and Dyck only. The rest needs an order rule that finds what the ideal greedy order finds.
- The small-size picture overstates the gain. At 10 positions a near-contiguous order was almost ideal under the chain
  proposal; at 64 positions left-to-right is level with random on counting, Dyck and the window. The hidden chain's
  dependence (correlation length about 6) is longer than a first-order chain with one pooled table carries.
- Not done: the chain proposal inside the sampler (a joint route sample under the chain law is needed when several
  positions are revealed in one step); other hidden chains; 256 positions; cost per step.

## 36. Next steps (2026-11-16; replaces section 34)

Done since section 34: the selection-fix run (section 31) and the refactor (section 35). One departure from item 2
of section 34: the studies are registered functions behind one command line, not separate scripts, so the existing
job scripts and arm names keep working.

0. Pull-in (sections 36j, 36l): it is the extra cost of the parallel reveal on CPU and GPU and admits nothing on
   window, counting and forbidden; decide when it should run (with the selection rule). Add calibrated arms
   (`twist_temp`) to the pressure case for both twists; the chain twist needs tempering of its masked marginals first.
   Until the pool mount is fixed, pin GPU jobs to a node that passes the storage probe. The order study (section 36i) is input to the
   selection rule; the user has thoughts on that rule and asked to be flagged before the work starts.
0a. Selection after sections 36q to 36t: the objective is the expected total tilt, and the ideal order is a long-horizon
   plan that a second-order surrogate recovers at 10 to 12 positions. To decide with the user: which planner scales
   (a fixed order optimised offline inside the surrogate; rollouts inside the surrogate at decode time; a fitted value),
   and what the surrogate is on a language model (a chain over the constraint's classes fitted to pilot samples).
   Then: the same test at 32 to 64 positions against the measured orders of section 36i; the parallel form (co-reveal
   term in the same objective); the exact search at 12 to 14 positions on the cluster.
   Also to decide (section 36w): move the chain twist into the proposal. At 64 positions it halves the KL on forbidden
   and Dyck under any order and gives the best arm on every constraint with the ideal greedy order.
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
