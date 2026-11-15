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

## 19. Next steps (as of 2026-11-15 early morning; superseded items are in sections 14 to 18)

1. Hidden-mass check at N = 512 and 1,024 (job `36c53d08`): is the policy's remaining error resampling, and which
   remedy keeps resampling (calibrated twist, 16 particles, lower trigger).
2. Budget sweep at N = 256 and 1,024 (job `b4e15799`): steps against budget, predicted against measured r, graded
   against two-bin kernel, order under a budget, dilated baselines.
3. Priority-order test at N = 64 (`order_N64.json`, local, in progress), then the least-coupled arms.
4. Constraint-aware closeness through Hessian-vector products of log Z (one extra backward pass), for the
   mixed-path effect on copy. [hypothesis]
5. An error measure sensitive to long-range joint structure.
6. Stress constraints (required motif, per-segment counts, window all-different, parity, motif and count) as the
   held-out evaluation of the priority score.
7. Owed: three proofs (resampling at unequal progress under rollback; unbiasedness with an adaptive proposal;
   accuracy of the first-order effective-coupling formula); the fix to Definition 5.1 of the notes; the
   `TorchChart` gradient; strip and grid oracles; a trained small model.
