# Twist design: look-ahead twists and native mixing (rounds 1–3)

**Testbed.** Synthetic denoiser with strong context dependence (scale 8), TWISTER Fig. 1(a) constraint "a.*",
N = 32, T = 8, 20,000 runs per arm, h₀ = 0.0706 from 200k native samples (`experiments/twist_lab.py`). This
reproduces the OWT-130M pathology in milder form: trace log-weights reach −48, and plain resampling inflates the
s.e. of Ẑ 2.7× over resampling-off.

**Efficiency** = 1 / (relative s.e.² × denoiser rows per run), relative to the frozen twist without resampling.

| arm | K | cost | Ẑ/h₀ | rel. s.e. | efficiency vs off |
|---|---|---|---|---|---|
| frozen, no resampling | 4 | 32 | 1.006 | 0.009 | 1.00 |
| frozen, plain | 4 | 32 | 0.979 | 0.025 | 0.14 |
| frozen, defensive ε = 0.25 | 4 | 32 | 0.994 | 0.014 | 0.47 |
| frozen, plain | 8 | 64 | 0.985 | 0.026 | 0.06 |
| frozen, defensive | 8 | 64 | 0.980 | 0.009 | 0.57 |
| look-ahead j = 1, plain | 4 | 64 | 1.113 | 0.179 | 0.001 |
| look-ahead j = 1, defensive | 4 | 64 | 1.011 | 0.014 | 0.20 |
| look-ahead j = 2, plain | 4 | 92 | 0.836 | 0.052 | 0.01 |
| look-ahead full (unbiased ĥ), plain | 4 | 176 | 0.810 | 0.131 | 0.001 |
| look-ahead full, defensive | 4 | 176 | 1.019 | 0.018 | 0.05 |
| native mixture a = 0.2, plain | 4 | 32 | 0.974 | 0.029 | 0.11 |
| native mixture a = 0.2, defensive | 4 | 32 | 0.980 | 0.013 | 0.50 |

**Hidden mass** (share of h₀ carried by particles below normalised weight ε at the first trigger step,
resampling-off traces under each twist):

| twist | ε = 10⁻⁸ | 10⁻⁶ | 10⁻⁴ | 10⁻² |
|---|---|---|---|---|
| frozen | 0.000 | 0.000 | 0.005 | 0.079 |
| look-ahead j = 1 | 0.001 | 0.008 | 0.048 | 0.195 |
| look-ahead full | 0.027 | 0.069 | 0.156 | 0.318 |
| native mixture | 0.000 | 0.000 | 0.006 | 0.074 |

## Reading

1. **Monte Carlo look-ahead twists make resampling worse, not better.** A single speculative rollout gives an
   unbiased estimate of h, but that estimate has the heavy-tailed distribution of a single-particle importance
   weight: usually far below h, occasionally far above. The hidden-mass proposition says resampling misallocates
   in proportion to the spread of log(h/ψ) across siblings, and estimation noise adds to that spread. The full
   rollout, which is unbiased for the Doob function, has the most hidden mass (6.9% below 10⁻⁶ against 0.04%).
   **Unbiasedness of the twist is the wrong criterion; low dispersion of log(ψ/h) is the right one.**
2. **Native mixing (a = 0.2) changes little.** Its efficiency matches the frozen twist under either resampler.
3. **No resampling is the most efficient arm** on this testbed, and defensive resampling recovers about half of
   that efficiency. This matches E1 on the regular family, where resampling-off also had the lowest TV.
4. **Defensive resampling rescues every twist** (look-ahead j = 1: efficiency 0.001 → 0.20), consistent with its
   variance bound not depending on the twist.

## Round 2: low-dispersion look-ahead

Same testbed, 20,000 runs per arm (`twist_lab.py --arms ...`, output `twist_lab_r2.json`). "geom4" averages
log-gains over m = 4 rollouts (a geometric mean of the gain estimates); "shrink50" sets
log ψ = log η + 0.5 · log Ĝ.

| arm | K | cost | Ẑ/h₀ | rel. s.e. | efficiency vs off |
|---|---|---|---|---|---|
| frozen, no resampling | 4 | 32 | 1.006 | 0.009 | 1.00 |
| frozen, defensive | 4 | 32 | 0.993 | 0.011 | 0.66 |
| look-ahead j = 1, geom4, plain | 4 | 160 | 0.986 | 0.072 | 0.003 |
| look-ahead j = 1, shrink50, plain | 4 | 64 | 0.937 | 0.026 | 0.07 |
| look-ahead full, geom4, plain | 4 | 608 | 1.049 | 0.126 | 0.0003 |
| look-ahead full, geom4 + shrink50, plain | 4 | 608 | 1.059 | 0.067 | 0.001 |
| look-ahead j = 1, geom4, defensive | 4 | 160 | 0.997 | 0.014 | 0.09 |
| look-ahead full, geom4, defensive | 4 | 608 | 0.984 | 0.015 | 0.02 |

Hidden mass (resampling-off traces):

| twist | ε = 10⁻⁸ | 10⁻⁶ | 10⁻⁴ | 10⁻² |
|---|---|---|---|---|
| frozen | 0.000 | 0.0003 | 0.006 | 0.074 |
| look-ahead j = 1, geom4 | 0.000 | 0.0009 | 0.011 | 0.114 |
| look-ahead j = 1, shrink50 | 0.000 | 0.0010 | 0.012 | 0.107 |
| look-ahead full, geom4 | 0.0006 | 0.0059 | 0.054 | 0.249 |

**Reading.**

1. **Averaging and shrinkage reduce the damage but do not remove it.** Geometric averaging cuts the full
   look-ahead's hidden mass below 10⁻⁶ from 6.9% to 0.6% of h₀. That is still 20× the frozen twist.
2. **Look-ahead loses even before cost is counted.** At equal K with defensive resampling, the rel. s.e. is
   0.014 (j = 1, geom4) and 0.015 (full, geom4) against 0.011 for the frozen twist. The 5–19× extra denoiser
   calls then push efficiency down to 0.02–0.09.
3. **Shrinkage with plain resampling shows a bias at finite K** (Ẑ/h₀ = 0.937, z = −2.5). This is the
   hidden-mass shortfall again, at a level where K = 4 does not reach the missing paths.
4. **Decision: drop Monte Carlo look-ahead twists.** On this denoiser the frozen twist is already close to the
   Doob function in the sense that matters (low sibling dispersion of log ψ/h). Any rollout estimate adds noise
   that exceeds the bias it removes. Further gains have to come from the proposal, or from deciding when to
   resample.

## Round 3: calibrated twist (route sum under q ∝ p^τ)

**Idea.** The frozen twist is the route sum of valid completions under the product of the denoiser's current
marginals p. Any positive product law q in its place keeps the sum exact for every route (automaton, chart,
factor graph) at the same cost, so it is a valid twist. `twist_temp = τ` uses q_i(c) ∝ p_i(c)^τ over classes.
The proposal is the locally optimal one for that twist (p at the reveal set, q at still-masked positions) and the
weights correct exactly. τ < 1 flattens over-confident marginals; following the importance-sampling rule that a
proposal's tails should dominate the target's, a twist that overestimates h costs a little efficiency, whereas one
that underestimates h creates hidden mass. Tests: E[Ẑ] = h₀ for forbidden, copy, ambiguous Dyck-1 and RNA pairing at
τ = 0.5, 0.25, with and without resampling (16 cases).

Same testbed as rounds 1–2, 20,000 runs per arm, K = 4, cost 32 denoiser rows per run for every arm (the twist change
is free). Efficiency is var(Ẑ) of frozen/no-resampling divided by var(Ẑ) of the arm, with 95% bootstrap intervals.

| τ | no resampling | plain | defensive ε = 0.25 | hidden mass < 10⁻² | resamplings/run (plain) |
|---|---|---|---|---|---|
| 1 (TWISTER) | 1.00 (0.96–1.05) | 0.14 (0.08–0.30) | 0.47 (0.28–0.77) | 7.8% | 1.78 |
| 0.7 | 0.99 (0.94–1.04) | 0.32 (0.20–0.54) | 0.78 (0.66–0.89) | 3.2% | 1.69 |
| 0.5 | 1.02 (0.98–1.07) | 0.63 (0.33–1.12) | 1.09 (0.93–1.23) | 0.9% | 1.56 |
| 0.35 | 1.01 (0.97–1.06) | 1.04 (0.94–1.16) | 1.15 (1.09–1.21) | 0.16% | 1.40 |
| 0.2 | 1.03 (0.98–1.07) | 1.09 (1.04–1.14) | 1.04 (0.99–1.09) | 0.15% | 1.17 |

All 15 arms have Ẑ/h₀ within 0.979–1.008 (|z| ≤ 1.8); the 12 calibrated arms lie within 0.983–1.008.

**Reading.**

1. **Calibration removes the resampling penalty.** At τ ≤ 0.35 plain resampling is no worse than resampling off
   (0.14 → 1.04–1.09), and defensive resampling is modestly better (1.15, interval 1.09–1.21). This is the first
   configuration in which resampling pays on this testbed.
2. **The mechanism is the hidden mass.** It falls monotonically with τ, from 7.8% to 0.15% of h₀ below weight 10⁻²,
   and nothing remains below 10⁻⁴ at τ ≤ 0.5. Fewer resampling events fire because the weights stay more even.
3. **Resampling-off is unaffected**, as expected: without selection the twist only changes the proposal, and here that
   is neutral.
4. **The gain from resampling itself is small at T = 8**, about 5–15%. The case for calibration is that it makes
   resampling safe, which is what long horizons will need, not that it helps much at short horizons.
5. **Caveats.** One testbed (synthetic denoiser, a regular constraint, short horizon). τ was chosen by sweep, and
   0.2 and 0.35 cannot be separated. The product family cannot represent correlated future tokens, so copy-type
   constraints still need testing.

## Next

- **Fit τ automatically** (controlled-SMC style): a short pilot run, then choose τ (or a per-step τ_t) to minimise the
  variance of log Ẑ; allow τ > 1 for under-confident denoisers.
- **Test where the product family is weakest**: copy constraints on OWT (k̂ = 1.07), long horizons (N = 256,
  T ≥ 64), and the E1 families. If correlation dominates, extend the family to mixtures of product laws.
- **Resampling allocation**: weighted-ensemble style bins on an estimate of the twist error, with defensive
  resampling as the special case of no information.
