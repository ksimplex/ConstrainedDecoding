# Constraint-aware parallel unmasking: working plan

**Status.** Design only; nothing here is implemented. Claims are tagged **[notes]** (from the method notes), **[derived]** (worked out in discussion, not run), **[checked]** (small numerical check on toy cases, not on a model) or **[conjecture]**.

Method notes: `docs/twistroute_method_notes.tex`

---

## 1. Goal

Reduce denoiser calls in trajectory-exact constrained decoding for masked diffusion models by revealing more positions per step where that is safe.

- The sampler of the notes (Algorithm 1) is kept as it is: route, twisted SMC weights, resampling.
- Only the rule that chooses which positions to reveal at each step changes.
- Setting: a denoiser whose dependence structure is unknown but hoped to be sparse, plus a hard constraint C that has a route (`Mass`, `Sample`, `Accepts`).

"Parallel" means positions revealed per denoiser call. One call predicts every masked position, so the cost is the number of calls.

## 2. Facts the plan relies on

1. The scheduler is part of the native law and cancels in the potential. Any rule s(r | x) that picks positions before drawing their values gives a sampler that is exact for its own law given C. **[notes, Thm 4.2]**
2. Positions revealed in one step are drawn jointly by the route. Direct constraint coupling among them costs nothing. **[notes, Lemma 4.1]**
3. A parallel reveal ignores only the model's dependence among the co-revealed positions. **[derived]**
4. The constraint transmits model revisions. Revealing a changes the model's prediction at a masked position j; if the constraint ties j to b, then b's constrained conditional moves too. These mixed paths raise weight variance even when a and b are model-independent. **[derived]**
5. For a mean-field model, one step is exact. **[notes, Prop 5.2; see section 10 for the definition this needs]**

## 3. The schedule policy

### 3.1 Principle

At each step, reveal as many positions as possible such that no two co-revealed positions are strongly coupled through the model, either directly or through a still-masked position that the constraint ties to one of them.

### 3.2 Ingredients

**Model coupling M, from a pilot.**
- Run m one-at-a-time traces (one position revealed per denoiser call).
- At each reveal of position a, record for every other masked position j the KL divergence between its new and old prediction. Its average over traces estimates the conditional mutual information between a and j.
- Pool by distance into two curves: pairs with a revealed position between them ("screened") and pairs without. If the screened curve is near zero, gaps between revealed positions are independent, and the schedule approaches bisection.
- This pooling is specific to 1-D sequences. On a grid, use lattice distance or a per-pair kernel; "screened" then means separated by a revealed set, not by a single position.
- The kernel is frozen after the pilot.

**Constraint coupling ρ, from the route.**
- Each step, draw a few dozen extra route samples per particle and estimate a normalised association in [0, 1] between the classes at each pair of masked positions (for example Cramér's V).
- The same route pass gives the forced positions and the constrained marginals μ.
- These extra samples must be independent of the route sample used for revealing.

**Effective coupling E.** For candidate positions a and b:

    E(a, b) = M(a, b) + Σ_j M(a, j) · ρ(j, b)        (sum over still-masked j other than a, b)

Use the symmetrised value E(a, b) + E(b, a). The first term is direct model coupling. The second is the mixed-path term of fact 4. Direct constraint coupling ρ(a, b) is deliberately not penalised (fact 2). **[derived, first-order; not checked]**

### 3.3 One step, per particle

1. Call the denoiser at the current state, batched over particles.
2. Run the route: constrained marginals, forced positions, association estimates from the extra samples.
3. Put every forced single-token position into the reveal set.
4. Rank the remaining masked positions by a priority score (section 3.5).
5. Go down the ranking. Add a position if its total effective coupling to the positions already chosen is at most θ.
6. Draw the route sample used for revealing and reveal the chosen set.
7. Update weights and resample exactly as in Algorithm 1.

Rules that keep this legal:
- A position forced by values drawn in this step waits for the next step. Revealing it in the same step drops the fresh/stale factor on the forced token, which is the correction of Example 1.1. Positions forced at the all-mask state are free. **[derived]**
- A particle that becomes fully revealed before the last step switches its twist to `Accepts` at that step. **[derived]**

### 3.4 Limiting cases

| Case | Behaviour |
|---|---|
| Mean-field model | All couplings are zero; everything is revealed in one step |
| No constraint | Reduces to a spacing-based parallel scheduler |
| Chain-like model | One position per gap, gaps halve; approaches bisection, about log2(L + 1) steps |
| Copy-like constraint | A pair is revealed together when the model does not couple its members |
| Counting-like constraint | Associations are weak; behaves almost as if unconstrained |
| Dense model dependence | Few positions pass; falls back to one or two per step |

### 3.5 Knobs

- **θ**: the single dial between number of steps and error. Tuned with the r measurement (section 4).
- **Priority score**: undecided. Candidates are most constraint-distorted first (KL(μ_i ‖ w_i), "bound-early"), most confident first (lowest entropy of μ_i), and random.
- **Pilot size m** and **number of extra route samples**.

## 4. Exactness

**What holds.** The policy depends only on the particle's own state, independent randomness and a frozen kernel, so it defines a fixed native law. The sampler is exact for that law given C in the sense of Theorem 4.2: Ẑ is unbiased and the particles are consistent as K grows. **[derived from notes]**

**What does not hold.** It is not exact for the one-at-a-time law. Neither is the notes' baseline, which co-reveals random positions with no independence check.

**Measuring the gap.** Let p be the one-at-a-time law and p̃ the policy's law. Define r(x) = log p(x) / p̃(x). It is the sum, over every co-revealed position after the first in its step, of the log-revision the model would have made to its prediction had the earlier positions been revealed first.

- Identity: KL(p(·|C) ‖ p̃(·|C)) = E[r] + log E[exp(−r)], expectations under p(·|C). **[checked]**
- Only the variation of r across valid outcomes costs accuracy. A constant r shifts Ẑ and nothing else.
- Bounds: KL ≤ osc²/8 and TV ≤ osc/4, with osc the range of r over C. For small r, KL ≈ Var(r)/2. **[checked: 35,261 random single-pair cases; worst ratios 0.999997 and 1.000000; KL/(Var/2) between 0.910 and 1.106]**
- Measuring r for a sample costs n − 1 extra denoiser calls for a step that revealed n positions; the calls batch.

**Three ways to pay for r.**

| Option | Result | Cost |
|---|---|---|
| Leave uncorrected | Exact for p̃ only | Bias, KL ≈ Var(r)/2 |
| Multiply weights by exp(r) | Exact for the one-at-a-time law | Verification calls; extra weight variance |
| Roll back rejected drafts | Exact for the one-at-a-time law | Verification calls; rejections |

The plan starts uncorrected and uses the r measurement to decide whether a correction is needed.

## 5. Expected gains

Simple cost model **[derived; ceilings, not predictions]**: if positions further apart than a range s are model-independent, a spaced schedule needs about s steps whatever the length.

| Setting in the notes | Assumption | Steps | Step speedup |
|---|---|---|---|
| L = 32, 16 steps | s = 4 | 4 | 4.0× |
| L = 32, 16 steps | s = 8 | 8 | 2.0× |
| L = 32, 16 steps | s = 16 | 16 | none |
| L = 64, 64 steps | s = 4 | 4 | 16.0× |
| L = 64, 64 steps | s = 8 | 8 | 8.0× |
| L = 64, 64 steps | s = 16 | 16 | 4.0× |
| L = 64, 64 steps | chain-like | 7 | 9.1× |

- The gain appears when the baseline step count exceeds the dependence range.
- On a grid, a spaced schedule with spacing s needs about s² steps, against s on a chain.
- The step saving is per particle. Particle count matters only for repaying the pilot: 8 to 32 particle-runs for an 8-trace pilot at these sizes.
- The particle count needed for a given accuracy is not expected to fall. **[conjecture]**
- The model omits long-range dependence, route overhead, and the longer effective range that a long-distance constraint creates through mixed paths.

## 6. Testbeds

Working assumption: most masked-diffusion tasks are roughly chain-structured (sequences) or grid-structured (images, some graphs), so the toys should cover both.

Recommended order:

1. **1-D oracle denoisers from known graphical models (no training).** The "denoiser" returns exact single-position conditionals computed by sum-product on a known model. Then the one-at-a-time law is the model's joint, the constrained ground truth is computable exactly at any length by sum-product on model × automaton, r is exact, and the true graph gives an oracle schedule to compare against.
   - Markov chain over classes: correlation length set by the transition matrix; revealed positions screen exactly.
   - Hidden Markov model: dependence decays with distance but a single revealed position does not screen.
   - Independent segments of unequal size: the block picture.
   - Mean-field: sanity check that one step is exact.
2. **Strip and grid oracle denoisers (Ising or Potts, no training).** Same idea, with exact conditionals from a transfer matrix over column configurations.
   - What it adds: separators that are whole rows or curves and are internally dependent; a known-graph step count that grows like the side length, about √L **[from memory]**; a temperature dial that changes correlation length on a fixed sparse graph; the two-colouring as a reference schedule (once one colour class is revealed, the other goes in one exact step).
   - Size limits: the transfer matrix costs 2^n per column for side n, comfortable up to roughly 16 × 16 (256 positions). A strip such as 8 × 128 gives 1,024 positions with 256 column states, and its width is a dial from chain-like to grid-like.
   - Constraints run over raster order: fixed magnetisation (counting), a count or parity per row (constraint blocks are rows while the model couples adjacent rows), row-equals-row copy. A fully 2-D local constraint needs a factor-graph route whose cost grows with the width.
   - Known gap: the pilot kernel of section 3.2 is 1-D and must be redesigned here.
3. **A small masked diffusion model trained on samples from stages 1 and 2.** Adds denoiser error and order inconsistency, and allows grids too large for the exact oracle. This is where the cluster GPU is needed.
4. **One real, non-text dataset.** Candidates, not yet vetted: structured records with schema and cross-field constraints; discretised images with 2-D locality; symbolic music.
5. **Text (OWT-130M).** Kept as a reference case where gains are expected to be small.

Constraint families for stages 1 to 3: copy, counting, a forbidden-motif regular language, Dyck-1.

## 7. Optimisations and alternatives

### 7.1 Constraint side

| Option | What it does | Status |
|---|---|---|
| Bound-early priority | Reveals constraint-bound positions first. Once the residual constraint is vacuous, every later potential is 1. Revealing all bound positions in one step gives constant weights but ignores all model dependence among them, so this is a dial between weight variance and bias. | Extreme case derived; best setting unknown |
| Explicit whole-block reveals | Reveal a constraint block in one step when its members are model-independent. The greedy rule should do this already. | Derived |
| Exact Hessian of log Mass | Replaces sampled associations with exact covariances; about one route pass per position. With random weights, the connected components of its nonzero pattern are the exact constraint blocks. | Block detection checked on 8 brute-force cases |
| Separator-seeking reveals | Prefer reveals that most reduce the largest block or the highest boundary-state entropy, so later steps can reveal more. | Idea only |
| Other priority scores | Commit lift log μ_i(σ) − log w_i(σ); joint constrained entropy log Z − Σ_i E_μ[log w_i] (exact routes). | Derived |

### 7.2 Correcting toward another law

| Option | What it does | Status |
|---|---|---|
| exp(r) weighting | Exact for the one-at-a-time law; verification calls batch, so it saves latency, not compute. | Derived |
| Rollback | Speculative sampling with the stale route as draft. Reproduces the one-at-a-time proposal with no added weight variance. Acceptance rate equals 1 − TV(stale, fresh), which makes it a direct measurement of re-conditioning strength. | Derived; unbiasedness when particles are at unequal progress needs a proof |
| Reweighting to the original native schedule | Potential gains a factor s/s̃. The new schedule must cover the original's support; mixing in a fraction ε of the original caps the factor at 1/ε. Under the fixed schedule the reveal count cannot change. | Derived |

### 7.3 Particle quality (independent of the schedule)

| Option | What it does | Status |
|---|---|---|
| Stratified first step | Systematic uniforms decoded through the route's top-down sampler, highest-entropy boundary state first; also within duplicate groups after resampling. Removes the between-stratum part of weight variance. Ẑ stays unbiased. | Derived; gain is a conjecture |
| Unequal stratum allocation | At least one particle per skeleton, with weight π(q)·K/n_q. Proposal-side analogue of defensive resampling. | Derived |
| Floor twist ψ = η + c | Bounds the over-confidence factor h/ψ by 1/c for every particle; tempering gives only an unbounded power. Set c near a low quantile of η from a pilot. | Derived; untested |
| Online cap | A particle's remaining contribution is at most W/ψ. Vacuous where twists are tiny. | Derived |
| Remedies already in the notes | Defensive resampling, tempering, calibration; no resampling on ambiguous routes. | Notes |

### 7.4 Estimating model dependence

| Option | What it does | Status |
|---|---|---|
| Richer kernel | Per-pair, or indexed by how much is revealed, in place of distance pooling. | Idea |
| Free probes | Every production step already yields before/after predictions; use them to refine the kernel across runs. | Idea |
| Cheap proxy model or attention maps | Predict dependence without denoiser calls. | Not analysed |
| Population-adaptive schedule | Adapt during a run from other particles. Requires the reweighting reading of 7.2. | Unbiasedness believed, not proved |
| Known graph | Minimum-height elimination tree, revealed root first; minimum steps equal the tree-depth. Useful as the oracle baseline on stage-1 toys. | Derived |

### 7.5 Larger changes, deferred

| Option | Assessment |
|---|---|
| Pairwise model inside the route | A chain-structured approximation of the model composes with an automaton route at a factor of the class count. The route would then sample all positions jointly from the approximate conditional, with weights correcting the approximation. Needs a source of pairwise terms. |
| Committing each particle to a route latent (for example a boundary state) | Valid, but gives up the route's ability to re-draw that latent under fresh predictions; expected to raise weight variance. Not analysed in detail. |
| Per-block recombination of particles | A recombined particle has no evaluated native path. Not worth pursuing now. |
| Look-ahead twists | Tested and dropped in the notes. |

## 8. Empirical checks, in suggested order

1. **Dependence range.** Fit the pilot kernel on each testbed. Is distance pooling adequate, or do residuals show structure?
2. **Mean-field sanity.** With an independent oracle denoiser, is one step exact on every constraint family?
3. **Baseline r.** Measure r for the notes' random schedule, so the new policy is compared against a known reference.
4. **θ sweep.** Steps against distance to the exact one-at-a-time conditional against weight variance.
5. **Priority score.** Bound-early against confidence-first against random, at equal steps.
6. **Copy family first.** Pair-joint reveals against the uniform schedule. This is the smallest test of the effective-coupling picture.
7. **Mixed-path term.** Does dropping the second term of E raise weight variance on constraints with long-distance coupling?
8. **Weight variance against step count.** Is it roughly unchanged when independent positions are co-revealed?
9. **Mutual information as a proxy.** Does small pilot mutual information predict small spread of r under a rare constraint?
10. **Association estimates.** How many route samples are needed; does the exact Hessian change the schedule?
11. **Forced positions.** Count them per family from the route alone; does revealing them early reduce twist over-confidence?
12. **Stratification.** Regress W on the step-0 latent in resampling-off runs; the explained fraction bounds the gain.
13. **Floor twist.** Hidden mass and Ẑ/h0 against c.
14. **Rollback acceptance rate**, as an instrument for how many positions per step a model tolerates.
15. **Grid temperature sweep.** Steps and error against temperature on a fixed grid, from short correlation length to near-critical.
16. **Two-colour reference.** Does the policy find the exact one-step reveal of the second colour class, and how does it compare with the nested-dissection oracle?
17. **Kernel form on grids.** Lattice-distance pooling against a per-pair kernel; what notion of screening is usable.

## 9. Proofs still owed

- Unbiasedness of Ẑ when resampling particles that are at unequal progress (rollback).
- Unbiasedness of Ẑ for a proposal schedule adapted from earlier steps of the same run.
- Accuracy of the first-order effective-coupling formula.

## 10. A point to fix in the notes

Definition 5.1 allows predictions to depend on which positions are masked. Prop 5.2 fails under it. Counterexample: L = 2, reveal position 1 then 2, C = {aa, bb}; position 2 is predicted (0.5, 0.5) from the all-mask state and (0.9, 0.1) once position 1 is revealed, whatever its value. Then G is 1.8 or 0.2, and step-exact gives aa with probability 0.5 against the native 0.9. The proposition needs predictions that are independent of both the revealed values and the mask set. **[derived]**

## 11. Supporting files

These are saved in the project artifact store, not in the repository.

- `parallel_reveal_bound_check_output.txt` and `parallel_reveal_bound_check.py`: check of the KL identity and bounds.
- `constraint_block_detection_check.txt`: brute-force check of block detection.
- `schedule_cost_model.txt`: cost-model table, including L = 256 and 1024.
