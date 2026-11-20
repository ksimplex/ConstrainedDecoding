# Refactor map: code that follows the notes

Agreed 2026-11-20 (decisions in section 5; open points in section 8). Progress in section 9. "Notes" = `Constrained_Diffusion.pdf` (version of 2026-11-20).
Out of scope and untouched: the directories marked off-limits, and `experiments/*.py` outside `experiments/parallel_unmasking/`.

## 1. Rules

1. Every object the notes define has one name in the code, and its docstring cites the equation or proposition.
2. Every table and figure of the notes has one script under `experiments/notes/` that regenerates it.
3. The code uses the notes' words (section 4 below).
4. Nothing that can be imported today stops working (section 2).
5. The refactor changes no numbers: a frozen reference set is reproduced exactly before and after (section 7).

## 2. Compatibility and backup: one mechanism for both

- Tag `pre-notes-refactor` on the current head of `parallel-unmasking`; a tarball of that tree saved as an artifact; the work happens on a new branch `notes-structure`.
- **The existing packages keep their paths and behaviour**: `twistroute.compile`, `twistroute.torch_routes`, `twistroute.decode`, `twistroute.models`, `twistroute.diagnostics`, `twistroute.experiment`. They become the *legacy surface*: frozen, no new features.
  `twisted_smc` stays there with all 29 arguments. That is the backup of the ruled-out options, and it is what the code elsewhere in the repository calls (it uses `ess_min`, `defensive`, `mix_native`, `twist_temp`, the look-ahead options and `schedule`).
- **The new core is added beside it** under the notes' names. Where the existing code is already right (the constraint implementations) the new module wraps or re-exports it; where it is tangled (the sampler, the schedules) the new module is a short fresh implementation.
- A test asserts that the new core and the legacy surface give identical results on the reference set.
- When nothing imports the legacy surface any more, it is deleted in one commit.

## 3. Layout as built

This section, section 4 and section 6 describe the code as it is (rewritten 2026-11-20 after the readability test, which found
that the earlier text was the plan and named things that were never built). What was planned and not done is listed at the end
of this section.

| Notes | Module | What is in it | Wraps or ports |
|---|---|---|---|
| §1 model | `twistroute/models/` | `noisy_hmm`, `lazy_chain`, ... (oracles; `OracleHMM`), `LLaDA`, `MDLM`, `TokenClasses`. A model is any object with `.logits(tokens)` (log-probabilities, despite the name), `.mask_id`; oracles also have `.S` (tokens), `.log_joint` | unchanged earlier code and names |
| §1 mean-field prediction, eq. (2) | `twistroute/state.py` | `State` (x_t and the prediction at it; `State.prediction()`), `Step` (x_t to x_s) | new |
| §1, eq. (4) unconstrained step | `twistroute/sampler.py` | `unconstrained_step`, `model_prediction`, `commit` | new |
| §2.1 families, Table 1 | `twistroute/constraints/`: `automaton.py`, `positions.py`, `factor_graph.py`, `grammar.py` | functions that return a `Constraint`: `automaton`, `forbidden_motifs`, `exactly_k`, `at_least_k`, `dyck1`; `on_positions`, `with_allowed_sets`; `factor_graph`, `copy_pairs`; `grammar` | wraps `torch_routes`, `compile` |
| §2.1 intersections, Table 2 | `constraints/intersect.py` | `intersect` | wraps `compile/products.py` |
| §2.2 Proposition 1 | `constraints/base.py` | `Constraint`: `log_prob`, `marginals`, `sample`, `contains`; `fix_unmasked` | wraps the `.logZ`, `.sample`, `.accepts` of a "route" |
| (tokens to classes) | `constraints/tokens.py` | `from_regex`, `token_codes` | wraps `compile/tokens.py` |
| §2.3 twist, proposal, eq. (5) | `twists/base.py` | `Twist` (`log_psi`, `draw_completion`, `log_weight`, `class_marginals`); `weight_from_ratio` is eq. (5); `drawn_log_prob` | port of `decode/twist.py` |
| §2.3 product twist, eq. (9), (10) | `twists/product.py` | `ProductTwist(temperature, power)`; `log_product_twist` | port of `decode/twist.py` |
| §2.3 chain twist, eq. (11) to (13) | `twists/chain.py`, `twists/chain_law.py` | `ChainTwist` (twist, proposal; `log_weight` is eq. 13); `ChainLaw` (`transitions`, `log_mass`, `log_mass_with`, `sample`, `marginals`) | port of `decode/chain_twist.py` |
| pair table K | `twists/tables.py` | `counted_pair_table`, `probed_pair_table` | wraps `decode/chain_twist.py` |
| §2.3 weighted sampler, eq. (6) | `twistroute/sampler.py`, `twistroute/resampling.py` | `weighted_sampler`, `SamplerResult` (`.returned()` draws the output); `Resampling`, `resample` | port of `decode/smc.py`, `decode/resampling.py` |
| §4.1 sequential schedules | `schedules/base.py`, `schedules/sequential.py` | `Schedule` (`unmask`), `RandomOrder`; `LeftToRight`, `FixedOrder`, `Confidence`, `Sum` | ports of `decode/policy.py`, `decode/schedule.py` |
| §4.2 parallel schedules | `schedules/parallel.py`, `schedules/dependence.py`, `schedules/baselines.py` | `Blocks`, `BudgetedCoReveal` (still calls `decode.policy.CouplingPolicy`); `fit_distance_dependence`; `Dilated` | wraps `decode/policy.py` |
| Table 3, Figures 1 and 2 | `twistroute/evaluation/` | `exact.py`: `target_law`, `one_particle_law`, `output_law`, `kl_of_output`, `kl`; `bounds.py`: `upper_bound`, `leading_term`, `marginal_lower_bound`; `spread.py` | new, after `kl_bounds_check.py` |

Planned and not built: an `IdealTwist` (eq. 7 and 8; the exact twist exists only in `experiments/studies/twist_error.py`); a
`Denoiser` interface class; `unconstrained_sample` (a full unconstrained run; `decode/native.py` has it); a `FixedPerStep`
schedule (use `RandomOrder` with fewer steps than positions); the diagnostics callback is the `observe` argument of
`weighted_sampler`, not a module in `evaluation/`.

Where the code differs from the notes: section 8, item 5, and the module docstring of `twists/chain.py`.

## 4. Vocabulary

| Notes | New code | Earlier code (still used in `decode`, `torch_routes`, `models`, experiments) |
|---|---|---|
| constraint C | `Constraint` | route (`TorchChain`, `TorchSubset`, ...; `TorchChain` is the automaton, not the chain twist) |
| q(C); constrained marginals; a draw | `constraint.log_prob`, `.marginals`, `.sample` | `route.logZ`, `constrained_marginals(route, .)`, `route.sample` |
| L | `length` | `N` |
| model, p_theta | `model` | `den`, denoiser |
| mean-field prediction, class level | `State.log_q`, `State.prediction()` | `lw`, `clamp_revealed(lw, ...)` |
| unconstrained | `unconstrained_step` | native |
| schedule pi(U \| x_t) | `Schedule.unmask` | policy, order, `reveal` |
| x-hat_0; commit on U | `completion`; `commit` | `y`; inside `_propose` |
| twist psi; step weight w | `log_psi`; `log_weight`, `weight_from_ratio` | `psi`, `eta`; `_log_potential` |
| chain prediction r; r(C \| x_t) | `ChainLaw` ("chain law"); `log_mass` | `ChainTwist.log_chain` |
| pair table K | `pair_table` | `kernel` |
| particles K | `particles` | `K` |
| weighted sampler | `weighted_sampler` | `twisted_smc` |
| estimate of P(C) | `log_Z_hat` | `logZ` |
| Dang and Ermon's sampler | one particle ("Unweighted" in Table 3) | `K=1` |

Two classes are called `ChainTwist`: `twistroute.twists.ChainTwist` (new; matching on by default) and
`twistroute.decode.ChainTwist` (earlier; `match=0` by default, argument names `kernel`, `eps`, `temp`, `match`).

## 5. Where each sampler and schedule option goes

**New core** (in the notes, or decided): number of particles; twist (`ProductTwist` or `ChainTwist`; the proposal follows the twist; the chain proposal can keep a share of the product proposal, `proposal_product_share` (earlier `proposal_mix`), off by default);
schedule; resampling threshold (`ess_min`); schedules left to right, random, confidence (token, class, entropy; blocks; several per step), `sum`; budgeted co-reveal; forced positions first; forced-only batch (off by default).

**Legacy surface only** (ruled out or superseded; still runnable through the old `twisted_smc` and `CouplingPolicy`): look-ahead (`lookahead*`), `mix_native`, the unmatched chain as reference for co-revealed sets (`reference_twist` with the ratio formula),
threshold admission (`theta`), the mixed coupling, the guard, the dynamic "least-coupled" orders, the separator and bound priorities, the thick-separator schedule, the resampling gate.

**Decided 2026-11-20**

| Option | Decision |
|---|---|
| Hidden-mass remedies | All three stay available in the core: the tempered twist, resampling in proportion to a power of the weight, the defensive mixture at resampling. The tempered twist is the default. Which tempering and at what value: open, see section 8. |
| `lex` order | Legacy surface only; `sum` is the order. |
| Pull-in of forced partners | Stays in the core. |
| `DilatedSchedule` (DUS baseline) | Stays, in `schedules/baselines.py`. |
| Diagnostics threaded through the sampler | One optional callback, defined in `evaluation/`. |
| Constraint implementations | Wrapped behind the Proposition 1 interface; the code itself is not moved. |

## 6. Experiments and tests as built

`experiments/INDEX.md` lists every script, what it does and the status sections that record it.

- `experiments/notes/`: `cells.py` (the oracle and constraints of the notes' experiments), `table3.py` (Table 3). Figures 1 to 3
  have no script: Figure 1 comes from `experiments/studies/surrogate_plan.py` (with the unmatched chain), Figures 2 and 3 from
  `experiments/studies/compare.py --arms matched` and plotting code that is not in the repository.
- `experiments/benchmarks/` (flat, one folder): the JSON-Mode-Eval, BFCL, xLAM and MDLM scripts; `llada_json.py` is the runner for
  LLaDA. They call the earlier sampler.
- `experiments/studies/`: `compare.py` with `pulib/`, `twist_error.py`, `surrogate_plan.py`, `parallel_exact.py`,
  `parallel_exact_check.py`, `kl_bounds_check.py`, `particle_rule_check.py`, `forced_batch_check.py`, `report.py`, `jobs/`.
- `experiments/archive/`: the answered questions.
- Tests: `tests/core` (new code: `test_constraints_prop1.py`, `test_sampler.py` with the chain prediction and the chain weight
  against enumeration, `test_evaluation.py`, `test_core_matches_legacy.py` with the frozen reference in `reference/`),
  `tests/algorithms`, `tests/legacy`, `tests/experiments`. The new schedule classes have no test file of their own: they are
  covered by the exact comparison with the earlier sampler.
- Style of the new code: one statement per line, at most 100 characters, type hints (`ruff`, settings in `pyproject.toml`).
  Scripts and some tests still add folders to the path by hand.

## 7. Order of work and checks

1. Tag, tarball, branch. Freeze the reference set: the 15 cells of `tests/test_pu_regression.py`, plus matched chain, forced batch, confidence schedule, and the exact tables behind Table 3 and Figure 1.
2. `constraints/` (interface of Proposition 1 over the existing implementations) with its tests.
3. `twists/` and `sampler.py`, checked against the legacy sampler on the reference set.
4. `schedules/` and `evaluation/`.
5. `experiments/notes/` scripts; each reproduces its table or figure from the new core.
6. Move and reformat the live experiments; archive the rest with the index.
7. A fresh agent gets only the notes and the repository, explains how eq. (13) is computed, and reproduces Table 3. Its points of confusion become the last round of fixes.

## 8. Decisions on the hidden-mass options, and what is still open

1. The core keeps all three remedies: tempered marginals in the product twist (`ProductTwist(temperature=)`), a power of the
   twist (`power=`), and the two resampling remedies (`Resampling(defensive=, weight_power=)`).
2. Decided (user, 2026-11-20): the default is tempered marginals with the product twist only. The value was given as 0.5.
   **Open again:** section 36u of the status document compared 0.7 and 0.5 at 256 positions; 0.7 is clearly better on one case
   (forbidden, one token per step: 1.39 | 0.56 against 1.92 | 1.03, RMS error of log Z_hat at 16 | 64 particles) and within
   error elsewhere. The value is to be confirmed when the default changes.
3. Correction to an earlier version of this section: the chain twist does have a tempering option (`ChainTwist(temperature=)`,
   the chain inside the twist built on tempered marginals; the proposal is not tempered). Section 36u tested it as a twist with
   the product proposal and found no gain under defensive resampling. It stays off by default. Tempering the chain proposal is
   not defined.
4. The default changes numbers, so it goes in as its own commit after everything else. Until then every default reproduces
   the frozen reference.
5. Two points where the code differs from the notes, kept and documented, for the user to settle: the chain twist mixes in a
   share 0.05 of the product twist (`product_share`); the sampler starts every weight at the twist of the fully masked
   sequence rather than at 1, so that the mean weight estimates P(C).

## 9. Progress

- Step 1 done: tag `pre-notes-refactor`, branch `notes-structure`, tarball, frozen reference (now `tests/core/reference/`, 150 cases).
- Step 2 done (`316c8ef`): `constraints/`, checked by enumeration. The grammar and factor-graph passes are single precision
  (exact to about 1e-7); the automaton families are double precision.
- Step 3 done (`01864be`): `state.py`, `twists/`, `resampling.py`, `sampler.py`. The frozen reference is reproduced exactly;
  six further option combinations match the old sampler side by side; the chain law is ported and checked by enumeration.
- Step 4, schedules done (`abeca65`): `schedules/`. `BudgetedCoReveal` still calls `decode.policy.CouplingPolicy`; it will be
  replaced when section 4 of the notes (the parallel schedule) is settled. Evaluation in progress: `evaluation/`.
- Step 5, Table 3 done (`77b28ea`): `experiments/notes/table3.py` (24 of 27 entries identical to the notes; 3 differ by one in
  the last digit, Monte Carlo over the other particles' weights). Figures 1 to 3 are not ported: Figure 1 needs the dynamic
  programme over all states in `experiments/studies/surrogate_plan.py`; the user said not to spend time on the figures.
- Step 6 done: `experiments/parallel_unmasking/` split into `experiments/benchmarks/`, `studies/` and `archive/`, mapped in
  `experiments/INDEX.md`. Of the 40 scripts, 36 start (`--help`), two more import cleanly, and two that run at import were checked statically; every
  cross-folder import resolves. The benchmark scripts still call the
  earlier sampler; porting them needs one GPU job to verify and is the user's call.
- Tests sorted into `tests/core`, `tests/algorithms`, `tests/legacy`, `tests/experiments` (none deleted; `tests/conftest.py`
  and the application's tests are where they were). `docs/setup.md` written. README: waiting for the user.
- Not checked, because those files are not opened: whether application code imports from the old folder
  `experiments/parallel_unmasking/` or from test modules that moved. Under pytest the test folders are all on the path
  (`pyproject.toml`), so imports between test modules keep working.
- Step 7, readability test done (report saved as an artifact, `readability_report.md`). Fixed after it: the chain equations
  were numbered wrongly in the docstrings (now (11) chain prediction, (12) constrained chain prediction, (13) weight); the
  frozen reference `.npz` had never been committed (`*.npz` was ignored); `ChainTwist.log_weight` is tested directly against
  eq. (13); the K-particle law is exact where feasible and reports its Monte Carlo error otherwise; `product_share = 0` (the
  notes' twist) is allowed; twists no longer need to fill `state.cache`; `Twist.complete` renamed `draw_completion`; this map
  rewritten to describe what was built.
- Added by the user on 2026-11-20, after step 6: `docs/setup.md` (what everything needs; what only the real-model
  benchmarks need, with where the weights and datasets come from; what needs a GPU); tests sorted into folders that follow
  the new layout (none deleted; the application's tests are not touched); the README rewritten for the new layout.
- Remaining helpers the new core imports from the earlier packages: `decode.policy` (dependence estimate, `CouplingPolicy`
  behind `BudgetedCoReveal`), `decode.chain_twist` (the two pair-table fits), `torch_routes` and `compile` (the constraint
  algorithms, wrapped on purpose), `models`.
