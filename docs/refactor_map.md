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

## 3. Target layout

| Notes | New module | Contents (new names) | Built from |
|---|---|---|---|
| §1 model, mean-field prediction | `twistroute/models/` | `Denoiser` interface; `OracleHMM` (consistent); `LLaDA`; `MDLM`; `TokenClasses` | `models/*` (already here; renames only) |
| §1 unconstrained step, eq. (2), (4) | `twistroute/sampler.py` | `unconstrained_step`, `unconstrained_sample` | `decode/native.py` |
| §2.1 families, Table 1 | `twistroute/constraints/` one file per family: `automaton.py`, `positions.py` (subset of positions; per-position allowed sets), `factor_graph.py`, `grammar.py` | one class per family; cost from Table 1 in the docstring | `compile/*`, `torch_routes/*` |
| §2.1 intersection, Table 2 | `constraints/intersect.py` | `intersect(constraint, automaton)` | `compile/products.py` |
| §2.2 Proposition 1 | `constraints/base.py` | `Constraint` interface: `log_prob(q)` = log q(C), `marginals(q)`, `sample(q)` | `torch_routes/routes.py`, `decode/policy.constrained_marginals` |
| (tokens to classes, regular expressions) | `constraints/tokens.py` | `from_regex`, token classes of a pattern | `compile/tokens.py` |
| §2.3 twist, proposal, eq. (5) | `twistroute/twists/base.py` | `Twist` interface (`log_psi`, `propose`); `step_weight` | `decode/twist.py` |
| §2.3 ideal twist, eq. (7)–(8) | `twists/ideal.py` | `IdealTwist` (exact, small oracles only) | `decode/exact.py` |
| §2.3 product twist, eq. (9)–(10) | `twists/product.py` | `ProductTwist` | `decode/twist.py` |
| §2.3 chain twist, eq. (11)–(13) | `twists/chain.py`, `twists/pair_table.py` | `ChainTwist` (matching on by default); `counted_table`, `probed_table` | `decode/chain_twist.py`, `compile/tokens.pair_ratio_table` |
| §2.3 weighted sampler, eq. (6) | `twistroute/sampler.py`, `twistroute/resampling.py` | `weighted_sampler` (particles, resampling, returned sample); resampling rules | `decode/smc.py`, `decode/resampling.py` |
| §4.1 sequential schedules | `twistroute/schedules/sequential.py` | `Schedule` interface (`unmask(state)`); `LeftToRight`, `RandomOrder`, `Confidence`, `Sum` | `decode/policy.py`, `decode/schedule.py` |
| §4.2 parallel schedules | `schedules/parallel.py`, `schedules/dependence.py` | `FixedPerStep`, `Blocks`, `BudgetedCoReveal`; the pilot dependence estimate; later the planner | `decode/policy.py` |
| (not yet in the notes) evaluation | `twistroute/evaluation/` | exact laws and exact KL at small L; exact K-particle output law; bounds; spread of the estimate of P(C) | `decode/exact.py`, `diagnostics.py`, `parallel_exact.OutputLaw`, `kl_bounds_check.py` |

## 4. Vocabulary (code follows the notes)

| Today in the code | After | Notes |
|---|---|---|
| route, `TorchChain`, `TorchSubset`, ... | constraint, `AutomatonConstraint`, `PositionSubset`, ... | §2 |
| `route.logZ(lw)`, `constrained_marginals(route, lw)`, `route.sample(...)` | `constraint.log_prob(q)`, `.marginals(q)`, `.sample(q)` | Prop. 1 |
| `N` (length) | `L` | §1 |
| `den`, denoiser | `model` | §1 |
| native (`native_sample`) | unconstrained | §1 |
| policy, order, schedule (three words) | schedule | §1, §4 |
| reveal | unmask | §1 |
| `lw` (class log-weights) | `log_q` | Prop. 1 |
| `twisted_smc` | `weighted_sampler` | §2.3 |
| `logZ` returned by the sampler | `log_Z_hat` | Fig. 2 |
| `K` (particles) and `K` (pair table) | particles stay `K`; the pair table takes whatever letter the notes settle on | §2.3 uses K for both |

## 5. Where each sampler and schedule option goes

**New core** (in the notes, or decided): number of particles; twist (`ProductTwist` or `ChainTwist`; the proposal follows the twist; the chain proposal keeps a share of the product proposal, today `proposal_mix`);
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

## 6. Experiments

**`experiments/notes/`** (one script per item of the notes)

| Notes item | Script | Source today |
|---|---|---|
| Table 3, KL by particles, L = 8 | `table3_kl_by_particles.py` | `kl_bounds_check.py` |
| Figure 1, exact one-particle KL by order, product against chain, L = 10–12 | `fig1_exact_kl_by_order.py` | `best_plan_exact.py`, `surrogate_plan.py` |
| Figure 2, spread of log Z_hat at L = 256 | `fig2_spread_L256.py` | `compare.py --arms matched` (pulib) |
| Figure 3 (no caption yet), the same with all orders and four other chains | `fig3_spread_other_chains.py` | same |

**`experiments/benchmarks/`** (real models): `json_mode/` (`json_mode_compile.py`, `json_classes.py`, `json_mode_classes.py`), `bfcl/` (`bfcl_constraints.py`, `bfcl_compile.py`, `bfcl_score.py`, `bfcl_eval_outputs.py`), `xlam/` (`xlam_constraints.py`),
`run_real_model.py` (today `llada_json.py`, split into stages), `llada_pilot_check.py`, `mdlm/` (`mdlm_first_look.py`, `mdlm_order_consistency.py`, `mdlm_orders_output.py`, `mdlm_timing.py`).

**`experiments/studies/`** (live, not yet in the notes): `parallel_exact.py`, `parallel_exact_check.py`, `particle_rule_check.py`, `forced_batch_check.py`, `hidden_mass_check.py`, `compare.py` with `pulib/` (the oracle comparison harness), `report.py`.

**`experiments/archive/`** (finished one-off studies, with `INDEX.md`: script, status-document section, verdict; they run against the legacy surface): `collective_check.py`, `control_check.py`, `counting_orders.py`, `error_estimate_check.py`,
`exposure_check.py`, `first_look.py`, `order_attribution.py`, `pair_cost_check.py`, `plan_check.py`, `regularity_check.py`, `reveal_stages.py`, `revision_check.py`, `twist_error.py`, `twist_order.py`, `twist_smc.py`, `weight_trust.py`.

**Tests**: existing tests stay and keep testing the legacy surface. New tests, one file per new module, named after the notes (`test_constraints_prop1.py`, `test_twists_weights.py`, `test_sampler.py`, `test_schedules.py`, `test_evaluation.py`), plus `test_core_matches_legacy.py`.

**Style for everything that stays live**: one statement per line, lines of at most 100 characters, type hints on public functions, no path set-up lines (the package is installed in editable mode).

## 7. Order of work and checks

1. Tag, tarball, branch. Freeze the reference set: the 15 cells of `tests/test_pu_regression.py`, plus matched chain, forced batch, confidence schedule, and the exact tables behind Table 3 and Figure 1.
2. `constraints/` (interface of Proposition 1 over the existing implementations) with its tests.
3. `twists/` and `sampler.py`, checked against the legacy sampler on the reference set.
4. `schedules/` and `evaluation/`.
5. `experiments/notes/` scripts; each reproduces its table or figure from the new core.
6. Move and reformat the live experiments; archive the rest with the index.
7. A fresh agent gets only the notes and the repository, explains how eq. (13) is computed, and reproduces Table 3. Its points of confusion become the last round of fixes.

## 8. Open

1. "Tempered twist as default": the sampler has two temperings. `twist_temp` (tau) computes the constraint probability under marginals raised to the power tau, and was run at tau = 0.5 in the earlier studies
   (`docs/twist_design_findings.md`); `twist_power` (beta) raises the whole intermediate twist to a power. Which one, and at what value?
2. Neither tempering is implemented for the chain twist: the sampler refuses the combination. With the chain twist as the proposal of choice, a default tempering needs a definition for it (for example the chain built on
   tempered marginals), a proof of unbiasedness and a check that it helps. That is method work, not restructuring.
3. The default changes numbers, so it goes in as its own commit after the new core has reproduced the reference with tempering off.

## 9. Progress

- 2026-11-20, step 1 done: tag `pre-notes-refactor` (commit `bd969a2`), branch `notes-structure`, tarball of the tagged tree, and the frozen sampler reference
  (`tests/reference/sampler_reference.npz`: 150 cases = 6 oracle cells x 25 combinations of twist, schedule and resampling that stay in the core; regenerates bit for bit).
