# Project status

**Updated:** 2026-11-05. **Repo:** `ksimplex/ConstrainedDecoding` (private), branch `main`, head `aeffd52` plus this
file. This is the single source of truth for decisions, results and next steps. Update it at every milestone.

Labels: **[verified]** = checked by tests or by data in a saved artifact; **[hypothesis]** = a working explanation that has
not been tested directly; **[citation unchecked]** = recalled, not yet verified against the source.

---

## 1. Goal and scope

Exact (trajectory-exact) constrained sampling for masked diffusion language models, for any constraint with a tractable
route (automata, grammar charts, factor graphs, spanning trees). It generalises TWISTER (arXiv:2609.35609) beyond DFAs.
Application stage: mRNA coding-sequence design with mRNAutilus (library generation, many diverse valid sequences).

The work is at a preliminary stage, to establish whether it gives useful results. A paper is the aim. Venue and timeline
will depend on progress over the next week. Writeups are by the user; the notes here aim at technical accuracy, progress
documentation and readability for collaborators, not submission. Math, algorithms and citations need a careful pass
before any paper.

## 2. Working conventions (agreed 2026-11-05)

- Replies that need input end with two numbered lists: (1) things needed from the user, (2) proposals awaiting approval.
- Ask clarifying questions early rather than proceed on assumptions.
- Numbers in chat are copied from saved artifacts; when chat and an artifact disagree, the artifact is authoritative.
- The user may step away; long waits on approvals should not block local work.

## 3. Code layout (three layers, docs/mrna_specs.md)

| layer | package | contents | depends on |
|---|---|---|---|
| 1 generic | `twistroute/` | routes, exact samplers, twisted SMC, calibrated twist, support guard, conjunctions, alphabet changes | nothing domain-specific |
| 2 domain | `twistroute_bio/` | codon alphabet, genetic code, codon lifting, synonymous masks, CDS constraint builders (in progress) | layer 1 |
| 3 model | `twistroute_bio/models/` (not started) | mRNAutilus adapter | layers 1-2 |

Tests: 191 passing at `aeffd52` (about 12 min on CPU). Each route is checked against brute-force enumeration; the sampler
is checked for E[Ẑ] = h₀.

## 4. Decisions

| date | decision |
|---|---|
| 2026-10-25 | Defensive resampling (ε = 0.25) is the default resampling-on arm; plain resampling and resampling-off remain controls. |
| 2026-11-04 | Monte Carlo look-ahead twists dropped (rounds 1-2): rollout noise widens the spread of log(ψ/h) more than it reduces bias. |
| 2026-11-04 | Calibrated twist `twist_temp=τ` kept as an option, not a default (helps copy strongly, mixed on regular). |
| 2026-11-04 | No further tuning of τ on toy cases; stress tests aim to break the method, not polish numbers. |
| 2026-11-04 | No resampling on ambiguous routes. An exact string-counting twist is #P-hard in general, so no exact fix exists. Fixed target structures (factor graphs) are unaffected. |
| 2026-11-04 | mRNA stage: library generation; coding sequences with mRNAutilus first. |
| 2026-11-04 | Layer 1 before layer 2; layer 2 builders contain no generic machinery. |

## 5. Results (all [verified] against the named artifact unless marked)

**Resampling shortfall (OWT-130M, TWISTER Fig. 1(a)).** Plain resampling gave Ẑ/h₀ = 0.930 at 100k runs. Defensive
resampling restores 0.998. Mechanism: an over-confident frozen twist puts mass on particles that resampling discards
("hidden mass"). `resampling_shortfall_diagnosis.json` (d984dee2-47c9-4bdc-b120-8567768a2342).

**Twist design (synthetic testbed).** `twist_design_findings.md` (72fceab9-450a-43c9-9fec-6cfe4fb86f72).
- Calibrated twist τ ≤ 0.35 removes the resampling penalty: plain resampling efficiency rises from 0.14 to 1.04-1.09.
- Hidden mass below 10⁻² falls from 7.8% to 0.15% of h₀.
- `twist_calibration_summary.csv` (c5a86446-bcbd-41fb-978e-27d8f9576a1d).

**Stress test A (OWT-130M, N = 32, T = 16, K = 16, τ = 0.5).** `e1_stress_summary.csv` (b476832c-4ca6-487e-8a4e-f12722818d71),
figure `stress_A_calibration.png` (fbd57892-9b32-4e16-bf32-bbd0d58b8b33). TV excess over the sampling floor:

| family | no resampling | + calibrated | defensive | + calibrated |
|---|---|---|---|---|
| copy | 0.136 | 0.068 | 0.148 | 0.040 |
| regular | 0.026 | 0.046 | 0.061 | 0.057 |
| Dyck-1 ambiguous | 0.086 | 0.091 | 0.143 | 0.158 |
| RNA pairing | 0.005 | 0.006 | 0.172 | 0.140 |

RNA pairing with plain resampling: 0.212. Why ambiguous grammars suffer under resampling: the derivation-weighted twist
misallocates particles [hypothesis].

**Stress test B, copy only (N = 64, T = 64, K = 8).** `stress_B_copy_summary.csv` (0733cb09-6f8a-41df-a492-4af3ee2b0f11).
- TV excess: no resampling 0.095, calibrated 0.056, defensive 0.104, defensive + calibrated 0.038.
- Defensive + calibrated has Ẑ/h₀ 0.902 (CI 0.827-0.983, k̂ 0.67): best samples, but a biased-low Ẑ at this K.
- Counting and RNA pairing crashed on a float32 underflow (fixed in `aeffd52`); rerun as job `232b232a`.

**E1 (all six families, N = 32, T = 16).** Complete. Rows are in `e1_stress_summary.csv`; earlier per-part tables are
`e1_A_summary.csv` (000d9d4c-9b76-476b-b4d5-8d3acb1fae20) and `e1_B_summary.csv` (0b88dc46-666a-451a-aa69-4b6cd8f788e4).

**E3 (synthetic, grammar ambiguity).** ESS fraction at K = 16 falls from 0.92 to 0.34 (ambiguous Dyck-1) and from 0.89
to 0.30 (RNA pairing) over L = 8-32. `e3_sweep_summary.csv` (eac54feb-17f6-4844-a707-5e0ee54e95d1).

## 6. Layer 1 additions (2026-11-04/05, all [verified] by tests)

- Span-bounded charts, with implied bounds inferred from the rules; 3.5x faster at L = 48 for a pair-span bound of 8.
- Exact class weights below the float floor. Previously floored at 1e-38, which biased weights in rare-event cases.
- Support guard (`guard="warn"|"raise"|"off"`): live particles must keep a finite clamped sum and twist.
- Conjunctions: automaton x automaton, automaton x grammar (Bar-Hillel), automaton x factor graph (augmented variables).
  Two grammars, or a grammar with a factor graph, are refused.
- Batched factor-graph route (variable elimination plus backward sampling); `factor_graph` specs.
- Alphabet changes: `lift_dfa` (inverse homomorphism), `minimize_dfa`, `symbol_classes`, `Masked` / `TorchMasked`
  (per-position allowed sets imposed as a constraint, not by renormalising predictions).
- Automaton forward pass in float64: float32 returned Z = 0 for a feasible count at log-mass -113.5.

## 6b. Layer 2 scorers and baselines (2026-11-05, done)

`twistroute_bio.score`: ViennaRNA scorers (MFE, structure distance, ensemble defect, fraction paired) for E6b
ranking, used only to score samples, never in a route; frequency-based and local-repair construction baselines
(the FactorDLM triviality guard). ViennaRNA 2.7.2, optional dependency `bio`. 7 tests [verified].

## 6c. Layer 3 native-process analysis (2026-11-05)

`docs/mrna_model_notes.md`. The mRNAutilus denoiser matches our `logits(x)` interface (SUBS-parameterised,
carry-over unmasking, no time conditioning); the net is RiNALMo (30-layer, 150M), loadable from the checkpoint,
self-contained but flash-attn dependent (GPU). Two decisions pending before building the adapter: (1) match the
model's independent-unmasking schedule vs our fixed-N/T (faithfulness; needs a layer-1 scheduler generalisation);
(2) run layer-3 on GPU with flash-attn. mRNAutilus's own conditional sampler imposes protein identity by masking,
the bias our method corrects; its MCTS sampler is the E6b mode-seeking comparator.

## 7. In progress

- **Step 2.2: CDS constraint builders, done (2026-11-05)** (`twistroute_bio/constraints`): motifs with IUPAC codes, GC band,
  target structure from dot-bracket, and `compile_cds(protein, constraints)`. Tested against independent nucleotide-level
  checks and enumeration [verified]. The automaton x structure product first needed 1,792^3 cells. It now enumerates
  only the feasible (codon, state) pairs per position, a generic layer-1 change in `DFAFactorProduct(allowed=...)`.
  With a 60-residue protein, 2 motifs + GC 40-60% + a 7-bp hairpin give domain 213, width 3, and compile in 0.5 s.
- **Job `232b232a`**: stress test B rerun (counting, RNA pairing at N = 64, T = 64).

## 8. Next steps

1. Layer 3 adapter, once the two decisions in `mrna_model_notes.md` section 8 are settled: vendor the RiNALMo
   net, load the checkpoint, wrap with SUBS and codon classes, smoke test.
2. If matching the model's schedule: generalise the reveal scheduler in layer 1 (independent per-position
   unmasking), tested the same way as the existing arms.
3. E6b best-of-K on mRNA, once layer 3 works.

## 9. Open questions for the user

- **Schedule faithfulness and GPU run for layer 3** (mrna_model_notes.md section 8). Blocks the adapter.
- Success thresholds: set per experiment just before running, with a variance argument (user, 2026-11-05).
- Licence: mRNAutilus CC BY-NC 4.0 (non-commercial); confirmed fine for research (user, 2026-11-05).

## 10. Key artifacts

| what | version id |
|---|---|
| notes (TeX / PDF, 20 pp, 17 refs) | 6652cbad-7e30-4af5-85af-11066c8721d5 / 1b920ed1-796e-431a-b5a3-73de434b39bc |
| experiment plan | c86c59db-6cf3-4cdc-bb43-42b224070c7d |
| mRNA specs | eaff9706-c362-4eda-aa86-1c1cbd6461b4 |
| code bundle (no .git) | ab6ff310-7b7a-4d5c-8fcd-0389f0d8f272 |
| mRNAutilus checkpoint | d4c53d94-b38a-488a-a043-48b2e0e40694 |
| E2 native pool, N = 32 | cd613b66-70fd-4d54-b727-ca399cff6c1e |

Cluster: MIT Engaging (`ssh:ondemand`), `mit_normal_gpu`, L40S, `--exclude=node3006`. After an app restart, SSH needs
re-authentication (Duo); a hung command usually means a pending prompt.
