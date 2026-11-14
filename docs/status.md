# Project status

**Updated:** 2026-11-06. **Repo:** `ksimplex/ConstrainedDecoding` (private), branch `main`, head `b78b8e9`. This is the single source of truth for decisions, results and next steps. Update it at every milestone.

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
- **Check the artifact store before concluding anything is lost.** `scripts/check_durability.py` reports
  which files under `results/` and `docs/` are in neither git-pushed state nor the artifact store; run it
  before ending a session *and* before claiming a file is missing. On 2026-11-13 two result files were
  declared lost and retrieved over SSH from cluster scratch when both had been artifacts for six days,
  byte-identical. The store is the authority on what exists; a recollection of what was saved is not. The
  store half of the check needs `host.query`, so run it from the repl tool.
- **Commit AND push before any long wait.** The workspace is cleared when a session sits idle. On
  2026-11-07 it was wiped with two commits unpushed; the results survived only because they were artifacts,
  and the code had to be reconstructed. An artifact is durable, a workspace file is not, and a local commit
  is not either. Anything worth keeping is either pushed or saved with `save_artifacts` before a wait.
- **Pin experiment inputs by identifier, not by query.** The gene panel was originally built by an NCBI
  `esearch`, which returns a different gene set on a different day and different transcript isoforms for the
  same symbol. `experiments/fetch_human_cds.py` now pins by (symbol, CDS length) and records accessions into
  `results/bias_cds.json`; prefer the accessions once they exist.
- **Before editing a shared document, check its artifact version history, not just the local file.** Sessions
  run in parallel frames against one artifact store. On 2026-11-06 two frames edited `twistroute_notes.tex`
  at once: frame `cda5a5e2` saved a clarity rewrite as v10/v11, this frame then saved v12 from a stale local
  copy and discarded it, and `cda5a5e2` reconciled at v13. Nothing was lost in the end, but only because the
  other frame merged by hand. The check is cheap:
  `host.query("SELECT version_number, id, frame_id FROM artifact_versions WHERE artifact_id=? ORDER BY version_number DESC", [aid])`
  — if the latest version came from another frame, start from it rather than from the working copy, and pass
  that version id as `version_of`. Note also that git and the artifact store drift independently: the other
  frame saved artifacts without pushing, so git sat two versions behind the canonical document.

## 3. Code layout (three layers, docs/mrna_specs.md)

| layer | package | contents | depends on |
|---|---|---|---|
| 1 generic | `twistroute/` | routes, exact samplers, twisted SMC, calibrated twist, support guard, conjunctions, alphabet changes | nothing domain-specific |
| 2 domain | `twistroute_bio/` | codon alphabet, genetic code, codon lifting, synonymous masks, CDS constraint builders (in progress) | layer 1 |
| 3 model | `twistroute_bio/models/` | vendored RiNALMo encoder (done, validated); mRNAutilus denoiser adapter (next) | layers 1-2 |

Tests: 236 passing at `b78b8e9` (about 13 min on CPU). Each route is checked against brute-force enumeration; the sampler
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
| 2026-11-05 | Match mRNAutilus's independent-unmasking schedule for layer-3 runs. Default stays `"fixed"` and bit-identical. |
| 2026-11-05 | Protein identity enters through the route (per-position allowed-codon masks), never by renormalising model predictions. |
| 2026-11-06 | mRNA headline shifts to constrained families (target structure, GC windows, boundary-straddling motifs, conjunctions), where rule-based construction fails outright. Plain codon optimisation is demoted to a sanity check, not a claim. **Amended 2026-11-13: "fails outright" is disproven (Sec. 7b) -- DNA Chisel solves every conjunction tested. The constrained families stay the headline, but on distributional grounds (they are not samplers), not feasibility.** |

## 5. Results (all [verified] against the named artifact unless marked)

**Resampling shortfall (OWT-130M, TWISTER Fig. 1(a)).** Plain resampling gave Ẑ/h₀ = 0.930 at 100k runs. Defensive
resampling restores 0.998. Mechanism: an over-confident frozen twist puts mass on particles that resampling discards
("hidden mass"). `resampling_shortfall_diagnosis.json` (d984dee2-47c9-4bdc-b120-8567768a2342).

**Twist design (synthetic testbed).** `twist_design_findings.md` (72fceab9-450a-43c9-9fec-6cfe4fb86f72).
- Calibrated twist τ ≤ 0.35 removes the resampling penalty: plain resampling efficiency rises from 0.14 to 1.04-1.09.
- Hidden mass below 10⁻² falls from 7.8% to 0.15% of h₀.
- `twist_calibration_summary.csv` (c5a86446-bcbd-41fb-978e-27d8f9576a1d).
- **Correction (2026-11-05, commit `3720c1a`).** An earlier claim that calibration leaves the resampling-off estimator
  untouched is false. A K = 1 trace decomposition showed summed per-step log-weight variance falling 11.1 -> 6.0
  (tau 1 -> 0.5) while total path-weight sd barely moved (3.278 -> 3.298) and rel. s.e. of Ẑ was flat (0.0128 -> 0.0132).
  A cross-family check at K = 16 with resampling off gave rel. s.e. ratios (tau 0.5 vs 1) of: prefix 1.02, copy 1.17
  (worse), counting 0.30 (3.3x better), forbidden 0.66. The effect on the resampling-off estimator is family-dependent
  and is **not** mediated by smoother per-step increments. The mechanism where it helps is not isolated [open].

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
- Counting and RNA pairing rerun, `stress_B2_summary.csv` (0c481b9c-99d4-4532-bf5a-bc133e30bfa5). Counting: TV excess
  ~0 on every arm, Ẑ/h₀ 0.99-1.18. RNA pairing: TV excess 0.109 off (Ẑ 1.105), 0.043 defensive but **Ẑ 0.576**
  (CI 0.359-0.858, k̂ 1.00), 0.142 off+calibrated, 0.075 defensive+calibrated. h₀ = 0.01865 (counting), 0.01559 (RNA).
  The defensive RNA arm was previously recorded here as "the clearest remaining failure".

  **Resolved 2026-11-13: it is not a defect, it is the ambiguity boundary, measured.** Reading the
  `resamplings` column of `stress_B2_summary.csv` (how often resampling actually triggered per run) splits the
  RNA-pairing arms exactly along the no-resampling rule, and nothing else does:

  | arm | resamplings/run | Ẑ/h₀ | 95% CI | k̂ |
  |---|---|---|---|---|
  | K8_off (`ess_min` 0, `defensive` 0) | 0.0 | 1.105 | 0.765-1.612 | 0.53 |
  | K8_off_cal50 | 0.0 | **1.061** | 0.925-1.211 | 0.36 |
  | K8_def (`ess_min` 0.5, `defensive` 0.25) | 11.25 | **0.576** | 0.359-0.858 | 1.00 |
  | K8_def_cal50 | 10.34 | 0.872 | 0.410-1.664 | 0.81 |

  Every sound arm has resampling OFF; every biased one has it ON, on an ambiguous route -- the configuration
  our own rule forbids, because the string-counting twist a correct resampling step would need is #P-hard.
  Three things make this a prediction borne out rather than a coincidence. The k̂ diagnostic tracks it
  monotonically (0.36/0.53 off, 0.81/1.00 on), and k̂ >= 1 is the formal signature of an importance-weight
  distribution with no finite variance. The effect is specific to ambiguity: on `counting`, an UNAMBIGUOUS
  route, K8_def resamples 7.2 times per run and still returns 1.176 with k̂ 0.89 -- degraded, not broken. And
  resampling-off on the same family, same K, is healthy at 1.061 with a tight CI.

  **Keep this result in the paper rather than dropping it.** It is the empirical face of the scope condition:
  the method is exact for routes whose accepting computations biject with strings (unambiguous automata and
  grammars, factor graphs, spanning trees), and here is what breaks when you leave that frontier, with k̂ as
  the detector. A reviewer who suspects the no-resampling rule is a convenient dodge is answered by the row
  where we violated it. **Nothing here touches the unambiguous results, and nothing touches mRNA:** target
  structures compile to explicit pair factors in a factor graph, one accepting computation per assignment.

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

## 6c. Layer 3: schedule, encoder, and model validation (2026-11-05/06)

**Scheduler (commit `1530c36`).** mRNAutilus unmasks each position independently with probability 1/(T-t), not a fixed
count per step. These are genuinely different processes: with the same model and constraint, h₀ differs by 4-10
standard errors between the two schedules (copy 10.5 s.e., forbidden 9.4 s.e., exactly_k 3.9 s.e. at n = 300k)
[verified]. Layer 1 now takes `schedule="fixed"|"independent"`; `reveal_indices` returns `(indices, valid)` with padded
columns for the ragged independent case. The reveal draw is kept in its original position in the RNG stream, so the
fixed schedule remains bit-identical to all earlier results [verified by golden regression tests].

**Vendored encoder (commit `1530c36`).** `twistroute_bio/models/_rinalmo/net.py` is a self-contained RiNALMo "mega"
(30 blocks, 640-dim, 20 heads, rotary, SwiGLU, fused Wqkv, no QKV bias; 148.1M parameters) with `impl="torch"|"flash"`.
It avoids a hard flash-attn dependency, so layer 3 runs on CPU for development and on GPU for production.

**Validation** (`docs/mrna_model_validation.md`, fb566ef8-6768-4554-ac5b-62be50969a3f). Summary:

- The vendored transformer block is **bit-identical** (max abs diff 0.0) to upstream's own non-flash block on identical
  inputs with mapped weights [verified]. The checkpoint loads with 399 tensors and no missing or unexpected keys.
- Two real bugs were found and fixed in the process: the **SwiGLU gate order was reversed** (upstream applies Swish to
  `linear` and multiplies by `linear_gate`), and **TokenDropout was missing** (parameter-free; zeroes masked embeddings
  and rescales by (1-0.12)/(1-observed); measured 0.061 vs 0.000 at 5% masking).
- Alphabet confirmed: V = 86 = 5 special (`<cls>`=0, `<pad>`=1, `<eos>`=2, `<unk>`=3, `<mask>`=4) + 64 codons at
  indices 5-68 + 17 nucleotide codes at 69-85. Note the model's codon order is TCAG-based and differs from
  `twistroute_bio.alphabet.CODONS`, which is lexicographic ACGT; the adapter must map between them.
- **Synonymous-codon top-1 (n = 548 paired, one CDS):** model 0.398, frequency baseline 0.387, chance 0.295.
  McNemar z = +0.53, not significant. **SUPERSEDED -- this protocol masks 20 positions at once**, so each
  prediction conditions on a sequence with 20 holes, which is not the setting the sampler runs in.
- **Codon choice under full conditioning (2026-11-13, `results/codon_scoring.csv`).** One position masked at a
  time with every other position revealed, scoring the log-probability assigned to the OBSERVED codon (a proper
  scoring rule; top-1 is not, since a natural CDS is one draw from a high-entropy conditional and even a perfect
  model would often rank it second). n = 307 positions, 40 each in 8 genes, both predictors renormalised over
  the same synonymous set: **mean gain +0.164 nats** (95% CI +0.100 to +0.229), paired t = 4.97 (p = 1.1e-6),
  Wilcoxon p = 3.4e-11, closer at 229/307 positions, **positive in 8/8 genes**, perplexity 2.465 vs 2.905.
  Top-1 on these same positions: **64.2% vs 53.1%, McNemar z = 4.49**. Same model, same metric as the row above;
  the entire difference is how much context was revealed. A pseudo-likelihood, not a joint (MDLMs give an ELBO).
  **So the model's conditional does carry information about codon choice beyond a lookup table**, and the earlier
  negative read was a low-context artefact rather than a property of the model.
- **But its joint does not factor.** TV of the predicted codon distribution against the real-context prediction:
  one distant codon flipped 0.002-0.006; whole context randomised 0.336; synonymous shuffle with protein preserved
  **0.261**.
- **Re-conditioning strength** (section 6 of the validation note). Holding one codon masked and revealing the rest in
  four stages, the prediction at the held-out position moves by TV 0.337 end to end (0.205 restricted to the
  synonymous set), with the largest single movement, 0.291, in the first quarter of reveals. Mean-field over the
  trajectory would give 0 in every row. n = 6 positions on one sequence: an order-of-magnitude indication only.

**Why re-conditioning is the quantity that matters.** If a model's predictions did not move as positions are revealed,
the frozen twist would equal the true twist, step-exact decoding would already be unbiased, and a single route sample
would draw from p(x | C) exactly — the entire particle apparatus would be unnecessary. The value of this method scales
with re-conditioning strength. The numbers above put mRNAutilus in the regime where there is bias to correct, at
roughly the magnitude that produces the 0.13-0.17 TV excess we measured for step-exact decoding on OWT-130M.

**Open caveats on the validation.** One sequence only (the `fluc` template from mRNAutilus's own `config.yaml`,
possibly in its training data); perturbed contexts are somewhat out of distribution; likelihood against the
frequency-product model not yet measured.

## 7. Done 2026-11-06

**Layer-3 adapter** (`twistroute_bio/models/mrnautilus.py`). `MRNAutilus` wraps the validated encoder as a
denoiser (`mask_id = 4`, V = 86, SUBS-parameterised `logits`); `CDSWindow` is a denoiser view over the coding
positions of a transcript with the UTRs clamped, so layer 1 sees an ordinary N-position problem;
`codon_classes()` maps the model's 86-token vocabulary onto the 64 codons of `twistroute_bio.alphabet`,
excluding every non-codon token. Note the model's codon order is TCAG-based and is **not** ours; the two lookup
tables in the adapter are the only place that mapping lives. `scripts/smoke_mrnautilus.py` samples a short
constrained CDS end to end: 16/16 particles live, 15 distinct sequences, every one translating to the target
peptide and free of EcoRI, BamHI and HindIII sites [verified].

**Refactor.** `decode/smc.py` split into `schedule.py` (reveal schedules), `resampling.py` (resamplers and the
defensive mixture), `twist.py` (proposal, twist variants, potential) and the driver, each with a docstring
stating what it establishes and citing the method where a reader would want the source. `log_mean_exp` promoted
from a private helper to a documented shared one. Method citations added across layers 1-3 (Bar-Hillel,
Moore, Hopcroft-Ullman, George-Liu minimum degree, Kitagawa systematic resampling, Cornish-Bowden IUPAC
codes, Crick wobble, Lorenz ViennaRNA, Su RoPE). All are verified against CrossRef (2026-11-06). **Correction:**
an earlier version of this entry claimed all of them had been verified before being written in; the Su RoPE
reference was in fact written from memory and checked afterwards. It resolves correctly
(doi:10.1016/j.neucom.2023.127063, Neurocomputing 568:127063, 2024), but the process claim was wrong when made.
Citations go through CrossRef before they are written, not after.

**New tests** (10 for the adapter, 4 for constraints, 2 for SMC): vocabulary round trip and class bijection,
CDS window slicing and template immutability, a motif straddling a codon boundary, wrong-protein and
missing-stop rejection, GC band over the whole sequence, optional wobble pairing, **mean-field collapse**
(a context-free denoiser gives zero spread in Ẑ and `eta0` matching brute-force enumeration), and
`log_mean_exp` stability far below the float floor.

**Notes rewritten** (`docs/twistroute_notes.tex`, 24 pp). Restructured per `notes_plan.md`: proofs to
appendices, every section 1-2 pages, boxed claims, open directions as questions, related work by component,
new sections on the mRNA application and the experiments. Prose pass removing AI tells.

**Benchmark landscape survey** (`codon_benchmark_landscape.md`, artifact
b00c68d8-c840-45be-8bd3-ef15c23a3ecb). The finding that matters: **no published learned codon-optimisation
method reports per-position codon reconstruction accuracy against a frequency baseline as a primary result.**
The single instance is CodonTransformer Supplementary Fig. 1, an unnumbered density plot at *nucleotide*
resolution in which the frequency baseline is barely separated from uniform random synonymous choice. The
field reports distributional agreement with held-out natural sequences (CAI/CSI, GC, GC3, ENC, %MinMax
profile agreement by DTW, length-normalised MFE), downstream property prediction from embeddings, and
measured expression. So our flat top-1 result does not contradict published gains; it is a metric those gains
were never claimed on. **Caveats from the survey's own deviation list:** ten verified papers were not read in
full; three quantitative claims are read off figures rather than reported numbers; GEMORNA's gains come from
the publisher abstract only; the 2024-2026 sweep was by search rather than exhaustive.

mRNAutilus's intended use, from its preprint. Provenance of each claim is marked, because this entry
previously stated the whole paragraph as established fact when parts of it came from a sub-agent's reading.

- **arXiv:2605.31296 [verified]** resolves via the arXiv API to "mRNAutilus: Multi-Objective-Guided Discrete
  Generation of mRNA with Optimized Therapeutic Properties", Patel, Tang, Kim, Zhang, Srijay, Lin, Shubham, Pi,
  Wu, Yao, Chatterjee, 29 May 2026.
- **Objectives and Pareto search [verified]** against the abstract, which names Monte Carlo search, a Pareto
  front, half-life, translation efficiency, protein abundance, UTRs and diffusion.
- **Three XGBoost regressors [verified]** against the model repository rather than the abstract (which does not
  mention XGBoost): `classifiers/reg_xg.py` imports `xgboost`, and `classifiers/scoring_functions.py` registers
  exactly three scorers, `HalfLife`, `TranslationRate` and `RiboSeq`. `pareto_mcts.py` is the sampler.
- **5'UTR evaluation [from the sub-agent's reading of the full text, not independently checked]**: evaluated
  only distributionally (MFE and GC against 1,000 natural vertebrate 5'UTRs) plus end-to-end expression, with
  no 5'UTR functional benchmark, no mean-ribosome-load comparison, and no ablation separating the UTR
  contribution from the CDS contribution. The no-ablation half is now **[confirmed]** -- every design varies CDS
  and UTRs jointly, so no reported number isolates either.

### What the paper establishes about its own model (read directly, pp. 1-20 of 32; 21-32 are references)

Four tiers of evidence, only the last of which measures efficacy:

1. **Pretraining [verified, text]** -- validated *purely distributionally*: 200-sequence libraries sampled
   unconditionally at 1000/2500/5000 tokens against length-matched Ensembl vertebrate mRNAs on GC content
   (slightly elevated, converging with length), length-normalised MFE (more stable than natural), Kozak motif
   frequency, pairwise diversity and per-sequence entropy. **No likelihood, no held-out perplexity, no
   reconstruction accuracy against any baseline.** Those five statistics are a low bar: a position-independent
   codon-frequency sampler would plausibly match GC, diversity and entropy. Pretraining is 5,526,848 transcripts
   from Ensembl 113 across 342 vertebrate species, 100K updates.
2. **Embeddings [verified, figure]** -- XGBoost regressors on frozen embeddings predict half-life, translation
   efficiency and protein abundance with held-out Spearman ~0.55-0.72 against ~0.92-0.95 on training data (read
   off Fig. 2 panels; re-check the decimals before citing). Tests the encoder representation, not the
   generative conditional, and the train/validation gap is large.
3. **In-silico design [verified]** -- better than GEMORNA and random on CAI, normalised MFE, optimal-codon
   frequency, GC and uridine content, and on their own three regressors, which are also the optimisation target.
4. **Wet lab [verified, genuinely experimental]** -- IVT from PCR-amplified T7 templates (HiScribe T7 ARCA kit),
   column-cleaned, DNase-treated, LiCl-precipitated, integrity-QC'd by capillary electrophoresis; transfected
   with Lipofectamine MessengerMAX, or Neon NxT electroporation for PEmax in A549. Readouts: luciferase over 12
   time points to 60 h; Spike surface protein by flow cytometry (MFI, 3,000 live cells); Cas9 immunoblot plus
   **editing efficiency by Sanger sequencing** at HEK site 3; and for the ubiquibody, qPCR, a TOPFlash/FOPFlash
   Wnt reporter and cytosolic beta-catenin immunoblot whose effect is **abolished by MG132**, placing it in the
   ubiquitin-proteasome pathway. ~400-fold over wild-type firefly luciferase (a cross-species codon-usage
   mismatch, so a low bar), and -- the comparisons that count -- above GEMORNA zero-shot and a GenScript-optimised
   F-Luc CDS with human alpha-globin UTRs across Jurkat/A549/HepG2, with 3 of 4 Spike designs above BNT162b2 and
   a commercial GenScript Spike construct in A549.

**Consequence for us.** Nothing they report measures the base model in isolation; tier 4 validates the whole
bundle (denoiser + three regressors + tree-guided steering + joint CDS/UTR design + template anchoring). So the
question "is the conditional any good?" is not answered by their paper, and our scoring result (+0.164 nats over
a frequency table under full conditioning, 8/8 genes) is a more direct measurement of it than anything they
report. The two absent things that would have settled it: a held-out likelihood against baselines, and a
CDS-versus-UTR ablation.

**Caveats to carry into any writeup.** Four co-authors are GenScript employees and GenScript constructs are the
main commercial comparator; n = 3 biological replicates, two-tailed t-test, not randomised, not blinded, one
technical outlier removed by stated criteria. Separately, the Fluc and uAb constructs were transcribed with
HiScribe ARCA (unmodified uridine) while constructs from GenScript plasmids used m1-pseudouridine with a cap1
analogue -- a chemistry difference that, if it tracks design vs comparator, works *against* their own result.

**Their constraint mechanism [verified, text] -- this is the one that matters for positioning.** Sec. 4.4.5:
during sampling they permit only codons encoding the same amino acid as the corresponding codon in the input
template, which "guarantees that the sampled mRNA sequence does not form an unintended protein product"; the
same rejection is applied during pretraining denoising. That is exactly our per-position allowed-codon masking.
**They do have hard constraints -- of the per-position, factorising kind.** What that cannot express is anything
with cross-position structure: a restriction site straddling a codon boundary, a global GC band, a target
secondary structure. Per-position masking under guided unmasking is also precisely the step-exact regime whose
trajectory error Sec. 7c measures. So the claim is not "they cannot constrain" but **"they constrain exactly the
factorising class, and we extend to the non-factorising class -- which is also where the trajectory correction
begins to matter."** Their expansion is additionally template-anchored, so like DNA Chisel it perturbs a given
sequence rather than sampling; neither baseline is a sampler, which is what the library-generation claim rests
on. Property data and regressors are deposited at 10.5281/zenodo.19600209 if we ever want the labels.

## 7b. First constrained-family result (2026-11-06)

**Family: restriction-site avoidance** (8 common sites), the first of the constrained families and the one
chosen to implement and validate before expanding (user, 2026-11-06).

**Setup.** 4 real human coding sequences whose natural form *violates* the constraint (HBB, B2M, HBA1, KRAS;
111-189 codons), pulled from RefSeq. Two of the six site occurrences straddle a codon boundary and every one
spans at least two codons, so none is visible to a single-codon substitution rule. Arms: corrected sampler
(twisted SMC on mRNAutilus, K = 8, defensive eps = 0.25, independent schedule), step-exact (K = 1, weights
ignored), frequency table + randomised local repair, and the natural sequence as reference. 16 samples per
arm per gene. Codon frequency table estimated from 381 human RefSeq CDSs (147,948 codons), following
Clarke & Clark's own practice of estimating from the organism's ORFeome rather than hardcoding a table;
cross-checked against the repo's Kazusa-derived table at Pearson r = 0.995, modal codon agreeing for 17/18
multi-codon amino acids [verified].

**Metric.** Distance between a design's %MinMax profile (Clarke & Clark 2008, window 18) and the natural
sequence's. This is the *positional* measure: it asks where rare codons sit, which a product model cannot
control. CAI (Sharp & Li 1987), GC and GC3 are distributional controls. `results/mrna_profile_compare.csv`,
figure `results/mrna_profile_compare.png`.

| arm | valid | %MinMax distance to natural | CAI | GC3 |
|---|---|---|---|---|
| corrected sampler | 64/64 | **34.1** | 0.774 | 0.733 |
| step-exact (uncorrected) | 64/64 | 37.8 | 0.757 | 0.696 |
| frequency table + repair | 64/64 | 66.6 | 1.000 | 0.993 |
| natural (reference) | 0/4 | 0 | 0.756 | 0.592 |

**What it shows.**
1. **Constraint satisfaction is table stakes**: every method reaches 64/64 once repair is allowed. It does
   not discriminate, so it cannot be the headline on its own.
2. **The rule baseline is degenerate on both controls**, not merely worse: CAI 1.000 is the ceiling by
   construction and GC3 0.993 against a natural 0.592. A frequency table has no positional structure at all,
   and its %MinMax profile is a constant at +100.
3. **Model samples are about twice as close to natural positional structure** (34.1 vs 66.6), paired by gene:
   mean difference -32.5, bootstrap 95% CI [-54.1, -13.2], favouring the model in 4/4 genes [verified].
4. **Nucleotide identity does not predict positional fidelity.** All arms sit at 0.78-0.80 identity to the
   natural sequence, yet differ twofold on the profile metric. The metric is measuring something identity is
   blind to, which is the point of using it.

**What it does NOT show [important].** The corrected sampler versus step-exact decoding is **not resolved**
at this scale: mean paired difference -3.8 with bootstrap 95% CI [-12.0, +7.5] over n = 4 genes, favouring
the corrected sampler in 3 of 4 (B2M runs the other way). The demonstrated effect is model-versus-rules, not
corrected-versus-uncorrected. Separating those two needs more genes, and that is what the next run is for.

**Incidental finding.** Max-CAI greedy and max-frequency greedy are the *same* construction: the CAI weight
w(c) = f(c)/max_family f is monotone in f, so the argmax within each synonymous family coincides. They were
planned as two arms and are one. Any paper comparing against "both" is comparing against one baseline twice.

**Cost note.** The CPU denoiser degrades superlinearly with batch width: 0.55 s/row at batch 4 against
1.86 s/row at batch 16 (memory pressure, not compute). Scaled-up runs should chunk the batch or move to GPU;
a 148-codon gene at T = 16 took about 8 minutes per model arm.

## 7c. The trajectory correction measurably changes the sampled distribution (2026-11-06/07)

This is the comparison that tests **our** contribution rather than mRNAutilus's, and it was the gap left open
by Sec. 7b.

**Why it needed a different design from E1.** E1 measured TV against a *rejection* reference: draw native
samples, keep those satisfying C. For a coding sequence the constraint includes the protein, whose native
probability is ~1e-40, so no rejection reference exists. Instead the reference is the corrected sampler at
K = 32; the arms are the corrected sampler at K = 8 and step-exact decoding (K = 1, weights ignored).

**What this measures, stated exactly.** The *size of the trajectory correction*: how far step-exact decoding's
output sits from the corrected sampler's at higher K. It does **not** independently establish that the
corrected sampler is exact on this model -- if K = 8 and K = 32 shared a bias, this design could not see it.
The warrant for the reference is the exactness theorem plus its enumeration check on toy families
(TV <= 4e-16 at L = 4, 6), not this experiment.

**Result** (`results/mrna_correction_size.csv`, figure `results/mrna_correction_size.png`,
raw `results/bias_results.json`). TV excess over a matched sampling floor, 400 sampled codon-position pairs,
2048 reference draws and 768 per arm, T = 16, one L40S:

| arm | TV excess over floor | above floor in |
|---|---|---|
| corrected sampler (K = 8) | **-0.0030** (95% CI -0.0051 to -0.0008) | 1/6 genes |
| step-exact (uncorrected) | **+0.0206** (95% CI +0.0137 to +0.0320) | **6/6 genes** |

Paired difference +0.0236 (95% CI +0.0165 to +0.0348). Step-exact's excess averages 0.37x the floor itself.
Both arms satisfy the constraint on 100% of samples, so this is purely distributional. The corrected sampler
sitting *at* the floor says it has converged by K = 8; step-exact sitting above it on every gene says the
correction changes what is sampled.

**Caveats.** 6 of 16 genes completed before the 5 h SLURM limit (exit 143 at 5.4 h wall); the 6 are the
shortest in the panel, 93-120 codons, so length is confounded with completion and the longer genes are
untested. Partial results survived only because the script writes after each gene. The run used the non-flash
attention path, ~2-3x slower than flash, chosen deliberately after three earlier GPU jobs died on flash/dtype
issues.

**Recovery note (2026-11-07).** The workspace was cleared while idle, destroying two unpushed commits. The
results were all safe as artifacts -- `mrna_correction_size.csv/png`, `structure_feasibility.csv`, this file,
and also `bias_results.json` and `bias_cds.json`, both saved on 2026-11-07 -- and only the code in the two
unpushed commits had to be rebuilt, from the transcript.

Every per-gene number in `mrna_correction_size.csv` reproduces from the raw record exactly, and the panel is
now pinned by RefSeq accession (`experiments/fetch_human_cds.py`), verified to refetch all 16 CDSs
byte-for-byte.

**Correction (2026-11-13), and the more useful lesson.** The paragraph above originally said that
`bias_results.json` and `bias_cds.json` had never been saved as artifacts and had been recovered from the
SLURM job's working directory on the cluster -- that the queued cleanup would have destroyed the only
surviving copy. That was wrong. Both files already had a version 1 in the artifact store, saved on
2026-11-07, byte-identical to what the cluster search returned; `host.query` on `artifact_versions` shows it
directly. The SSH search was unnecessary work that independently re-derived files already in hand, and the
claim was then propagated into a pushed commit message before being caught in review.

The lesson is not the one the original paragraph drew. §2 of this document already says to check an
artifact's version history before acting on it, and that rule was simply not applied: a one-line
`host.query` would have answered the question that a multi-step cluster search was used for. **Check the
artifact store before concluding anything is lost, and before reconstructing or re-deriving it.** The
store -- not the workspace, not git, not a recollection of what was saved -- is the authority on what exists.
The audit-before-delete habit for cluster scratch is still worth keeping, but on its own merits, not because
of a near miss that did not happen.

**Infrastructure lessons** (recorded in `compute_details` for ssh:ondemand): a 592 MB artifact stage stalled
51 min on a degraded link and succeeded at 296 MB once converted to bf16 -- the dtype the GPU job computes in
anyway; weights now live at `~/orcd/pool/cfg-twister/weights/` and later jobs symlink them with an `ssh://`
locator, needing no upload. One failure was mine: a CPU RNG generator against CUDA tensors, which no CPU test
can reproduce. `twisted_smc` now raises a clear error on that mismatch (test in `test_smc.py`).

## 7d. Target-structure family: feasibility is the binding constraint (2026-11-07)

Pre-test for the second constraint family, before committing GPU time.

**Exact target structures are mostly infeasible under a fixed protein.** Only ~44% of nucleotide positions
are free once the protein is set, and they concentrate at wobble positions (116 of 162 at offset 2 on B2M).
Feasible hairpin placements, 8 genes, `results/structure_feasibility.csv` [verified]:

| stem | feasible placements |
|---|---|
| 3 bp | 12.0% (8.5-16.3) |
| 4 bp | 7.7% (4.9-11.5) |
| 5 bp | 6.0% (3.7-8.2) |
| 6 bp | 2.0% (1.1-3.7) |
| 7 bp | 1.6% (0.8-2.8) |

This is a *capability* argument, not a quality one: the compiler answers "infeasible" exactly and instantly,
while a heuristic that fails to find a solution cannot distinguish infeasible from not-found.

**Correction to an earlier claim.** Sec. 8 said target structure is where rule-based construction "fails
outright". That is not supported: for a single feasible hairpin, randomised repair finds a solution easily,
because ~1e57 sequences satisfy it. Where repair does fail is conjunctions:

| constraint | log10 #solutions | repair, 50k tries |
|---|---|---|
| stem5 + GC 47-53% | 57.7 | found |
| stem5 + motifs + GC 47-53% | 57.4 | found |
| stem5 + motifs + GC 49.5-50.5% | 56.0 | **failed** |
| stem5 + motifs + GC 60-62% | 36.6 | **failed** |

Repair fails with 1e56 solutions available: a narrow GC band is a thin shell that random local moves rarely
enter, so the failure is in the search landscape rather than in solution scarcity. **Caveat:** this tests one
randomised repair heuristic, not all heuristics; a purpose-built constraint solver (DNA Chisel) is the right
comparator before any claim here.

### The caveat was decisive: the feasibility claim is dead (2026-11-13)

DNA Chisel 3.2.16 run on the same conjunctions, B2M, same stem-5 hairpin, same eight restriction sites
(`results/conjunction_comparison.csv`; exact counts recomputed here, so the log10 figures differ slightly from
the table above, which used a different feasible placement):

| constraint | log10 #solutions | ad-hoc repair | DNA Chisel |
|---|---|---|---|
| structure only | 59.1 | found | solved, <0.1 s |
| motifs only | 59.6 | found | solved, <0.1 s |
| motifs + GC 47-53% | 58.9 | found | solved, <0.1 s |
| motifs + GC 49.5-50.5% | 57.3 | not tested | solved, <0.1 s |
| stem5 + motifs + GC 47-53% | 58.0 | found | solved, <0.1 s |
| stem5 + motifs + GC 49.5-50.5% | 56.4 | **failed** | solved, <0.1 s |
| stem5 + motifs + GC 60-62% | 37.1 | **failed** | solved, <0.1 s |

Every DNA Chisel solution was verified against an independent checker -- protein preserved, no sites, GC inside
the band, all five stem positions paired -- including the hardest case, which needed 72 nucleotide changes.
**So "rule-based construction fails on conjunctions" is false.** What failed was one randomised repair
heuristic; a real constraint solver with localised backtracking handles every conjunction posed, instantly.
Had the conjunction experiment been built before this check, its central comparison would have been a strawman.

**What survives.** Not feasibility but *distribution*. DNA Chisel returns ONE sequence, 1-72 nt from its input:
a minimal perturbation, not a sample, and it has no notion of relative frequency among the 1e56 valid
alternatives. mRNAutilus's own expansion is likewise template-anchored. For library generation -- many valid,
diverse candidates in correct relative proportion -- neither is a sampler at all, and that is the measurable
claim. Reframe the conjunction experiment around diversity at matched validity, with DNA Chisel as a baseline
that SUCCEEDS on feasibility, rather than as one that fails.

**Also found here.** The stem-5 placement tested is feasible only via GU wobble: two of its five positions have
zero Watson-Crick options once the protein is fixed. `TargetStructure` has `wobble=True` by default, so the
feasibility fractions in Sec. 7b (12.0% of placements at stem 3 down to 1.6% at stem 7) count wobble-dependent
stems as feasible. A GU-held stem is weak biophysically, so the fraction usable for real design is lower than
those figures state -- unquantified as yet, and a one-line change to re-measure.

**Compiler note.** The structure x GC-band conjunction needs `max_cells` above the 2e7 default (363^3 cells);
at 2e8 it compiles in 0.4 s. The default is conservative rather than a capability limit.

## 8. Framing (revised 2026-11-06)

The project's claim is **exact sampling from p(x | C) under hard constraints**, not better codon prediction. The two
mRNA measurements above are compatible and say different things: the model's per-position *mode* matches a frequency
table, while its *joint* does not factor and shifts substantially as context arrives. A mean-field construction
reproduces the first and cannot reproduce the second.

This matters because if the model were mean-field, conditioning on a constraint would still give a non-product
distribution, but our route would sample it exactly in one pass with no particles at all. The sequential machinery
earns its keep only through re-conditioning.

Consequences for the mRNA experiments:

1. Constrained families become the headline: target secondary structure, GC content in sliding windows, motif
   avoidance where the forbidden motif straddles a codon boundary, and conjunctions. Rule-based construction fails
   outright on these rather than merely underperforming.
2. Plain codon optimisation is a sanity check. **Open question raised by the user (2026-11-06):** published learned
   codon-optimisation methods do report gains over frequency baselines, so if we are building on such a model we
   should be able to recover those gains. Our top-1 reconstruction metric may simply be the wrong one — this is what
   the delegated survey is for. Resolve before concluding anything from the flat top-1 result.
3. Negative results are publishable but not sufficient on their own. If step-exact decoding turns out to be nearly
   unbiased on this model, due diligence means checking the constrained families properly and then looking for
   another concrete application — not retreating to toy models (user, 2026-11-06).

## 8b. Decisions and approvals, 2026-11-06 (second session)

| item | status |
|---|---|
| Positional-profile comparison on real human CDSs (%MinMax-DTW, CAI, GC3, MFE) vs frequency table, max-CAI greedy and the natural sequence | **approved** |
| 5'UTR design as a second application | **approved** (but see the provenance flag in Sec. 7: the claim that mRNAutilus's UTR evaluation is only distributional is relayed, not independently checked, and it is what motivates this) |
| Go/no-go pre-tests before E6b | **deferred**: the collaborative design approach is endorsed, the tests themselves are to be discussed first |
| Scope of the mRNA implementation | **one constraint family end to end, validated, before expanding to the others** (user) |

**Notes editing pass (pending, user instruction 2026-11-06).** Further detail to follow before it runs. Settled
so far: leave the biology and mRNA sections alone for now; when they are edited, findings need not be hedged,
and the withdrawn-calibration-claim remark comes out. The target is concision. These notes are long-term
reference, and things better said in a meeting do not belong in them.

**Voice reference.** The user supplied two samples of their own writing (coordination-architectures notes; an
expository paper on the sum-product algorithm). Both use em dashes freely, bold run-in paragraph headers,
first-person plural with contractions, explicit section-by-section signposting, and state plainly what will and
will not be proved. The earlier prose pass stripped every em dash from the notes under a generic rule; that was
wrong for this author and should be reverted during the editing pass.

## 9. Next steps

1. **Status doc current** (this file) — done 2026-11-06.
2. **Layer-3 adapter and local smoke test.** Approved.
3. **Refactor pass** over layers 1-3 for readability and structure, with references where a method is obscure enough
   that a reader would want one, and additional tests. Approved 2026-11-06.
4. **Notes update** (`twistroute_notes.tex`) with the independent schedule, the calibration correction, the
   ambiguity result and the mRNA validation, for collaborators. Approved 2026-11-06.
5. **Replace top-1 accuracy** with a held-out likelihood comparison against the frequency-product model, and
   replicate the synonymous comparison across several real human CDSs rather than one.
6. **Go/no-go diagnostic before E6b**: re-conditioning strength on real human CDSs plus the TV between step-exact
   decoding and the corrected sampler on one constrained task. The user asked to discuss the design of these
   pre-tests, and what we expect them to show, before running them.
7. **E6b best-of-K on mRNA.** Arms: corrected sampler, uncorrected step-exact, uniform-over-valid, mRNAutilus's own
   MCTS sampler, plus a rejection reference where h₀ permits. Scored by ViennaRNA; report diversity at fixed K and
   the rule baseline per family.
8. Deferred: E5 (conjunctions), E6a (best-of-K wiring check), 7B-model confirmation subset, online hidden-mass guard,
   a trajectory-level ESS bound under ambiguity, an observable bound on hidden mass.

## 10. Key artifacts

| what | version id |
|---|---|
| notes (TeX / PDF, 20 pp, 17 refs) | 6652cbad-7e30-4af5-85af-11066c8721d5 / 1b920ed1-796e-431a-b5a3-73de434b39bc |
| experiment plan | c86c59db-6cf3-4cdc-bb43-42b224070c7d |
| mRNA specs | eaff9706-c362-4eda-aa86-1c1cbd6461b4 |
| code bundle (no .git) | ab6ff310-7b7a-4d5c-8fcd-0389f0d8f272 |
| mRNAutilus checkpoint | d4c53d94-b38a-488a-a043-48b2e0e40694 |
| mRNAutilus validation note | fb566ef8-6768-4554-ac5b-62be50969a3f |
| twist design findings | 8dd213ed-2b8f-41a4-827b-5bd424db75e6 |
| stress test B, counting + RNA | 0c481b9c-99d4-4532-bf5a-bc133e30bfa5 |
| E2 native pool, N = 32 | cd613b66-70fd-4d54-b727-ca399cff6c1e |

Cluster: MIT Engaging (`ssh:ondemand`), `mit_normal_gpu`, L40S, `--exclude=node3006`. After an app restart, SSH needs
re-authentication (Duo); a hung command usually means a pending prompt.
