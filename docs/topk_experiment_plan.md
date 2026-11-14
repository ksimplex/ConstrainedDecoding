# Top-k experiment: does sampling a constrained conditional beat optimising under the same constraints?

Status: **design, not yet run.** Written 2026-11-13 so the metric list is fixed before any results exist.
Dependencies are still being verified; the "Open dependencies" section is the gate.

## 1. The claim this tests

Claim 2b of the paper: the exactly-sampled constrained distribution is *useful*, not merely correct.

The argument has a specific shape, and it is worth stating because it determines the metrics. Existing tools
either optimise a fixed objective subject to constraints, or sample without being able to express the
constraints. If the objective a tool optimises were the objective we actually cared about, sampling would have
no advantage -- the optimum is the optimum. The claim is that **the metric that matters is generally not the
metric the optimiser uses**, and that under hard constraints a sample from a good prior lands better on the
former than an optimum of the latter.

This is falsifiable and the falsification condition is in Sec. 6.

**What this experiment is not.** It is not a feasibility claim. DNA Chisel solves every conjunction we have
posed, verified (`results/conjunction_comparison.csv`); the earlier "rule-based construction fails" framing is
dead. Both methods will produce valid sequences. The question is only which valid sequences.

## 2. Comparators, and what each is missing

| method | kind | constraints it can express | objective | gap |
|---|---|---|---|---|
| **ours** | exact sampler from p(x \| C) | non-factorising: motifs across codon boundaries, global GC, target structure | none -- samples the prior | does not optimise anything |
| **DNA Chisel** | constraint solver + local search | rich, same class as ours in practice | soft objectives, hill-climbed | no global optimality; structure handled by an `AvoidHairpins` proxy, not thermodynamics |
| **LinearDesign** | exact global optimiser (max-times over a DFA x grammar lattice) | **none** -- the shipped binary takes only protein, a codon table and `--lambda` | MFE + lambda*log CAI, exactly optimised | cannot express C at all; one sequence per run, no k-best |
| **mRNAutilus (guided)** | reward-steered sampler | per-position only (synonymous masking) | its three learned regressors | cannot express non-factorising C |

**DNA Chisel is the primary comparator** and should be the headline. It is the open instance of the
architecture industry actually uses: hard manufacturability constraints first (restriction sites, GC window,
repeats -- a sequence that violates these fails cloning or synthesis), soft codon objective second. Commercial
vendor optimisers are the real-world default but are proprietary and undocumented.

LinearDesign enters as a **constraint-blind reference point**, not a competitor: it gives the unconstrained
optimum, which is what lets us price the constraints.

## 3. Arms

Every arm produces k = 32 candidates per gene. All arms must satisfy C; validity is checked independently,
not trusted.

1. **ours** -- trajectory-exact constrained sampler, unguided, defensive resampling off (see Sec. 7).
2. **step-exact** -- same proposal, K = 1, weights ignored. Isolates how much of any gap is the trajectory
   correction rather than sampling per se. This is the arm that corresponds to what a per-position-masking
   decoder does.
3. **DNA Chisel** -- k independent runs from different random seeds, constraints = C, objective = MaximizeCAI.
4. **LinearDesign** -- lambda sweep across k values. Expected to violate C; the *fraction satisfying C* is
   itself a recorded result (how hard the constraints bind), and its outputs are used only as the
   unconstrained reference frontier.
5. **max-CAI greedy** -- deterministic floor. Note this is identical to max-frequency greedy (w = f/f_max is
   monotone in f within each synonymous family); they are one arm, not two.

Constraint sets, run **both ways** per the 2026-11-13 decision to explore before selecting:

- **C1 (all methods):** protein identity + 8 restriction sites + GC band 45-60%.
- **C2 (ours and DNA Chisel only):** C1 + a target stem-5 hairpin. LinearDesign cannot express it.

## 4. Metrics -- FIXED NOW, before any results

### Physical (primary; unrestricted, already installed, nobody trained them)

The decomposition matters. Global MFE is *LinearDesign's own objective*, so scoring on it is a home-turf match
that proves nothing. But minimising global folding energy is not unambiguously good: an over-structured 5' end
impedes ribosome loading. So we separate the two:

- **5' accessibility** -- probability that a defined start window is unpaired (ViennaRNA). **Primary metric.**
- **Body ensemble free energy** -- stability of the remainder, excluding the start window.
- **Ensemble defect** against the target structure (C2 only).
- MFE of the whole transcript -- reported for reference only, flagged as the comparator's objective.

### Learned (corroboration; see Sec. 7 for availability)

Used to check the physical result is not an artefact of one instrument. Scored, never optimised -- we sample
unguided, so no circularity.

mRNAutilus's three regressors are XGBoost fits on their own embeddings of PUBLISHED datasets. We therefore
do not need their restricted deposit: the originals are obtainable, and two of them ship the purpose-built
trained model that their XGBoost was re-fitting.

| property | instrument | source | status |
|---|---|---|---|
| **translation efficiency** | **RiboNN** (Zheng, Wang & Agarwal) | zenodo 10.5281/zenodo.15360345 | **OPEN** -- 207 MB weights + 1.8 MB code. Weights licence: academic / non-commercial (Sanofi). Paper: PMID 40715456, *Nat Biotechnol* 2026, doi 10.1038/s41587-025-02712-x, PMC12323635 [verified via NCBI esummary] |
| **mRNA half-life** | **Saluki** (Agarwal & Kelley 2022) | zenodo 10.5281/zenodo.6326409 | **OPEN, CC-BY-4.0** -- single 18.75 GB archive; weights in `datasets/deeplearning/train_gru`. Needs a ranged extraction or a large download |
| protein abundance | Eichhorn et al. 2014 | GEO **GSE61073** (SuperSeries, 95 samples) = Eichhorn SW, Guo H, McGeary SE, Rodriguez-Mias RA, Shin C, Baek D et al., *Mol Cell* 2014, PMID 25263593. GEO also lists **GSE22004** = Guo H, Ingolia NT, Weissman JS, Bartel DP, *Nature* 2010, PMID 20703300 — a DIFFERENT study sharing an author (Guo H); its relationship to the mRNAutilus training set is **not verified** | **raw sequencing only, no trained predictor.** Rebuilding a training set from ribosome profiling + proteomics is a project, not a dependency |
| protein abundance (alt) | mRNAutilus XGBoost on embeddings | zenodo 10.5281/zenodo.19600209 | **restricted (403)**, CC BY-NC-ND |
| degradation (alt) | DegScore (Eterna) | github eternagame/DegScore | open; built for in-line hydrolysis, transfer unclear |

**Decision: use Saluki and RiboNN; drop protein abundance.** It is the weakest of mRNAutilus's three
(validation Spearman ~0.57 against ~0.92 training, read off their Fig. 2), the hardest to obtain, and
half-life and translation efficiency are the two properties that matter most for mRNA therapeutics. Two
independent instruments that agree is stronger evidence than three where one is an unverifiable re-fit.

**Prefer the published originals over mRNAutilus's re-fits.** They are independently peer-reviewed, they are
not derived from the model we sample from (removing any circularity), and they carry no ND restriction.
Note both share an author (V. Agarwal), so they are methodologically consistent rather than arbitrary picks
-- which is a mild correlation to acknowledge, not a flaw.

### Distributional (ours, no external dependency)

- **%MinMax distance from the natural human profile** (validated: hits exactly +/-100 at max/min-frequency
  encodings; natural human genes sit at +30.8 +/- 27.8).
- CAI, GC, GC3 -- context for how far each method drifts from typical.
- Diversity among the k candidates: measured as the *mechanism* by which best-of-k improves, not as a claim.
  Diversity is near-trivial for a sampler and the point is whether it cashes out in properties.

## 5. Panel and reporting

The 16 pinned genes (93-216 codons, `results/bias_cds.json`, pinned by RefSeq accession). Paired by gene
throughout -- every comparison is within-gene, differences aggregated across genes with a paired CI.

Report **best-of-k** and the **full distribution** per arm. Best-of-k is where a sampler's diversity is
supposed to pay; the full distribution shows whether it pays on average or only in the tail.

## 6. What would falsify it

Stated now so the metric cannot be chosen afterwards:

> If our best-of-32 does not beat DNA Chisel's best-of-32 on 5' accessibility at matched body stability,
> paired across the 16 genes, the properties claim fails and gets reported as a negative.

Secondary: if the learned instruments disagree in sign with the physical result, neither is reported as
decisive and we say so.

## 7. Open dependencies (the gate)

1. **RiboNN** -- RESOLVED: open Zenodo deposit, 207 MB weights. Download and confirm it runs on our
   sequences. Lowest-risk of the learned instruments; do this one first.
2. **Saluki** -- 18.75 GB download, or ranged extraction of the `train_gru` subdirectory from the remote zip.
   Decide which before starting. Input format needs a full transcript plus splice track, so a fixed UTR
   context is held constant across all arms.
3. **LinearDesign** -- clones and documents cleanly, but its driver is Python 2.7 with a vendored `gflags`;
   needs either a port of the ~30-line wrapper or direct invocation of the compiled binary. Build not yet
   attempted.
4. **Our sampler on C2** -- the structure x GC-band product needs `max_cells` above the 2e7 default.

**Resampling setting.** Run with resampling OFF. The 2026-11-13 analysis of `stress_B2_summary.csv` showed
every biased Z-hat on an ambiguous route came from a resampling-on arm (K8_def: 0.576, k-hat 1.00) while
resampling-off arms were sound (1.061, k-hat 0.36). The mRNA routes here are factor graphs and therefore
unambiguous, so this is precaution rather than necessity -- but there is no reason to spend the risk.

## 7a. Dependency status as of 2026-11-13 (all verified by running them)

| dependency | status |
|---|---|
| **RiboNN** | **WORKING.** Weights installed, `ribonn/RiboNN-1.0.0/ribonn_score.py` wraps it; scored 4 panel genes across 78 cell types (mean predicted TE 1.35-1.57, ordering not monotone in length). |
| **LinearDesign** | **BUILT AND VERIFIED.** `make` succeeds on Darwin arm64; the README example reproduces exactly (`AUGAACGAUACGGAGGCGAUC`, -1.10 kcal/mol, CAI 0.695). |
| **Saluki** | **WORKING AND VALIDATED.** Weights by RANGED extraction -- 400 `model*_best.h5` at 425 MB, not the 18.75 GB archive (Zenodo honours HTTP range requests, so the central directory is readable remotely). Keras 2.4 under `tf_keras`; the only custom layer, `StochasticShift`, is a training-time augmentation that is identity at inference, so a pass-through stub suffices (`saluki/saluki_load.py`). |
| OpenReview reviews | **CLOSED -- none exist.** The forum is an ICML 2025 Workshop submission (#109) and shows abstract + PDF only; workshop venues keep reviews private. The API's `ChallengeRequiredError` was Cloudflare Turnstile on the same empty page, not a route to hidden content. |

### Instrument validation (2026-11-13) -- each checked against something external, not assumed

**Saluki reproduces its own shipped predictions exactly.** The archive ships per-fold `preds.h5` and
`targets.h5` -- the model's own test-set outputs -- which makes the loader checkable rather than merely
plausible. Decoding the fold's TFRecords (one byte per position; 4 one-hot nucleotide channels + coding-frame
+ splice-site tracks, zero-padded to 12,288) and running our loaded model:

| comparison | result |
|---|---|
| our predictions vs **shipped predictions** | **Pearson 1.00000, max abs diff 5e-4** (float16 storage noise) |
| our predictions vs targets | 0.75596 |
| shipped predictions vs targets | 0.75594 (fold `acc.txt` reports 0.74655 over all 1,297; we used the first shard of 128) |

**Saluki can see synonymous change -- the check that decides whether it is usable at all.** If its signal came
mostly from the UTRs we hold fixed, half-life differences between our arms would be noise and the instrument
would be dead. Six panel genes, each scored as natural / three random synonymous recodings / all-first-synonym:

- within-gene spread across synonymous variants: **mean 0.269, max 0.338**
- between-gene spread of the natural sequences: **0.250**
- **ratio 1.08** -- recoding one gene moves the prediction as much as swapping to a different gene

The degenerate all-first-synonym encoding gives the LOWEST predicted half-life in 6 of 6 genes (negative for
CRYAB and B2M), which is the direction a correct instrument should move and mirrors the max-CAI atypicality
effect. **Caveats:** the splice track is all-zero for synthetic constructs, which is off-distribution relative
to training where it carries information, so only WITHIN-design comparisons are meaningful, never absolute
predicted half-lives; and this used one fold of a 400-model ensemble.

**Saluki's 50 checkpoints are NOT 50 replicates -- averaging them is a silent correctness bug.** The
checkpoints under `train_gru` are `f{0..9}_c{0..4}`: 10 cross-validation folds x **5 separate half-life
datasets**. The `c` index carries its own measurement scale -- on identical sequences c0 centres near 0.2,
c2 near 1.0, c3 near 1.4 -- so a mean over all 50 mixes incommensurable units. Measured on six panel genes,
the naive cross-checkpoint sd was **1.159** against a between-gene sd of **0.098**: the apparent variance was
almost entirely scale disagreement, and a harness built on it would have reported half-life as uninformative
noise rather than revealing the mistake.

The correct protocol, implemented in `saluki/saluki_score.py` (`score_ensemble`, `paired_contrast`): score
every design with every checkpoint, then compare designs **within** a checkpoint and aggregate the paired
differences. Demonstrated on CCL4L1 with 20 checkpoints and three designs:

| comparison | paired mean | 95% CI | checkpoints positive |
|---|---|---|---|
| natural - degenerate (all-first-synonym) | **+0.168** | [+0.010, +0.327] | 90% |
| random synonymous - degenerate | +0.156 | [-0.015, +0.328] | 70% |
| natural - random synonymous | +0.012 | [-0.032, +0.057] | 45% |

The instrument separates a degenerate encoding from a natural one, declines to invent a difference between
two reasonable encodings, and gets the sign right in both cases. Absolute values remain meaningless (zero
splice track, off-distribution), so **every reported Saluki quantity must be a within-checkpoint contrast.**
RiboNN gets the matching treatment: it predicts per cell type, so average within cell type, never across.

**Three tooling gotchas, each already cost a run:**

1. **RiboNN's CLI hardcodes `num_workers=4`**, which dies in a sandbox with "no response from
   torch_shm_manager". Use `num_workers=0` -- free at our scale. This is why we call its `predict` function
   directly instead of `make predict`.
2. **LinearDesign interleaves `j=N` progress markers with results on stdout**, carriage-return delimited.
   Parse defensively.
3. **LinearDesign's Python 2.7 driver is not needed at all.** It is a 20-line `gflags` wrapper around
   `./bin/LinearDesign_2D <lambda> <verbose> <codon_table>` with stdin piped through. Call the binary.

**Finding that changes the LinearDesign arm: lambda saturates.** On a test protein, CAI reaches exactly
1.000 at lambda = 0.5 and the sequence is then IDENTICAL at lambda = 1, 2 and 4 (same folding energy, same
string). Past a low threshold the optimiser degenerates to max-CAI -- which we already showed is the same
solution as max-frequency greedy, since w = f/f_max is monotone in f within each synonymous family. So:

- a lambda sweep yields far fewer than k DISTINCT sequences; the useful range must be calibrated per gene
  rather than assumed, and the number of distinct outputs recorded as a result in its own right;
- the saturation point is exactly where the optimiser stops trading off and starts producing the degenerate
  maximum, which is the atypicality effect visible on a 7-residue test.

## 7b. Provenance note

Every accession, DOI, PMID and **author attribution** in this document was fetched from a primary registry
(NCBI esearch/esummary, CrossRef, the Zenodo REST API, or the repository itself) on 2026-11-13.

This paragraph exists because the table was wrong twice, in the same way, and the second time was *inside an
earlier version of this very note*:

1. GEO accessions that were correct but were attributed to a web-search result containing no such text.
2. `GSE22004` attributed to "Baskerville & Bartel 2010". The accession and PMID were right; the authors were
   never fetched. It is **Guo H, Ingolia NT, Weissman JS, Bartel DP, *Nature* 2010** -- Bartel is the LAST
   author of a four-author paper, which is the shape of the error: a real name adjacent to the real citation.

The lesson is narrower than "check your sources" and worth stating exactly. **Verifying an identifier does
not verify the sentence it sits in.** An esummary call that confirms an accession exists says nothing about
who wrote it, what it contains, or why another paper cites it -- those are separate fields requiring separate
calls. The failure mode is recalling surrounding detail that feels entailed by the verified fact and is not.
A right number with an invented attribution is still a defect, because the attribution is what a reader
checks.

Note also a real distinction the checks turned up: GSE61073 is the Eichhorn SuperSeries (95 samples);
GSE60976 is a 10-sample array-only sub-series of it, so citing the sub-series would understate the data.

## 8. Why this is worth running

If it works, it is the paper's concrete practical result: under the hard constraints that real manufacturing
imposes, sampling a learned conditional yields sequences that are better on a physically-motivated metric than
what a constraint solver optimising a codon objective produces -- and no existing tool can produce them,
because LinearDesign cannot express the constraints and mRNAutilus cannot express the non-factorising ones.

If it fails, the honest result is that exactness buys distributional correctness rather than better designs,
and the paper is a method paper with mRNA as a worked example. That is a smaller paper but not a wrong one.

## 9. Pilot observations (2026-11-14) -- single gene, NOT findings

Two separate runs on CCL4L1 (93 codons). They are recorded separately because an earlier writeup merged
their numbers into one paragraph and thereby tripled the apparent sample size of the LinearDesign arm.

**Run A -- standalone lambda sweep, k = 12.** 12 lambda values produced **11 distinct sequences**, of which
**0 satisfied C**: GC landed at 0.609-0.638 against a 0.60 ceiling, and one sequence also carried a
restriction site. 46.5 s.

**Run B -- harness pilot, k = 4** (`results/topk_pilot.csv`, the saved artifact). 4 LinearDesign candidates,
4 distinct, **0/4 satisfying C**, GC 0.6087-0.6377. All other arm numbers below come from this run only.

| arm | valid C | 5' access | body E/nt | CAI | %MinMax dist |
|---|---|---|---|---|---|
| natural | 0/1 | 0.4285 | -0.348 | 0.731 | 0.0 |
| greedy CAI | 0/1 | 0.5363 | -0.411 | 1.000 | 37.0 |
| LinearDesign | 0/4 | **0.3410** | **-0.508** | 0.832 | 12.9 |
| DNA Chisel | 4/4 | 0.4346 | -0.369 | 0.898 | 21.6 |
| step-exact | 4/4 | 0.4323 | -0.356 | 0.751 | 6.1 |
| ours | 4/4 | **0.4768** | -0.361 | 0.759 | 8.4 |

Both runs agree on the qualitative point -- LinearDesign's optimum sits outside the manufacturable GC window,
because stabilising the fold favours GC -- and it achieves the best body energy together with the worst 5'
accessibility, which is the over-structured-start trade-off the metric decomposition was designed to catch.

**None of this is evidence yet.** One gene, k of 4 and 12. It establishes that the harness runs and that the
predicted direction is visible; the falsification condition in Sec. 6 is stated over 16 genes at k = 32.

**Numbers from different runs must not be merged into one paragraph.** Each row above names the run it came
from, for exactly that reason.

## 10. Where this runs, and why (2026-11-14)

**Measured:** one denoiser forward costs **~450 ms per row on this CPU**, and that figure is FLAT across
batch sizes 2-16 (475 / 524 / 444 / 455 ms). Flatness matters: it means the forward is already saturating
the available cores, so larger batches buy no throughput on CPU -- they only buy memory pressure.

**Implication.** The `ours` arm needs k x K x T = 32 x 8 x 16 = 4,096 row-forwards per gene, plus 512 for
step-exact. At 450 ms that is ~34 min per gene, or **~9 hours for 16 genes** before any scoring. This is a
GPU workload, as the bias panel was.

**The incident that produced this measurement.** The first attempt used `chunk_rows=256`, inherited from the
GPU bias job where an L40S has 48 GB. On a 16 GB machine that is ~1.3 GB per attention matrix recurring per
layer; with two copies of the 148M model resident (the run plus a profiler started to diagnose the slowness)
the host hit **~14 MB free of 16 GB and a load average of 440 on 8 cores**, and crashed the user's app. The
profiler never printed a byte -- it was thrashing, not computing.

Two fixes, both landed:

1. `default_chunk_rows(n_codons, device)` in `experiments/topk_arms.py` sizes batches from ACTUAL free
   memory (`psutil`, or `torch.cuda.mem_get_info` on GPU) instead of a constant. Deliberately conservative:
   an over-small batch wastes throughput, an over-large one takes the machine down.
2. Real runs go to the cluster GPU. Local CPU is for smoke tests only.

**A tuning constant carried across hardware is a correctness bug, not a performance one.** The value was
right where it was written and catastrophic where it was reused, and nothing in the code recorded which
machine it assumed.
