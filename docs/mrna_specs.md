# mRNA stage: module layout and route specifications

**Status.** Design document. Nothing here is implemented yet. Written 2026-11-04, after the twist-design
rounds in `twist_design_findings.md` and before the stress-test results land.

**Decisions taken** (user, 2026-11-04): target application is **library generation** (many diverse candidates
all satisfying hard constraints), and the first scope is **coding sequences with mRNAutilus**.

---

## 1. Three layers

The point of the separation is that layer 1 is the paper's contribution, layer 2 is reusable by anyone doing
nucleotide design, and layer 3 is a single adapter that a different model would replace.

| layer | what belongs | where | depends on |
|---|---|---|---|
| **1. generic** | routes, samplers, twisted SMC, diagnostics, the calibrated twist | `twistroute/` (exists) | nothing domain-specific |
| **2. nucleotide / mRNA domain** | codon tables, motif and GC constraint *builders*, structure factor graphs, folding scorers, the constraint-instance suite | `twistroute_bio/` (new, separate importable package) | layer 1 only |
| **3. mRNAutilus adapter** | checkpoint loading, tokenizer, class map, native-process adapter | `twistroute_bio/models/mrnautilus.py` (new, one file) | layer 2 + the vendored mRNAutilus source |

Rules that keep the layers honest:

- **Layer 1 never imports layer 2 or 3.** It already satisfies this; the test suite runs without any
  biological dependency.
- **Layer 2 never imports layer 3.** Every builder returns a layer-1 route object (`WDFA`, `Inside`,
  `FactorGraph`) over an abstract symbol alphabet. A builder takes an *alphabet description* as an argument
  (which symbol index is which codon), never a model.
- **Layer 3 is one adapter class** satisfying the existing denoiser interface (`.mask_id`, `.logits(x) ->
  (B, N, V)` log-probs, `.V`), plus a function returning the `TokenClasses` map for a given constraint
  alphabet. Swapping in CodonMamba or an ncRNA model means writing one more file at this layer.
- **New generic capability goes to layer 1, not layer 2.** If the mRNA work needs, say, a product-of-routes
  construction or a mixture twist, it is a layer-1 feature with layer-1 tests, used by layer 2.

### What is *not* a new layer

The constraint *instances* (which motifs, which GC band, which target structures, which proteins) are data,
not code: one JSON file per suite under `twistroute_bio/configs/`, in the same style as
`experiments/configs/e2_cells.json`.

---

## 2. Layer 2: `twistroute_bio`

```
twistroute_bio/
  __init__.py
  alphabet.py        codon tables, alphabet descriptions, nucleotide <-> codon lifting
  constraints/
    motifs.py        forbidden / mandatory motifs at nucleotide level, lifted to codons
    composition.py   GC band, codon-usage band, repeat limits
    structure.py     target-structure factor graph; span-bounded hairpin chart
    protein.py       synonymous-codon masks (not a route: a per-position restriction)
  score/
    vienna.py        ViennaRNA wrappers: MFE, structure distance, ensemble defect
    rules.py         model-free construction baselines (frequency-based codon choice + repair)
  configs/
    *.json           constraint suites
  models/
    mrnautilus.py    LAYER 3 (see section 4)
```

### 2.1 `alphabet.py` — the lifting problem

The model's CDS tokens are the 64 codons. Most biological constraints are stated over nucleotides. Three
alphabet descriptions, each a small dataclass holding the symbol list and the index maps:

- `NUC4` — A, C, G, T (U is mapped to T, as mRNAutilus does on encode).
- `CODON64` — the 64 codons in `CODON_DICT` order.
- `CODON_CLASSES(k)` — a coarsening of the 64 codons into `k` classes, needed when a route's state count
  must stay small (`A = 64` makes a DFA transition table 64x larger than the toy experiments' `A = 4`).

Two operations, both pure functions with brute-force tests:

1. `lift_nuc_dfa_to_codons(dfa_nuc) -> WDFA` — given a nucleotide automaton with `Q` states, build the codon
   automaton whose transition on codon `c = n1 n2 n3` is the composition of the three nucleotide
   transitions. States: the same `Q`. Edges: `Q * 64`. **This is the key routine for the whole stage**: it
   makes every nucleotide-level regular constraint (motifs, GC, repeats) available at codon granularity
   without changing the model's tokenization. Reading-frame effects come out automatically, because a motif
   that straddles a codon boundary simply ends in a different state mid-codon.
2. `fix_protein(aa_sequence) -> per-position allowed-symbol mask` — from `CODON_DICT`, the synonymous set of
   each residue. Not a route: it enters as a per-position restriction on the class weights, as revealed
   positions already do.

### 2.2 `constraints/motifs.py`

`forbidden_motifs(patterns_nuc, alphabet, frame=...) -> route`. Compiles the nucleotide Aho-Corasick
automaton with the existing `dfa_forbidden`, then lifts it. Constraint instances worth including, all of
which couple codons and so are *not* solvable by per-codon rules:

- restriction sites (`GAATTC` EcoRI, `GGATCC` BamHI) for cloning compatibility;
- out-of-frame `ATG` (spurious start codons);
- `AATAAA` (cryptic polyadenylation signal);
- homopolymer runs (`AAAAAA`, `TTTTTT`) and dinucleotide repeats;
- cryptic 5' splice sites (`GTAAGT` and variants).

`mandatory_motifs(...)` uses `dfa_contains` the same way.

### 2.3 `constraints/composition.py`

- `gc_band(lo, hi, alphabet)` — count-based automaton over the lifted alphabet. At codon level, each codon
  contributes a known GC count in 0..3, so the state is the running GC total: `3L + 1` states, built with
  the same counting construction as `dfa_fraction_band`.
- `gc_window_band(w, lo, hi, alphabet)` — GC fraction within every sliding window of `w` nucleotides. State
  is the GC pattern of the last `w - 1` positions; exact but exponential in `w`, so it is only used at small
  `w`, or approximated by a per-window conjunction. **The compiler must refuse** rather than silently blow
  up; `build_dfa`'s `max_states` already does this.
- `codon_usage_band(...)` — bound the count of rare codons (CAI-like, as a hard band rather than an
  average).

### 2.4 `constraints/structure.py` — the correlated family

Two constructions, both already supported by layer 1:

1. **Target structure, `target_structure_fg(pairs, alphabet)`.** Given a target secondary structure as a
   list of base pairs `(i, j)`, build a `FactorGraph` with one pairwise factor per pair whose table allows
   only complementary nucleotide combinations (`RNA_PAIRS`). A non-crossing pair set has treewidth 1 at
   nucleotide level, so variable elimination is exact and linear. **This is the constraint that most
   directly tests the product-form twist**: the twist must express "position `i` and position `j` must agree",
   which a product law cannot represent. At codon level the factor's scope is the two codons containing the
   paired nucleotides, so the table is `64 x 64` with the other four nucleotides free.
2. **Span-bounded hairpin control, `hairpin_chart(W, min_loop, alphabet)`.** The Nussinov-style grammar
   already in `rna_pair_grammar`, restricted to pairs of span at most `W`. Cost `O(L W^2 |G|)` instead of
   `O(L^3 |G|)`. Two uses: forbid strong local hairpins near the start codon (a known translation-initiation
   effect), or require a structure of a given class. Ambiguity is a known cost here, measured in E3.

### 2.5 `score/vienna.py` — evaluation only, never a route

ViennaRNA (`RNA.fold`, `RNA.pf_fold`) provides MFE, the MFE structure, base-pair distance to a target, and
ensemble defect. These are **deterministic scores for E6b**, chosen precisely because an optimizer cannot
exploit predictor error. They never enter the route. If we later want to *steer* toward low MFE, that is a
soft twist (`prop:free` in the notes covers it), kept as a separate arm.

### 2.6 `score/rules.py` — the triviality guard

Per the FactorDLM warning already in the experiment plan: for each constraint family, a model-free
construction (frequency-based codon choice, then local repair to satisfy the constraint). A family where
this baseline scores as well as the sampler is a family where the model's distribution does no work, and it
is reported as such rather than hidden.

---

## 3. What each constraint needs from layer 1

| constraint | layer-1 object | exists? | cost at CDS length L codons |
|---|---|---|---|
| protein identity | per-position mask | yes (class weights) | free |
| forbidden / mandatory motifs | `WDFA` via lift | yes + lift routine | `O(L * Q * 64)` per pass, Q = motif-trie size |
| GC band | counting `WDFA` | yes | `O(L^2)` states x transitions; fine to L ~ 500 |
| GC in windows | `WDFA`, refuses past budget | yes | exponential in window width |
| target structure | `FactorGraph` | yes | linear for non-crossing pairs |
| local hairpin bound | `Inside` chart, span-bounded | **needs span bound added** | `O(L W^2 |G|)` |
| conjunction of the above | `product_dfa`, or a factor graph combining them | partially (E5 open) | state product |

Two layer-1 gaps this stage would fill, both generic:

1. **Span-bounded charts.** A `max_span` argument to `Inside` / `TorchChart`. Generic, benefits any grammar.
2. **Products of heterogeneous routes** (an automaton AND a factor graph). This is E5's open question and is
   generic; the mRNA suite is its first real user.

---

## 4. Layer 3: the mRNAutilus adapter

**Model.** `atombio/mrnautilus` on Hugging Face (code, CC BY-NC 4.0); weights are a Google Drive
checkpoint (`best.ckpt`) linked from its README; pretraining data on Zenodo. 150M-parameter masked diffusion
model, 500 diffusion timesteps, trained on full-length mRNA.

**Alphabet (from `src/rinalmo/data/alphabet.py`, `Alphabet_uni`).** 5 special tokens
(`<cls> <pad> <eos> <unk> <mask>`) + 64 codon tokens + 17 nucleotide tokens = 86. The CDS is encoded by
codon (`cds_encode`), the UTRs by nucleotide. `<mask>` is index 4. The class also precomputes
`neighbors_idx`, the synonymous codons of each codon, which is exactly the protein-identity mask.

**What the adapter must provide:**

```python
class MRNAutilus:            # satisfies the existing denoiser interface
    mask_id: int             # 4
    V: int                   # 86
    def logits(self, x) -> Tensor   # (B, N, V) log-probs
```

plus `codon_classes(alphabet_description) -> TokenClasses` mapping the 86 tokens onto the constraint
alphabet, with every non-codon token excluded at CDS positions.

**Three things to verify before any experiment** (each a cheap script, not a research question):

1. **The native process.** Our exactness guarantee is relative to a specific reverse process: reveal order
   from the scheduler, temperature 1, no truncation. mRNAutilus's own sampler is MCTS over multiple
   objectives, which is a *different* process. We must read its diffusion sampler
   (`src/mrna/tasks/lm/mdlm.py`) and implement the plain ancestral sampler it corresponds to. If its
   schedule is state-dependent (`noise.state_dependent: true` in `config.yaml`), that must be reflected in
   `kappa`, or we are exact with respect to the wrong target.
2. **Positional structure.** `<cls>`, `<eos>`, and the UTR/CDS boundary are not free positions. The route
   applies to the CDS span only; everything else is clamped.
3. **Numerics.** bf16 underflow and the `1e-38` floor in `class_logw` matter more here than on OWT, because
   64 codon classes spread the probability mass thinner. The support-loss guard (a feasible state receiving
   twist `-inf`) belongs in layer 1 and should be in place before this stage.

**Licence note.** CC BY-NC 4.0 is non-commercial. Fine for the paper; worth stating in the README.

---

## 5. Experiment shape (E6b, library generation)

Unchanged from the experiment plan, instantiated here:

- **Arms:** corrected sampler; uncorrected step-exact; uniform-over-valid; mRNAutilus's own MCTS sampler
  (mode-seeking, a flat line in K); rejection from the native model where h0 permits.
- **Score:** ViennaRNA MFE and distance to target structure (deterministic).
- **Headline:** best-of-K crossover K*, plus diversity at fixed K (distinct sequences, mean pairwise edit
  distance) and base-model likelihood of satisfying samples.
- **Per family:** the model-free construction baseline, reported alongside.

---

## 6. Open dependencies

1. **Weights** — user to download `best.ckpt` from the Drive link.
2. **Stress-test results** (jobs `329673ef`, `f6db4ec5`) decide whether the product-form twist handles
   correlated constraints. The target-structure family in section 2.4 is the mRNA analogue of the copy and
   RNA-pairing families being tested. If those fail, mixtures of product laws (notes, remark after
   `prop:calibrated`) come first.
3. **ViennaRNA** in the cluster environment (conda `bioconda::viennarna`).
4. **Layer-1 additions**: span-bounded charts; heterogeneous route products; the support-loss guard.
