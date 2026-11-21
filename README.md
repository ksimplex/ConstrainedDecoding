# twistroute

Exact constrained sampling for masked diffusion language models (MDLMs), for any constraint that compiles to a
tractable *route*: automata (regular languages, motifs, counting), grammar charts (context-free, including
ambiguous grammars such as RNA pairing), factor graphs (copies, low-width relations) and spanning trees.

Decoding is twisted sequential Monte Carlo in the style of TWISTER (Thimmaiah et al., 2026), with the automaton
replaced by a route interface. Every output satisfies the constraint, the weighted particles target the model's
own conditional `p(x | x ∈ C)`, and the average weight `Ẑ` is an unbiased estimate of `P_model(C)` at every
particle count.

## Where to start

| You want | Read |
|---|---|
| the method | the project notes, `Constrained_Diffusion.pdf` |
| how the code follows the notes, section by section and equation by equation | `docs/refactor_map.md` |
| to install and run something | `docs/setup.md` |
| what each experiment script does and where its result is recorded | `experiments/INDEX.md` |
| the results log | `docs/parallel_unmasking_status.md` |

## The code that follows the notes

| Notes | Module | Main names |
|---|---|---|
| §2, Proposition 1: constraints | `twistroute/constraints/` | `Constraint` (`log_prob`, `marginals`, `sample`, `contains`); `forbidden_motifs`, `exactly_k`, `dyck1`, `on_positions`, `factor_graph`, `grammar`, `intersect`, `from_regex` |
| §1: a partly masked sequence and the model's prediction, eq. (2) | `twistroute/state.py` | `State`, `Step` |
| §2.3: twists and proposals; weight of a step, eq. (5) | `twistroute/twists/` | `Twist`; `ProductTwist` (eq. 9, 10); `ChainTwist`, `ChainLaw` (eq. 11 to 13); `counted_pair_table`, `probed_pair_table` |
| §2.3: unconstrained step, eq. (4); weighted sampler, eq. (6) | `twistroute/sampler.py`, `twistroute/resampling.py` | `unconstrained_step`, `weighted_sampler`, `Resampling` |
| §1, §4: unmasking schedules | `twistroute/schedules/` | `RandomOrder`, `LeftToRight`, `Confidence`, `Sum`, `Blocks`, `BudgetedCoReveal` |
| Table 3, Figures 1 and 2: evaluation | `twistroute/evaluation/` | `target_law`, `one_particle_law`, `kl_of_output`; `upper_bound`, `leading_term`; `sd_log_Z_hat` |

Three places where the code differs from the notes on purpose, each marked by a comment in the code: the chain twist mixes in
a 5% share of the product twist (`ChainTwist(product_share=0.05)`; 0 is the notes' twist); the pairwise joints of the chain are
matched to the model's marginals by 5 rounds of scaling, not to convergence; the sampler starts every weight at the twist of
the fully masked sequence, so that the mean final weight estimates P(x_0 in C).

## Example (CPU, oracle model)

```python
import numpy as np
import torch

from twistroute import constraints
from twistroute.models import noisy_hmm
from twistroute.sampler import weighted_sampler
from twistroute.schedules import LeftToRight
from twistroute.twists import ChainTwist, counted_pair_table

L = 12
model = noisy_hmm(L, 3, 0.85, 0.3, seed=1, concentration=2.0)   # an oracle model with 3 tokens
classes = model.classes()                                        # here every token is its own class
constraint = constraints.forbidden_motifs([(0, 0), (1, 2)], 3)   # "00" and "12" never occur
table = counted_pair_table(model, classes, 2000, torch.Generator().manual_seed(0))
out = weighted_sampler(
    model, classes, constraint, L, L, runs=1000, particles=4,
    twist=ChainTwist.for_constraint(constraint, table), schedule=LeftToRight(),
    generator=torch.Generator().manual_seed(1),
)
samples = out.returned(np.random.default_rng(0))    # (1000, 12): one sequence per run, all in C
print(np.exp(out.log_Z_hat).mean())                 # 0.141: the estimate of P(x_0 in C) (exact: 0.141)
```

`twist=None` gives the product twist with tempered marginals (temperature 0.7), the library's default.
`twist=ProductTwist(temperature=1.0)` is the product twist of the notes: TWISTER (arXiv:2609.35609), and with `particles=1` and
no weights the constrained sampler of Dang and Ermon (arXiv:2607.07026). Table 3 of the notes: `python experiments/notes/table3.py` (about one minute).

## Layout

```
twistroute/
  constraints/    the constraint interface of Proposition 1 and the families of Table 1
  state.py        State (x_t and the prediction at it), Step (x_t to x_s)
  twists/         base.py (interface, eq. 5), product.py, chain.py, chain_law.py, tables.py
  sampler.py      unconstrained_step, weighted_sampler
  resampling.py   plain, defensive and weight-power resampling
  schedules/      base.py, sequential.py, parallel.py, baselines.py, dependence.py
  evaluation/     exact.py (exact laws at short length), bounds.py, spread.py
  models/         oracle models, LLaDA and MDLM wrappers, token classes
  compile/, torch_routes/, decode/, diagnostics.py, experiment.py
                  the earlier packages, unchanged: the algorithms behind `constraints`, and the
                  earlier sampler `decode.twisted_smc` with every option that was tried
experiments/
  notes/          tables of the notes from the new code
  benchmarks/     LLaDA and MDLM; JSON-Mode-Eval, BFCL, xLAM (the runners use the new sampler)
  studies/        oracle studies that current claims rest on; pulib/ is their shared library
  archive/        answered questions, kept runnable
  INDEX.md        every script, what it does, where its result is recorded
  (top level)     scripts from before this line of work, unchanged
tests/
  core/           the code that follows the notes; reference/ holds the frozen outputs of the earlier sampler
  algorithms/     compile, torch_routes, models
  legacy/         the earlier sampler and policies
  experiments/    the experiment plumbing and the comparison studies
docs/             setup.md, refactor_map.md, the status documents, earlier working notes
```


### Domain layer

`twistroute_bio/` builds ordinary routes for nucleotide and mRNA constraints (codon alphabet, genetic code,
codon lifting of nucleotide automata, synonymous-codon masks, CDS constraint builders) plus ViennaRNA scorers
and rule-based construction baselines. It depends on `twistroute`, never the reverse. Design:
`docs/mrna_specs.md`.

`twistroute_bio/models/` holds the model adapter: a self-contained RiNALMo encoder (`_rinalmo/net.py`, no
flash-attn dependency, block verified bit-identical to upstream) and the mRNAutilus denoiser
(`mrnautilus.py`): `MRNAutilus` for a whole transcript, `CDSWindow` for a denoiser view over just the coding
positions with the UTRs clamped, and `codon_classes()` mapping the model's 86-token vocabulary onto the 64
codons. Validation: `docs/mrna_model_validation.md`.

```bash
python scripts/smoke_mrnautilus.py <checkpoint.ckpt>   # samples a short constrained CDS on CPU
```

Protein identity enters **through the route**, as per-position allowed-codon masks, never by renormalising the
model's predictions: masking predictions step by step is the step-local projection that biases the trajectory
law, which is exactly what this method corrects.


## Setup and tests

`docs/setup.md` has the details in three levels: the core (CPU only), benchmark preparation (datasets and a tokenizer, CPU),
and real-model runs (weights and a GPU).

```bash
pip install -e ".[test]"
pytest tests/core                      # the code that follows the notes, about 2 minutes
python experiments/notes/table3.py     # Table 3 of the notes, about 1 minute
```

All tests of this work: `OMP_NUM_THREADS=4 pytest tests/core tests/algorithms tests/legacy tests/experiments` (489 tests,
25 to 35 minutes on an 8-core laptop).

## The earlier interface: `twistroute.decode`

The sections from here to "Reproducing TWISTER Figure 1(a)" were written before the code was restructured to follow the notes.
They describe the earlier sampler `twistroute.decode.twisted_smc` in its own vocabulary (a "route" is a constraint; "native" is
unconstrained; "reveal" is unmask; `logZ` is log q(C)). That code is unchanged and still runs. The new sampler reproduces it
exactly on a frozen set of cases (`tests/core/test_core_matches_legacy.py`).


## How it works

A route provides three operations on per-position class log-weights `lw[L, A]` (revealed positions one-hot):

| operation | meaning |
|---|---|
| `logZ(lw)` | `log Σ_{y∈C} d(y) Π_i exp(lw[i, y_i])` (d = 1, or the parse count for ambiguous grammars) |
| `sample(lw)` | an exact draw from that law |
| `accepts(y)` | membership, used as the terminal twist `1_C` |

At each denoising step, with `lw` from the denoiser at the current state `x`:

```
η(x)   = logZ(lw)                        # twist: can C still be met?
r      = native reveal set               # same scheduler as unconstrained decoding
y      ~ q(· | C; lw); reveal y_r        # step-exact proposal (TWISTER Prop. 3.6)
w     *= η(x') / Z(x'; x)                # re-freezing potential (TWISTER Alg. 2); η(x_T) = 1_C(x_T)
resample when ESS < ess_min · K       # ancestors from (1 − ε)·w̃ + ε/K, weight w̃/π (ε = defensive)
```

The weights telescope (TWISTER Thm. 4.2): `q(path) · w(path) = p_native(path) · 1_C(x_T)`. The proof uses the
automaton only to compute `Z`, so it holds for every route. For ambiguous grammars the intermediate twists are
derivation-weighted, but the target is unchanged because the terminal twist is the exact indicator; ambiguity
only costs variance.

| route | families | logZ | sample |
|---|---|---|---|
| automaton | regular, forbidden/required motifs, counting | forward algorithm | forward-filter backward-sample |
| grammar chart | Dyck, CFGs, RNA pairing | inside algorithm | top-down from the chart |
| factor graph | copy, pairwise relations, low-width constraints | variable elimination | backward sampling |
| determinant | arborescences | matrix-tree theorem | Wilson's algorithm |

**Span limits.** A grammar can bound the yield of chosen nonterminals (`Inside(..., span_limits={"X": W})`, or
`"span_limits"` in a cfg spec). Bounds implied by the rules are inferred, and the batched chart skips split
points no derivation can use, so a grammar whose only unbounded nonterminal is a right-branching spine costs
O(L² W |R|) instead of O(L³ |R|).

**Conjunctions.** A spec `{"kind": "and", "parts": [...]}` compiles to a single route: automata by the product
construction, an automaton and a grammar by the Bar-Hillel product (a grammar, so it runs on the batched
chart). An automaton and a factor graph combine through augmented variables (symbol, automaton state), which gives
another factor graph. Two grammars are refused, since their intersection is not context-free in general.

**Factor graphs on the GPU.** `TorchFactorGraph` runs variable elimination for a fixed order (default greedy
min-degree) as one broadcast sum and logsumexp per bucket, and samples backwards through the stored bucket
tables. Spec: `{"kind": "factor_graph", "factors": [{"scope": [i, j], "table": [[...]]}, ...]}`. Cost is
exponential only in the induced width; nested pairs plus chain factors at L = 200 have width 3.

### Options of `twisted_smc`

| option | effect |
|---|---|
| `ess_min` | resample a run when ESS < `ess_min`·K (0 = no resampling) |
| `defensive=ε` | ancestors from (1 − ε)w̃ + ε/K with offspring weight w̃/π; recommended ε = 0.25 |
| `resampler` | `"multinomial"` (default), `"cpu"`, `"systematic"` |
| `twist_power=β` | tempered intermediate twists η^β (1 = TWISTER) |
| `lookahead=j`, `lookahead_draws=m` | look-ahead twist: η times the mean potential product of m speculative step-exact rollouts of j steps; an unbiased estimate of the j-step-ahead frozen twist, and of the Doob function when j reaches T (experimental) |
| `mix_native=a` | propose from (1 − a)·step-exact + a·native kernel; bounds each potential by ψ′/(aψ) (experimental) |
| `twist_temp=τ` | calibrated twist: the route sum under the class law ∝ p^τ, with its locally optimal proposal (1 = TWISTER); τ ≤ 0.35 removes the resampling penalty on the synthetic testbed (efficiency 0.14 → 1.04) |
| `schedule` | `"fixed"` (N/T positions per step, the default and TWISTER's) or `"independent"` (each masked position reveals with probability 1/(T−t), the MDLM reverse step). These target **different** processes: h₀ differs by 4–10 standard errors between them |
| `guard` | `"warn"` (default), `"raise"`, `"off"`: support invariants checked every step (a live particle must keep a finite clamped sum and twist); counts returned under `out["guard"]` |

All options keep E[Ẑ] = h₀ (tests/legacy/test_smc.py, tests/legacy/test_resampling.py). The maths is in `docs/twistroute_notes.pdf`.


## Quick start (CPU, synthetic denoiser)

```python
import torch
from twistroute.decode import twisted_smc, select_particles
from twistroute.models import SyntheticMDLM
from twistroute.torch_routes import build_route

den = SyntheticMDLM(N=8, scale=4.0)
spec = dict(kind="cfg", grammar="dyck1_amb", A=2, N=8)            # balanced brackets, ambiguous grammar
classes = den.classes(4).merge([0, 1, 0, 1])                       # token -> bracket symbol
out = twisted_smc(den, classes, build_route(spec), N=8, T=4, nrun=1000, K=4,
                  gen=torch.Generator().manual_seed(0), return_particles=True)
samples = select_particles(out, __import__("numpy").random.default_rng(0))   # (1000, 8), all valid
```

## Experiments (OWT-130M, one L40S)

| experiment | script | status |
|---|---|---|
| E4 route costs vs denoiser forward pass | `experiments/e4_bench.py` | done: cheapest route predicted in 26/27 cells; refusals at 3-CNF (10 clauses) and Dyck-2 automaton (L = 36) |
| E2 native pools | `experiments/e2_native.py` | done: 200k samples at N = 32, 64 |
| E2 `Ẑ` unbiasedness | `experiments/e2_smc.py`, `configs/e2_cells.json` | K = 4, 16 with 2,000 runs fall short of h₀ in 23/24 cells; under investigation (below) |
| E2 two-sided check | `experiments/e2_twosided.py` | K = 1 with 50k runs is unbiased in all 6 cells (Ẑ/h₀ 0.94–1.03, all within 1.5 s.e.) |
| resampling on/off | `experiments/e2_smc.py`, `configs/resampling_cells.json` | done: copy N=32, K=16, 8k runs. On 1.016 (CI 0.94–1.13), off 1.007 (CI 0.985–1.03) |
| TWISTER Fig. 1(a) reproduction | `experiments/twister_fig1.py` (+ `--synthetic` CPU dry run), `experiments/analysis/twister_fig1_analysis.py` | done, qualitatively. Step-exact TV rises 0.028 → 0.126 from T = 1 to 16; TWISTER K = 4: 0.028 → 0.054; K = 8: 0.028 → 0.040; noise floor ≈ 0.03 |
| resampling arms, 100k runs each | `experiments/resampling_arms.py` (+ `--synthetic`) | done on the Fig. 1(a) config (seed 1, T = 8, K = 4). Plain resampling 0.930 (CI 0.910–0.949); off 0.998; K = 1 0.987; CPU / systematic resamplers 0.931 / 0.921; defensive ε = 0.25 0.998 (0.977–1.019); tempering alone 0.932 (β = 0.5), 0.953 (β = 0.25) |
| hidden-mass decomposition | `resampling_arms.py --arms K4_off_trace`, `diagnostics.hidden_mass` | done: 13% of particles have normalised weight < 10⁻⁶ at the first trigger step yet carry 4.7% of h₀ |
| E3 ambiguity, CPU | `experiments/e3_ambiguity.py` | done: exact TV ≤ 4×10⁻¹⁶ at L = 4, 6; K = 16 ESS fraction 0.93–0.97 (unambiguous Dyck-1) vs 0.92 → 0.34 (ambiguous) and 0.89 → 0.30 (RNA) over L = 8–32 |
| E1 trajectory bias across families | `experiments/e1_families.py`, `experiments/analysis/e1_analysis.py` | done: six families at N = 32, T = 16. TV excess at K = 16 — copy 0.136 (0.040 with defensive + τ = 0.5), regular 0.026, counting ≈0, Dyck-1 0.086, RNA 0.005; step-exact 0.03–0.165 |
| stress test, long horizon | `experiments/e1_families.py` at N = 64, T = 64, K = 8 | done: copy TV excess 0.038 (defensive + τ = 0.5) but Ẑ/h₀ 0.902; counting exact; RNA pairing Ẑ/h₀ 0.576 (k̂ = 1.00) under defensive resampling — the clearest remaining failure |
| twist design, rounds 2–3 | `experiments/twist_lab.py` | done: look-ahead twists dropped; calibrated twist swept over τ, all 15 arms Ẑ/h₀ within 0.979–1.008 |
| mRNAutilus encoder validation | `docs/mrna_model_validation.md` | done: block bit-identical to upstream; re-conditioning strength TV 0.337 end to end, so there is bias to correct |
| twist design, round 1 | `experiments/twist_lab.py` (CPU, synthetic, scale 8) | done: Monte Carlo look-ahead twists increase hidden mass (full rollout: 6.9% of h₀ below 10⁻⁶ vs 0.04% frozen) and lose to the frozen twist; resampling-off is the most efficient arm, defensive resampling recovers about half of its efficiency |
| CPU controls | `experiments/controls/synthetic_control.py` | done: resampling, routes and float32 sampling all unbiased on synthetic denoisers |

Cluster jobs ran on MIT Engaging (`mit_normal_gpu`, `--gres=gpu:l40s:1`), with `PYTHONPATH` set to the repo root.

**Resampling.** Single-particle SMC is unbiased on the real model (two-sided check). With multinomial resampling
the frozen twist's over-confidence produces a finite-run shortfall: particles whose twist is ~10⁻¹⁸ at the
resampling step but later recover carry a few percent of h₀ and are almost never kept. Defensive resampling
(`defensive=0.25`: ancestors from (1 − ε)w̃ + ε/K, offspring weight w̃/π) removes it at little cost and is the
default resampling-on arm from E1 on; `ess_min=0` stays as a control. Proofs are in the notes (Sec. 7).

## Reproducing TWISTER Figure 1(a)

`experiments/twister_fig1.py` follows TWISTER Sec. 4.3. The paper does not state the sequence length, the pool size
or the exact pairwise-TV averaging, so these are choices: N = 32; pool = 8 tokens (two per class, giving ≈3%
acceptance for "a.*", consistent with the paper's ≈0.09 noise floor); TV averaged over all position pairs.
Three random partitions are averaged (two of the three drew the same class *a* = {`,`, ` of`}, so two distinct constraints). Their code was not public as of 2026-10-15. Absolute TV values differ from the paper's (their pool size, N and TV averaging are unstated), but the shape matches: step-exact bias grows ~4× over T, and K = 8 tracks the floor more closely than K = 4.

## References

- Thimmaiah, Marinov, Srinivasa, Vikalo, Li & Gligoric. *Twist, Don't Tilt: Trajectory-Exact Constrained Decoding
  for Masked Diffusion Models.* arXiv:2609.35609 (2026). Prop. 3.4, 3.6, Def. 4.1, Thm. 4.2, Alg. 1–2.
- Dang & Ermon. arXiv:2607.07026 (2026). Step-exact automaton-constrained sampling for MDLMs.
- Sahoo et al. *Simple and Effective Masked Diffusion Language Models.* arXiv:2406.07524 (2024). OWT-130M.
- Carter & Kohn. *On Gibbs sampling for state space models.* Biometrika 81 (1994). FFBS.
- Aho & Corasick. *Efficient string matching.* CACM 18 (1975). Motif automata.
- Johnson, Griffiths & Goldwater. *Bayesian inference for PCFGs via MCMC.* NAACL (2007). Top-down chart sampling.
- Dechter. *Bucket elimination.* Artificial Intelligence 113 (1999). Variable elimination.
- Tutte (1948); Koo, Globerson, Carreras & Collins, EMNLP (2007). Matrix-tree partition and marginals.
- Wilson. *Generating random spanning trees more quickly than the cover time.* STOC (1996).
- Jerrum & Snir (1982). Monotone circuit lower bound for spanning trees.
- Nussinov & Jacobson. PNAS 77 (1980). RNA pairing grammar.
- Del Moral. *Feynman–Kac Formulae.* Springer (2004). SMC evidence estimator.
- Vehtari, Simpson, Gelman, Yao & Gabry. *Pareto Smoothed Importance Sampling.* arXiv:1507.02646. Zhang & Stephens,
  Technometrics 51 (2009). Tail diagnostic.
- Meng & Wong. Statistica Sinica 6 (1996). Bridge sampling.
- Gumbel (1954); Maddison, Tarlow & Minka, NeurIPS (2014). Gumbel-max sampling.
- Nie et al. *Large Language Diffusion Models.* arXiv:2502.09992 (2025). LLaDA.
- Luxembourg, Permuter & Nachmani. arXiv:2506.19037 (2025). Dilated unmasking (the `Dilated` schedule).
- Hesterberg. *Weighted average importance sampling and defensive mixture distributions.* Technometrics 37 (1995).
- Kitagawa. *Monte Carlo filter and smoother for non-Gaussian nonlinear state space models.* JCGS 5 (1996). Systematic resampling.

**Ambiguous routes are run without resampling.** The chart's twist is derivation-weighted, so resampling
concentrates particles on prefixes with many parse trees rather than high probability. A string-counting twist
would fix it and is #P-hard in general, so no exact fix exists. Fixed target structures compile to factor
graphs, which count each string once, and are unaffected.

No license file is included yet; choose one before making the repository public.

## License

MIT; see `LICENSE`.
