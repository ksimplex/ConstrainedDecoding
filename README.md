# twistroute

Exact constrained sampling for masked diffusion language models (MDLMs), for any constraint that compiles to a
tractable *route*: automata (regular languages, motifs, counting), grammar charts (context-free, including
ambiguous grammars such as RNA pairing), factor graphs (copies, low-width relations) and spanning trees.

Decoding is twisted sequential Monte Carlo in the style of TWISTER (Thimmaiah et al., 2026), with the automaton
replaced by a route interface. Every output satisfies the constraint, the weighted particles target the model's
own conditional `p(x | x ∈ C)`, and the average weight `Ẑ` is an unbiased estimate of `P_model(C)` at every
particle count.

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
| factor graph | copy, low-width relations | variable elimination | backward sampling |
| determinant | arborescences | matrix-tree theorem | Wilson's algorithm |

### Options of `twisted_smc`

| option | effect |
|---|---|
| `ess_min` | resample a run when ESS < `ess_min`·K (0 = no resampling) |
| `defensive=ε` | ancestors from (1 − ε)w̃ + ε/K with offspring weight w̃/π; recommended ε = 0.25 |
| `resampler` | `"multinomial"` (default), `"cpu"`, `"systematic"` |
| `twist_power=β` | tempered intermediate twists η^β (1 = TWISTER) |
| `lookahead=j`, `lookahead_draws=m` | look-ahead twist: η times the mean potential product of m speculative step-exact rollouts of j steps; an unbiased estimate of the j-step-ahead frozen twist, and of the Doob function when j reaches T (experimental) |
| `mix_native=a` | propose from (1 − a)·step-exact + a·native kernel; bounds each potential by ψ′/(aψ) (experimental) |

All options keep E[Ẑ] = h₀ (tests/test_smc.py, tests/test_resampling.py). The maths is in `docs/twistroute_notes.pdf`.

## Layout

```
twistroute/
  compile/        CPU reference routes (clear, brute-force verified): automata, grammars, factor graphs, trees
  torch_routes/   batched GPU routes: TorchChain, TorchChart, TorchCopy, det_logZ; build_route(spec)
  decode/         twisted_smc (smc.py), native sampling + two-sided weights (native.py), exact toy laws (exact.py)
  models/         OWT-130M wrapper, synthetic denoiser, token-class projections (semantic, TWISTER-ranked)
  diagnostics.py  PSIS tail shape, bootstrap CIs, subsample curves, pairwise TV with a matched floor,
                  hidden-mass decomposition of weight traces
  experiment.py   shared plumbing for the scripts: denoiser loading, Fig. 1(a) setup, batched SMC runs,
                  native rejection counts, Z_hat summaries
experiments/      E1, E2, E3, E4, resampling arms, TWISTER Fig. 1 reproduction, CPU controls; analysis/ and
                  configs/ (cell specs)
tests/            pytest suite (~6 min on CPU)
docs/             working notes (LaTeX + PDF): theory, algorithms, proofs
```

## Install and test

```bash
pip install -e ".[test]"          # numpy, scipy, torch, pytest
pytest                            # 128 tests, ~6 min on CPU, no GPU or model weights needed
```

For OWT-130M: `pip install -e ".[models]"` plus `flash-attn` (its remote code imports it). On torch 2.5.1 + cu124
the prebuilt wheel `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE` works; building from source failed against
a CUDA 12.9 system toolkit.

The tests cover:
- every CPU route's `logZ` against enumeration, chart membership against an independent parse counter, the
  max semiring against brute-force MAP, and the compiler's refusal past budget;
- every exact sampler against the enumerated law (TV within 2× the multinomial noise floor, zero invalid draws);
- every batched route against the CPU compiler (with revealed positions), membership on random strings, and the
  law of one batched sampling call;
- exact path laws on toy problems: the Feynman–Kac law equals the Doob law and `Ẑ/h₀ = 1` to 1e-10 for six
  families, including an ambiguous grammar; step-exact decoding is biased unless the denoiser is frozen;
- `twisted_smc` on a synthetic denoiser with strong re-conditioning: mean `Ẑ` matches native `h₀` (|z| < 4) for
  five route types at K = 1 and 4 with resampling active; the two-sided identity `E_p[1_C / w] = 1`;
- golden values from the code that produced E2, so behavioural changes are caught as well as statistical ones;
- the resampling options: every resampler draws ancestors with the requested probabilities (systematic counts
  within one of K·π), a defensive step preserves the weighted measure exactly with offspring weights at most
  1/(1 − ε), defensive resampling, tempered twists and the alternative resamplers keep E[Ẑ] = h₀, the weight
  trace agrees with Ẑ, and out-of-range options are rejected;
- the diagnostics: the hidden-mass decomposition on a constructed trace, the perfect-twist bound (mass below ε
  at most K·ε), and the matched TV floor;
- end-to-end dry runs of `resampling_arms.py`, `e1_families.py` and the E1 analysis on the synthetic denoiser.

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
| E1 trajectory bias across families | `experiments/e1_families.py`, `experiments/analysis/e1_analysis.py` | running: six families at N = 32, T = 16; step-exact, K = 4/16 defensive, K = 16 plain and no-resampling |
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

No license file is included yet; choose one before making the repository public.
