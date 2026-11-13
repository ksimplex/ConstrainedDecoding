# Validating the vendored mRNAutilus encoder, and what the model is good at

**Date:** 2026-11-06. **Status:** implementation validated; one negative result on codon choice that shapes
constraint selection. All numbers below are measured; the testbed caveat in section 4 is important.

## 1. Implementation is correct

| check | result |
|---|---|
| transformer block vs upstream's own non-flash class, random weights, identical inputs | max abs diff **0.0** (bit-identical) |
| my flash vs my torch attention, real weights, GPU (job `b4fa07f0` lineage) | max abs diff 0.21 on logits of scale 4.7 = bf16 rounding |
| checkpoint load | 399 tensors, no missing or unexpected keys (token_dropout is parameter-free) |
| context sensitivity (section 3) | large, so attention is active |

Two bugs were found and fixed while reaching this point, both invisible in the checkpoint:

- **SwiGLU gate order.** Upstream applies Swish to `linear` and multiplies by `linear_gate`; I had it reversed.
- **TokenDropout was missing.** It has no parameters, so it does not appear in the state dict, but the RiNALMo
  config sets `active: True` and it applies at inference: masked embeddings are zeroed and the whole sequence
  rescaled by (1 - 0.12) / (1 - observed mask rate). Measured effect on masked-codon accuracy: 0.061 with it,
  0.000 without (5% masking); 0.049 vs 0.010 (15% masking).

Alphabet confirmed: codons at indices 5-68, nucleotide codes 69-85 (`Alphabet_uni` default ordering, which the
config's `any_tokenizer_discrepancies` check enforces). The `default_config["alphabet"]` entry lists the
opposite order but is not used for tokenization.

## 2. Codon choice: the model beats codon usage bias once it is measured properly

Task: mask codon positions, restrict the model's output to the synonymous codons of the true residue (which is
what our setting does, since the protein is fixed), and take the top-1 codon. Reference is not chance but the
**frequency baseline**: always pick the commonest synonymous codon in the human genome.

n = 548 masked positions (30 draws of 20 positions), paired on position:

| | accuracy |
|---|---|
| model | 0.398 |
| frequency baseline | 0.387 |
| chance (1 / synonymous set size) | 0.295 |

Paired: 152 both correct, 66 model only, 60 baseline only, 270 neither. McNemar z = +0.53 on 126 discordant
pairs: **not significant**.

Both beat chance, so the model has learned codon usage. At top-1 synonymous choice under THIS protocol it
adds nothing measurable over a lookup table.

### 2a. The protocol above understates the model (2026-11-13)

Two things were wrong with reading the table above as "the model does no work".

**The context was mostly removed.** Each draw masks 20 positions at once, so every prediction conditions on a
sequence with 20 holes in it. That is not the setting the sampler runs in: constrained decoding conditions on
everything already decided, which late in a trajectory is nearly the whole sequence. The right question is
what the model knows given maximal context, which is one position masked at a time.

**Top-1 accuracy is the wrong instrument.** A natural CDS is one draw from a high-entropy conditional, so even
a perfect model of that conditional will frequently rank the observed codon second. Near-baseline accuracy is
weak evidence of a weak model. The field agrees by its practice: no published learned method reports
per-position reconstruction accuracy as a primary result (`codon_benchmark_landscape.md`), and mRNAutilus's own
paper validates its pretraining distributionally -- GC, length-normalised MFE, Kozak frequency, pairwise
diversity, per-sequence entropy -- never by accuracy or likelihood.

Rescoring with one position masked at a time, every other position revealed, and scoring the log-probability
assigned to the OBSERVED codon (a proper scoring rule, uniquely optimised by the true conditional), both
predictors renormalised over the same synonymous set. n = 307 positions, 40 sampled positions in each of 8
genes (`results/codon_scoring.csv`):

| | value |
|---|---|
| mean log-probability gain over the frequency table | **+0.164 nats** (95% CI +0.100 to +0.229) |
| paired t / Wilcoxon | t = 4.97, p = 1.1e-6 / p = 3.4e-11 |
| positions where the model is closer | 229 / 307 (74.6%) |
| genes with a positive gain | 8 / 8 |
| per-position perplexity | model 2.465 vs frequency 2.905 |
| top-1 on the SAME positions | model 64.2% vs frequency 53.1%, McNemar z = 4.49 |

Top-1 moves from 39.8/38.7 (z = +0.53) to 64.2/53.1 (z = 4.49) on the same model and the same metric. The
whole difference is how much context was revealed. So the triviality warning does not apply to this family:
the model's conditional carries real information about codon choice beyond a lookup table, and the earlier
negative read was an artefact of masking 20 positions simultaneously.

Caveats: this is a pseudo-likelihood (a sum of per-position conditionals), not a joint likelihood -- masked
diffusion models give only an ELBO for the latter. It is also the right object for the design question, which
is per-position codon choice given context rather than the probability of a whole transcript. 8 genes, 40
positions each, one masked position at a time.

Earlier, weaker runs that pointed the same way: single-position masking gave model 0.435 vs baseline 0.522
(n = 23, far too small to interpret); 20-position masking gave 0.420 vs 0.366 (n = 112, inside the noise).

**Unmasked-codon prediction (no synonymous restriction) is near chance** (0.06 vs 0.016) and is an
uninformative measure: predicting a codon outright requires guessing which amino acid belongs there, and
context barely determines that. It should not be used as a health check.

## 3. But the joint distribution is not a product of frequency tables

Total variation between the predicted codon distribution at one masked position under the real context and
under a perturbed context:

| perturbation | TV |
|---|---|
| one distant codon flipped | 0.002 - 0.006 |
| whole context randomised | 0.336 |
| synonymous shuffle (protein preserved, codon choices resampled) | 0.261 |

The last row is the one that matters. With the protein held fixed, changing *which synonymous codons* appear
elsewhere shifts the conditional at this position by TV 0.26. The frequency baseline is by construction a
product of independent per-position distributions; the model's joint is not. So the two can agree on per-position
argmax (section 2) while differing substantially as distributions over whole sequences.

This is the quantity our method targets: we sample from the model's joint conditional, not from a product of
marginals. Top-1 accuracy is the wrong lens for that, and section 2 should not be read as "the model is useless
here" -- only as "its mode matches the frequency table".

## 4. Caveats

- **One sequence**, the `fluc` template from mRNAutilus's own `config.yaml`. It may be in training data.
  Memorisation would inflate accuracy rather than depress it, so section 2's negative result is not explained
  by it, but a multi-sequence replication is needed before the number is quoted anywhere.
- Section 3's perturbed contexts are somewhat out of distribution (randomised codons, unusual codon usage).
  The conclusion "context matters" is safe; the exact TV values are testbed-specific.
- Top-1 accuracy was measured; likelihood or KL against the frequency-product model was not. That is the
  cleaner distributional comparison and is not yet done.

## 5. Consequences for the experiment design

1. **Do not build the headline experiment on plain codon optimisation.** A frequency table matches the model's
   mode, so the constraint family would not exercise the model's distribution.
2. **Prefer constraint families where a rule-based construction fails or biases**: target-structure
   compatibility, motifs that straddle codon boundaries, GC bands over windows, conjunctions of these. The
   rule baseline (`twistroute_bio.score.rules.rule_construct`) is reported per family to make this visible.
3. **Report the frequency baseline alongside every family**, as planned.
4. **Diversity and joint-distribution fidelity are the axes where the model contributes**, consistent with
   section 3 and with the library-generation framing.

## 6. Re-conditioning strength (the quantity that decides whether our correction matters)

If a masked diffusion model's prediction at a still-masked position did not change as other
positions are revealed, the model would be mean-field over the trajectory. In that case the frozen
twist equals the true twist, step-exact decoding is already unbiased, and the whole particle/twist
apparatus is unnecessary: a single route sample draws from p(x | C) exactly. The value of this
method therefore scales with how strongly the model re-conditions.

Measurement: hold one codon position masked throughout; reveal the other CDS positions in random
order in four stages; record the predicted 64-codon distribution at the held-out position at each
stage. 6 held-out positions, one CDS (the `fluc` template), CPU, non-flash path.

| fraction of other positions still masked | TV of the prediction at the held-out position |
|---|---|
| 1.00 -> 0.75 | 0.291 |
| 0.75 -> 0.50 | 0.134 |
| 0.50 -> 0.25 | 0.111 |
| 0.25 -> 0.00 | 0.121 |
| all-masked -> all-revealed | **0.337** |
| same, restricted to the synonymous set (n=5) | **0.205** |

Mean-field over the trajectory would give 0 in every row. The largest single movement (0.291) occurs
in the first quarter of reveals, which is exactly where step-exact decoding commits its earliest and
most consequential choices. The synonymous-restricted figure (0.205) is the one relevant to our
setting, since protein identity is fixed by the route.

Caveat: n=6 positions on one sequence; this is an order-of-magnitude indication, not a calibrated
estimate. It should be repeated on several real human CDSs alongside item 3 of the open list.
