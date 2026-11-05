# Layer 3: the mRNAutilus adapter — native-process analysis

**Status:** design note, 2026-11-05. No adapter code written yet. This records what the mRNAutilus source and
checkpoint say about whether our exactness guarantee can hold against this model, and the two decisions that
need the user before building the adapter.

Sources read from `atombio/mrnautilus` (HF): `src/mrna/tasks/lm/mdlm.py`, `noise_schedule.py`,
`src/rinalmo/model/*`, and the checkpoint `best_public.ckpt` (artifact d4c53d94-b38a-488a-a043-48b2e0e40694).

## 1. The denoiser matches our interface [verified from source]

`Diffusion.forward(zt, sigma)` returns SUBS-parameterised log-probabilities over the 86-token vocabulary:
- the mask token is set to -inf and the row renormalised (`subs_parameterization`);
- revealed positions are forced one-hot on their current token (carry-over unmasking).

This is exactly our denoiser contract: `logits(x) -> (B, N, V)` log-probs with the mask excluded and revealed
positions clamped. `time_conditioning = False` (a default in the model code; it is not among the checkpoint hparams I inspected), so the
network ignores the noise level and is a pure function of the partially-masked sequence `zt` — no timestep
input to thread through. Token draws use Gumbel-argmax at temperature 1 with no truncation
(`sample_categorical`), which is the native kernel our proofs assume.

## 2. The architecture loads from the checkpoint alone [verified]

The net (`model.net.rm.*`, 399 tensors) is the RiNALMo transformer: an 86x640 embedding, 30 blocks of
fused-QKV rotary attention plus a SwiGLU transition (1706-wide gate), a final layer norm, and a mask head
(640->640->86). 150M parameters. No timestep or sigma embedding anywhere, consistent with (1). The model code
is self-contained (`src/rinalmo/model`, 464 lines) but imports `flash_attn`, which is GPU-only; there is a
non-flash `MultiHeadSelfAttention` class alongside the flash one.

## 3. The reveal schedule differs from our experiments [verified, and is the main decision]

TWISTER and our E1-E3 runs reveal exactly N/T positions per step, uniformly among the masked ones
(`reveal_sets`). mRNAutilus uses the standard MDLM reverse step (`single_reverse_step` / `cached_reverse_step`
with the log-linear schedule `sigma = -log(1 - (1-eps)t)`): at a step from t to s = t - dt, each still-masked
position **independently** unmasks with probability (t - s)/t and, if it does, draws a token from p_theta;
otherwise it stays masked. The number of positions revealed per step is random (Binomial), not fixed, and a
final denoise step fills any remainder.

**Why this is not a correctness blocker.** Our trajectory-exactness holds relative to *a* native kernel kappa,
for *any* reveal scheduler s(r|x), because s cancels in the potential (notes, Lemma "step-exact proposal":
M/kappa has no s term). Both schedulers are valid MDLM reverse processes with the same denoiser. So we can be
trajectory-exact against either. The choice is about *faithfulness*: to sample mRNAutilus's own finite-T
distribution exactly, the proposal and native kernel should use the model's independent-unmasking schedule.
With our fixed-N/T schedule we would sample a different (also valid) model-induced conditional, which differs
from the model's at finite T and coincides only in the T -> infinity limit.

**Work implied by matching it:** generalise the proposal's reveal step to a per-position independent schedule
(a random, possibly zero, reveal set per step), in `reveal_sets`, the native kernel, and `_propose`. The route
machinery is unchanged. This is generic layer-1 work, tested the same way (E[Zhat] = h0, two-sided identity),
but it changes a core loop, so it is held pending the decision below.

## 4. Their conditional sampler imposes protein identity by masking [verified]

`mask_logits_vectorized_conditional` restricts each position's logits to the synonymous codons of the template
residue. mRNAutilus's own protein-conditioned generation therefore renormalises the per-step predictions onto
the allowed codons -- exactly the step-local projection our method replaces, because it biases the trajectory
relative to the true conditional p(x | protein). This strengthens the motivation: the comparison "our exact
conditional vs their masked sampler" is a like-for-like test of the trajectory correction on this model.

## 5. MCTS is the comparator, not the native process [verified]

Their headline sampler is Monte Carlo tree search over multiple learned objectives (half-life, ribosome load).
That is a mode-seeking, guided procedure -- the per-step-MAP / optimiser arm of the E6b best-of-K comparison,
drawn from the same model. The base reverse process in section 1-3 is what we condition.

## 6. Numerics [verified in principle]

86 tokens, 64 of them codons, spread probability thinner than our 4-class experiments. The float64 automaton
pass and the support guard (both added 2026-11-04/05) are the relevant protections. SUBS renormalises in
float32, which is adequate for the denoiser output; route sums are the sensitive part and are already guarded.

## 7. Adapter plan (once the decisions below are settled)

1. Vendor `src/rinalmo/model` into `twistroute_bio/models/_rinalmo/` (464 lines, self-contained) and load the
   checkpoint's `model.net.rm.*` tensors. Run on GPU with flash-attn (the faithful path), or adapt the
   non-flash attention class for a CPU smoke test.
2. `MRNAutilus` class satisfying the denoiser interface (`mask_id`, `V = 86`, `logits(x)`), applying SUBS.
3. `codon_classes()` -> TokenClasses mapping the 86 tokens onto a constraint alphabet, excluding non-codon
   tokens at CDS positions.
4. Smoke test: sample a short CDS with one motif constraint; check every output satisfies it and Zhat behaves.

## 8. Decisions for the user

1. **Schedule faithfulness.** Match mRNAutilus's independent-unmasking schedule (exact against the model's own
   finite-T distribution; needs the layer-1 scheduler generalisation in section 3), or use our fixed-N/T
   schedule (simpler, also exact against a valid model-induced conditional, differs at finite T)? Recommend
   matching it, since the paper's claim is "exact sampling of the model's constrained distribution".
2. **How to run the model.** The faithful path is a GPU job on Engaging with flash-attn (as the model was
   trained and run). Confirm that is where layer-3 experiments should run; a CPU smoke test is possible with
   the non-flash attention path but is not how the model is meant to run.
