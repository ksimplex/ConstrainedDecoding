# Prior work on dependency-aware parallel unmasking: what it means for us

Date: 2026-11-15. Basis: five readers each read one paper's full text (arXiv HTML) against a fixed question list. I have
read their reports, not the papers. Performance numbers that the readers took by eye from figures are left out.
Statements marked (inference) are ours, not the papers'.

Per-paper notes: [lit_demask.md]({{artifact:f6fca248-75ff-4850-9e8d-433ba70c5a3f}}), [lit_punt.md]({{artifact:e95a009c-0e21-4b76-baed-dcae0bf5a633}}), [lit_coupling.md]({{artifact:68b1bb24-c896-4251-b505-522ea88fd1ae}}), [lit_schedules.md]({{artifact:6a65048b-0706-41ad-a205-feffda82b9a6}}), [lit_sirs.md]({{artifact:89ce5aef-900a-4703-aba9-4acaec6d379f}}).

## 1. The five papers in one table

| paper | what it does | dependence estimate | positions fixed before values? | target / guarantee | constraints | weights or correction |
|---|---|---|---|---|---|---|
| DEMASK (Ringel, Ali, Romano; arXiv:2604.02560) | greedy set under a cumulative pairwise-dependency budget tau, plus a confidence gate; left-most position always revealed | expected TV shift of the prediction at i when j is revealed; predicted by a learned 26M-parameter head in the same forward pass, per state | yes | per-step TV(joint, product) <= tau under a sub-additivity assumption (Thm 4.2); assumption tested empirically | no | no |
| PUNT (Azangulov et al.; arXiv:2510.21961) | drops lower-confidence positions whose prediction moves by more than eps in KL when higher-confidence candidates are filled in; about log2(m) extra passes per step | set-level KL shift measured online with the sampled candidate tokens | no: the set is a function of the sampled values | no theorem; an 'independence stability' assumption | no | no |
| Measured coupling (Oved, Wolf; arXiv:2611.08307) | greedy argmax decoder; commits positions whose mean-field commit score passes a threshold | signed change in log-margin when another position is set to its argmax; measured by probe rows inside the main pass | no (deterministic argmax) | no target law; benchmark accuracy | no | no |
| Chen, Cong, Li (arXiv:2511.04647) | theory of how many positions to reveal per step when positions are uniformly random | none (works from the 'information curve' of the distribution) | yes (positions random) | expected KL of any step-size schedule, exactly; impossibility of a universal adaptive schedule; O(min(TC, DTC) log n / eps) steps | no | no |
| SIRS (Zhang, Li, Di, Gu; arXiv:2610.22238) | per-block rejection step that turns the product draw into the block joint; acceptance estimated from few single-increment queries | none; agnostic to how blocks are chosen | yes | exact without clipping; KL bound with clipping; expected queries per step <= (1 + 2B) e^(2B) | no | rejection |

Correction to the earlier side-chat note: the O((TC + DTC) / eps) rate belongs to Li and Cai; Chen, Cong and Li prove
O(min(TC, DTC) log n / eps).

## 2. What is not new in our work

- The greedy threshold rule. DEMASK's Algorithm 2 is our rule in structure: a pairwise expected-shift matrix (TV there, KL
  here), summed coupling to the positions already chosen, a threshold, and positions fixed before values.
- Measuring dependence by revealing one position and reading the change at another. All three method papers do this.
- The additivity assumption behind summing pairwise couplings. DEMASK states it (Assumption 4.1) and tests it; Oved and
  Wolf report that sums stay accurate for sets of 2 to 3 and degrade for sets of 8 or more, overstating the joint effect.

## 3. What none of the five does

- Hard constraints. Our split (constraint coupling exact through the route, only model coupling estimated) and the
  forced-position correction have no counterpart.
- Importance weights: a sampler that stays exact for the law of its own schedule, with the gap to the target carried
  along paths as r and audited.
- Any study of how the reveal order affects particle weights. Order appears only as confidence heuristics.
- A priority designed to create separation. DEMASK picks the least-dependent candidate next, which enlarges the current
  set; it does not consider what the reveal does for later steps.

Suggested framing: dependency-aware parallel unmasking exists for unconstrained decoding; we extend it to constrained
decoding, where the constraint part of the dependence is exact and the sampler remains exact for its schedule, and we
study the reveal order because it changes the weights.

## 4. Ideas worth taking (all untested here)

1. **Sub-additivity audit of our kernel** (DEMASK Sec. 5.4; Oved-Wolf App. J). Compare the set-level shift with the sum
   of pairwise kernel values on the sets our rule selects, by set size. This is our planned 'predicted against measured
   r' check; their evidence says our sum is probably conservative for large sets.
2. **A total per-step budget** in place of a per-candidate threshold (DEMASK). Under sub-additivity the per-step bound
   then does not grow with set size.
3. **Least-coupled-first** as a further priority arm (DEMASK), to compare against the separator score.
4. **State-dependent kernel on real models**: a small learned head on hidden states (DEMASK) or in-pass probes (Oved-Wolf)
   in place of our pooled pilot kernel. For probes, the probed value must not be the value then committed.
5. **Theory for the error of a schedule.** For a fixed partition, KL(target, sampler) is the sum of the conditional total
   correlations of the blocks (proof of Thm 3.3, Chen et al.), which is E[r] in our notation (inference). SIRS proves its
   bounds for any block-selection rule that depends only on the revealed state, which is our setting; that proof is a
   template for history-dependent sets.
6. **Set sizes from an estimated information curve** (Chen et al.): the principled form of coarse offline planning. Their
   guarantee is for uniformly random positions and exact marginals, so for us it is a guide and a baseline.
7. **A correction for theta > 0 as a random weight** (from SIRS, inference). The missing weight factor for a co-revealed
   set is a pure model quantity, and the Poisson construction gives a non-negative unbiased estimate of it from a few
   extra denoiser calls. In SMC that would turn the fixed bias into weight variance. Open: the variance grows with set
   size and the estimate is heavy-tailed; all of SIRS's own evidence is on synthetic models with exact scores.
8. **Value-free version of PUNT's test** (inference): run it with throwaway values to choose the set, then draw the
   committed values through the route. Costs about log2(m) extra passes per step.

## 5. What to read ourselves before writing

DEMASK in full (closest; also for its related-work section, which names KLASS, APD and Bansal-Sanghavi), then SIRS
(Sections 4, 5 and 7), then the proof of Thm 3.3 in Chen, Cong and Li.
