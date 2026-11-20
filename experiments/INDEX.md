# Experiments: what is where

Moved on 2026-11-20 from `experiments/parallel_unmasking/` (branch `notes-structure`). Every script was `experiments/parallel_unmasking/<name>.py` before;
results stay in `results/parallel_unmasking/`. The status document refers to scripts by their old paths; this file is the map.
Scripts at the top level of `experiments/` and the folders `analysis/`, `configs/`, `controls/` are from before this line of work and are unchanged.

## `experiments/notes/`

Tables and figures of the notes, from the new core (`twistroute.sampler`, `twists`, `schedules`, `evaluation`).

| Script | What it does |
|---|---|
| `cells.py` | The oracle model and the constraints of the notes' experiments. |
| `table3.py` | Table 3 of the notes: exact KL divergence (nats, lower is better) from the law of the sampler's output to the constrained target, on the hidden-chain oracle with L = 8 positions, one position unmasked per step, product proposal, no resampling. |

## `experiments/benchmarks/`

Real models and text benchmarks: LLaDA and MDLM; JSON-Mode-Eval, BFCL, xLAM. These need model weights, datasets and, for the runs, a GPU (docs/setup.md). They call the earlier sampler `twistroute.decode.twisted_smc`.

| Script | What it does | Recorded in status sections |
|---|---|---|
| `bfcl_compile.py` | Compile a sample of BFCL single-turn examples into token-level routes for one tokenizer (CPU), and report sizes. | 36aq |
| `bfcl_constraints.py` | Regular-expression constraints for BFCL function calls, in the two output formats of Dang & Ermon (2026). | 36aq |
| `bfcl_eval_outputs.py` | Score stored BFCL outputs (llada_json.py --stage baseline / compare on routes from bfcl_compile.py) with the official checker. | 36av |
| `bfcl_score.py` | Score function calls with the official BFCL checker (package bfcl-eval, installed with --no-deps). | 36aq |
| `json_classes.py` | How many token classes does a JSON constraint have under a real tokenizer? | not named there |
| `json_mode_classes.py` | Token classes of the JSON-Mode-Eval schemas under a real tokenizer (the setting of TWISTER, appendix F). | not named there |
| `json_mode_compile.py` | Compile JSON-Mode-Eval (NousResearch) into token-level routes for one tokenizer, once, on CPU. | 36an |
| `llada_json.py` | Constrained decoding of JSON-Mode-Eval on LLaDA-8B-Instruct: product against matched-chain proposal. | 36ar, 36av (named there by its stages) |
| `llada_pilot_check.py` | Score the unconstrained LLaDA pilot samples (llada_json.py --stage pilot) on JSON-Mode-Eval. | 36aq |
| `mdlm_first_look.py` | First look at parallel unmasking on a real masked diffusion language model (MDLM, OpenWebText), with no exact answer to compare against. | 28 |
| `mdlm_order_consistency.py` | How order-consistent is a masked diffusion model? | 36p, 36v |
| `mdlm_orders_output.py` | Planned orders and the chain proposal on a real masked diffusion language model (MDLM, OpenWebText). | 36ak |
| `mdlm_timing.py` | Where does the time of one sampler step go on a real masked diffusion language model? | 36j |
| `xlam_constraints.py` | xLAM (Salesforce/xlam-function-calling-60k) in the form bfcl_constraints.py expects. | 36at |

## `experiments/studies/`

Studies on oracle models that current claims rest on, and the library they share (`pulib/`, `twist_error.py`, `surrogate_plan.py`). They call the earlier sampler.

| Script | What it does | Recorded in status sections |
|---|---|---|
| `compare.py` | Policy against the fixed schedule on oracle models, measured against the exact target p(. \| C). | 8, 8b, 9, 10, 20, 34, 35, 36a, ... |
| `forced_batch_check.py` | Exact check of the forced-only batch step (CouplingPolicy(forced_batch=...)) at a size where every sequence can be listed. | 36ap |
| `kl_bounds_check.py` | Bounds on the error of the returned sample's law, checked where the law itself is known (8 positions, one token per step, no resampling). | 36as |
| `parallel_exact.py` | Parallel reveals, exactly: the best trade of steps against error at a length where every state can be tabulated. | 36y, 36au |
| `parallel_exact_check.py` | Cross-check of the sampler against the exact tables of parallel_exact.py at 8 positions. | 36z |
| `particle_rule_check.py` | How many particles for a tolerance on the returned sample's law? | 36ai |
| `report.py` | Standard report of comparison outputs, and the log of results across settings. | 36a |
| `surrogate_plan.py` | Can the ideal reveal order be found without the exact twist? | 36t, 36w |
| `twist_error.py` | How wrong is the twist, and does a first-order chain stand-in fix it? | 29, 35, 36f |

Also here: `pulib/` (cells, arm sets, runner, metrics, summary: the library behind `compare.py`), `jobs/` (queue scripts as they were run, with paths
updated to the new folders), and `README.md` (the layout of the comparison studies).

## `experiments/archive/`

Questions that are answered. Kept runnable (they import the shared library from `studies/`); the answers are in the sections of `docs/parallel_unmasking_status.md` listed.

| Script | What it does | Recorded in status sections |
|---|---|---|
| `best_plan_exact.py` | The ideal reveal order, exactly, at a length where every partially revealed state can be enumerated. | 36s |
| `collective_check.py` | Why is the pair account poor on a sum constraint? | 36o |
| `control_check.py` | Is the one-reveal-per-step control unbiased in practice? | 8 |
| `counting_orders.py` | Where along a decode does each order pay on a sum constraint, and does revealing well-connected positions early lower the spread of the remaining count for later steps? | 36q, 36w |
| `error_estimate_check.py` | Does the disagreement between the chain twist and the product twist estimate the product twist's error? | 36g |
| `exposure_check.py` | Does exposure predict the variance of the weights? | 36b |
| `first_look.py` | First look at the coupling policy on oracle chains (CPU, small). | 3, 28 |
| `hidden_mass_check.py` | Is the accuracy loss at long lengths hidden mass (notes Sec. | 11 |
| `order_attribution.py` | Attribute a sequential order's KL(sampler \|\| exact law) to the positions it reveals: the sum over steps of the true error KL(proposal marginal \|\| exact conditional) at the revealed position, split by distance from the window. | 36i |
| `pair_cost_check.py` | Does a pairwise account of the model's dependence predict the true cost of revealing a position? | 36n |
| `plan_check.py` | Does one step of planning on an estimated cost beat the greedy use of the same estimate? | 36r |
| `regularity_check.py` | Are sequences that satisfy a forbidden-motif constraint unusually regular? | 36m |
| `reveal_stages.py` | Where does the time of the schedule go over a whole decode? | 36l |
| `revision_check.py` | Does the revision of the twist after a reveal predict the true per-step error? | 36ad |
| `twist_order.py` | What does raising the order of the chain twist buy, and what does it cost? | 36h |
| `twist_smc.py` | Does a better twist fix the weights? | 29, 35 |
| `weight_trust.py` | Do intermediate weights predict final weights? | 15, 18, 20 |
