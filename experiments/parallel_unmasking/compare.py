"""
Policy against the fixed schedule on oracle models, measured against the exact target p(. | C).

Cells are (model, constraint, N). Models: Markov chains with two correlation lengths (a revealed position
separates its two sides exactly), a hidden chain (dependence decays but nothing separates), and a chain cut
into independent segments. Constraints: forbidden motifs (local), exactly-k (global, weak), same-order copy
(hard, pairwise, long-range), Dyck-1 (nested). Tokens are classes.

Arms per cell
    fixed          the existing schedule at 1, 2, 4, 8 reveals per step (1 per step samples the target itself)
    policy         CouplingPolicy with a pilot kernel, priority "bound", thresholds THETAS, without the mixed term
    small pilot    the same with a kernel fitted on few traces
    priorities     "confident" and "random" at two thresholds
    mixed          with the mixed-path term, fewer runs (it costs N * A extra route passes per particle per step)

Metrics (pooled over runs; standard errors by a delete-one-batch jackknife over 10 batches of runs)
    steps          reveal steps per particle = denoiser calls
    marg_tv        mean over positions of TV(weighted class marginal, exact marginal)
    pair_tv        the same for adjacent pairs (automaton constraints only)
    z_ratio        mean Z_hat / exact p(C)
    sd_r, kl_r     spread of the parallel-reveal gap r over the output distribution, and Var(r) / 2
    ess            mean over runs of the minimum effective sample size fraction
    kept           effective fraction of outputs left if each were weighted by exp(r)
    *_corr         marg_tv and pair_tv after that weighting
    route_passes   batched route passes per particle per step requested by the policy
    top1_share     share of the sum of Z_hat carried by the top 1% of runs
    hill_k         Hill estimate of the upper-tail index of Z_hat over runs (top 2%); above 1/2 the variance is not
                   finite and z_ratio_se is flagged as unreliable (z_se_unreliable)

Guards against hidden mass (notes Sec. 6). Every arm uses defensive resampling (eps = 0.25) except the arm
"fixed 1/step, no resampling", which cannot hide mass. Each cell records `control_check`: mean Z_hat of the
one-per-step arm with and without resampling. They estimate the same p(C); a gap is the truth-free sign that
resampling is hiding mass. Per-run log(Z_hat / p(C)) of every arm is saved to <out>.logZ.npz.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulib.cells import MODELS, CONSTRAINTS, stream_seed, make_model, make_constraint          # noqa: E402,F401
from pulib.metrics import metrics, Footprint                                                    # noqa: E402,F401
from pulib.gate import GATE_ALPHAS, resampling_gate                                             # noqa: E402,F401
from pulib.arms import (ARM_SETS, Arm, Cell, THETAS, BUDGETS, THICK_SMALL_BUDGETS, THICK_WIDTHS, THICK_SPACINGS,    # noqa: E402,F401
                        SEQ_ORDERS, RESAMPLE_KS, INTERACT_BUDGETS, INTERACT_ORDERS)
from pulib.runner import run_cell, main                                                         # noqa: E402,F401

if __name__ == "__main__":
    main()
