"""Evaluation (notes, Table 3 and Figures 1 and 2).

exact   exact laws of the sampler's output at a short length, and their KL from the target
bounds  bounds on that KL from the weights, for lengths where the exact law is out of reach
spread  spread of the estimate of P(x_0 in C) across runs
"""

from .bounds import bracket, leading_term, marginal_lower_bound, upper_bound
from .exact import all_sequences, kl, one_particle_law, output_law, target_law
from .spread import dead_runs, mean_ratio, rms_log_ratio, sd_log_Z_hat

__all__ = [
    "all_sequences",
    "kl",
    "target_law",
    "one_particle_law",
    "output_law",
    "upper_bound",
    "leading_term",
    "marginal_lower_bound",
    "bracket",
    "sd_log_Z_hat",
    "rms_log_ratio",
    "mean_ratio",
    "dead_runs",
]
