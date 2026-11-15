"""Decoders: route-generic twisted SMC (GPU, batched), native sampling, and exact toy path laws (CPU)."""
from .smc import twisted_smc, select_particles, reveal_sets, reveal_indices, clamp_revealed
from .native import native_sample, native_path_weights
from .policy import (CouplingPolicy, DilatedSchedule, ThickSeparatorSchedule, DistanceKernel, COUNT_EDGES, fit_distance_kernel, constrained_marginals,
                     pairwise_mi)
