"""Decoders: route-generic twisted SMC (GPU, batched), native sampling, and exact toy path laws (CPU)."""
from .smc import twisted_smc, select_particles, reveal_sets, clamp_revealed
from .native import native_sample, native_path_weights
