"""Pair tables K(a, b) for the chain twist (notes, text before eq. 11): an estimate of the
probability that class b follows class a at adjacent positions. Any positive table keeps the
sampler exact; a better table gives a better proposal.

    counted_pair_table  frequencies of adjacent classes in unconstrained samples of the model
    probed_pair_table   the model's own conditionals: unmask one position of the fully masked
                        sequence, call the model, read the next position (a few batches per
                        prompt; no samples needed, so it is usable with a large model)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..decode.chain_twist import fit_pair_kernel, probe_pair_kernel


def counted_pair_table(
    model: Any,
    token_classes: Any,
    n_samples: int,
    generator: Optional[torch.Generator] = None,
    *,
    smoothing: float = 1e-3,
    length: Optional[int] = None,
) -> torch.Tensor:
    """(A, A) row-stochastic table from `n_samples` unconstrained samples; `smoothing` is added
    to every count so that no entry is zero."""
    return fit_pair_kernel(model, token_classes, n_samples, generator, smooth=smoothing, N=length)


def probed_pair_table(
    model: Any,
    token_classes: Any,
    length: int,
    generator: Optional[torch.Generator] = None,
    *,
    stride: int = 1,
    top_classes: int = 4,
    min_mass: float = 0.01,
    tokens_per_class: int = 2,
    floor: float = 1e-4,
    batch_rows: int = 16,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """(A, A) row-stochastic table from the model's conditionals at the fully masked sequence,
    and a record of what it cost (number of probes and model batches)."""
    return probe_pair_kernel(
        model,
        token_classes,
        length,
        generator,
        stride=stride,
        top=top_classes,
        min_mass=min_mass,
        tokens_per_class=tokens_per_class,
        floor=floor,
        rows=batch_rows,
    )
