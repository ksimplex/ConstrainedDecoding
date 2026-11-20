"""An estimate of the dependence between positions under the model, for the constraint-aware
schedules (`Sum`, `BudgetedCoReveal`).

M(i, j) estimates the mutual information, in nats, between the tokens at two masked positions
given what is unmasked. It is pooled by the distance |i - j| and by the number of unmasked
positions strictly between i and j (an unmasked position in between screens the two from each
other). It is fitted once, before sampling, from unconstrained runs of the model: after each
single-position unmasking, the KL divergence between the model's new and old prediction at
every other masked position is recorded. Cost: L + 1 model calls of batch `n_samples`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from ..decode.policy import COUNT_EDGES, DistanceKernel, fit_distance_kernel

DistanceDependence = DistanceKernel


def fit_distance_dependence(
    model: Any,
    length: int,
    n_samples: int,
    generator: Optional[torch.Generator] = None,
    *,
    device: str = "cpu",
    screening_bins: Sequence[int] = COUNT_EDGES,
) -> DistanceDependence:
    """Fit M from `n_samples` unconstrained runs. `screening_bins` are the edges of the bins of
    the number of unmasked positions in between (default: 0, 1, 2, 3-4, 5-8, 9-16, 17 or more)."""
    return fit_distance_kernel(
        model, length, n_samples, generator, device=device, count_edges=screening_bins
    )[0]
