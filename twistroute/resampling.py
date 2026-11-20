"""Resampling (notes, section 2.3): which particles survive, and with what weights.

After a step the K particles of a run carry weights W_j. Resampling draws K particles with
replacement and resets the weights so that the weighted set still estimates the same thing.
Plain resampling draws in proportion to W_j and resets every weight to 1 (the notes' rule). It
discards particles whose twist is too low now but whose completions are likely: the hidden-mass
problem. Two remedies act here; a third acts on the twist (`twists.ProductTwist`).

    defensive     draw from (1 - e) W~ + e / K, W~ the normalised weights; a survivor a gets
                  weight W~_a / ((1 - e) W~_a + e / K). Every particle keeps at least the chance
                  e / K (Hesterberg 1995, doi:10.1080/00401706.1995.10484303).
    weight_power  draw in proportion to W~^a for a in (0, 1]; a survivor gets weight
                  W~ / (its drawing probability), so part of the weight is kept, not reset.

Both keep the estimate of p(C) unbiased, because the weight after resampling is the exact
ratio of the particle's weight to its drawing probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

SCHEMES = ("multinomial", "systematic")


@dataclass(frozen=True)
class Resampling:
    """When and how to resample.

    threshold     resample a run when its effective sample size falls below threshold * K;
                  0 never resamples, 1 resamples after every step
    defensive     e in [0, 1), the uniform share of the drawing distribution
    weight_power  a in (0, 1], the power of the weights in the drawing distribution
    scheme        "multinomial" (independent draws) or "systematic" (one uniform per run,
                  Kitagawa 1996, doi:10.1080/10618600.1996.10474692)
    """

    threshold: float = 0.5
    defensive: float = 0.0
    weight_power: float = 1.0
    scheme: str = "multinomial"

    def __post_init__(self) -> None:
        if not 0.0 <= self.defensive < 1.0:
            raise ValueError("defensive must lie in [0, 1)")
        if not 0.0 < self.weight_power <= 1.0:
            raise ValueError("weight_power must lie in (0, 1]")
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")


NO_RESAMPLING = Resampling(threshold=0.0)


def log_mean_exp(values: torch.Tensor, dim: int) -> torch.Tensor:
    """log of the mean of exp(values) along `dim`."""
    return torch.logsumexp(values, dim) - math.log(values.shape[dim])


def effective_sample_size(normalised: torch.Tensor) -> torch.Tensor:
    """(runs, K) normalised weights -> (runs,): 1 / sum_j W~_j^2, between 1 and K."""
    return 1.0 / (normalised.double() ** 2).sum(-1)


def drawing_distribution(normalised: torch.Tensor, rule: Resampling) -> torch.Tensor:
    """(runs, K) normalised weights -> (runs, K) probabilities the survivors are drawn from."""
    if rule.weight_power != 1.0:
        powered = normalised.double() ** rule.weight_power
        normalised_power = (powered / powered.sum(-1, keepdim=True)).to(normalised.dtype)
        return (1.0 - rule.defensive) * normalised_power + rule.defensive / normalised.shape[-1]
    return (1.0 - rule.defensive) * normalised + rule.defensive / normalised.shape[-1]


def draw_survivors(
    probabilities: torch.Tensor, rule: Resampling, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """(runs, K) probabilities -> (runs, K) indices of the surviving particles."""
    n_particles = probabilities.shape[-1]
    if rule.scheme == "multinomial":
        return torch.multinomial(probabilities, n_particles, replacement=True, generator=generator)
    device = generator.device if generator is not None else "cpu"
    uniform = torch.rand(
        probabilities.shape[0], 1, generator=generator, dtype=torch.float64, device=device
    )
    grid = torch.arange(n_particles, device=probabilities.device, dtype=torch.float64)[None]
    points = (uniform.to(probabilities.device) + grid) / n_particles
    cumulative = torch.cumsum(probabilities.double(), -1)
    cumulative = cumulative / cumulative[:, -1:]
    return torch.searchsorted(cumulative, points).clamp_max(n_particles - 1)


def survivor_log_weights(
    normalised: torch.Tensor, probabilities: torch.Tensor, survivors: torch.Tensor, rule: Resampling
) -> torch.Tensor:
    """log(W~_a / probability_a) for each survivor a: zero under plain resampling."""
    if rule.defensive > 0 or rule.weight_power != 1.0:
        kept = torch.log(torch.gather(normalised, 1, survivors))
        return (kept - torch.log(torch.gather(probabilities, 1, survivors))).float()
    return torch.zeros_like(normalised, dtype=torch.float32)


def resample(
    log_weights: torch.Tensor, rule: Resampling, generator: Optional[torch.Generator]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One resampling decision for every run.

    log_weights (runs, K). Returns
        fired         (runs,) bool: the runs that resampled
        survivors     (runs, K) indices into each run's particles (the identity where not fired)
        log_weights   (runs, K) weights after resampling (unchanged where not fired)
        log_mean      (runs,) log of the mean weight removed from the fired runs (0 elsewhere);
                      the sampler adds it to log Z_hat
    """
    n_particles = log_weights.shape[-1]
    normalised = torch.nan_to_num(torch.softmax(log_weights, -1), nan=1.0 / n_particles)
    fired = effective_sample_size(normalised) < rule.threshold * n_particles
    identity = torch.arange(n_particles, device=log_weights.device)[None].expand_as(log_weights)
    log_mean = torch.zeros(log_weights.shape[0], dtype=torch.float64, device=log_weights.device)
    if not fired.any():
        return fired, identity, log_weights, log_mean
    log_mean = torch.where(fired, log_mean_exp(log_weights, -1), log_mean)
    probabilities = drawing_distribution(normalised, rule)
    uniform = torch.full_like(normalised, 1.0 / n_particles)
    draw_from = torch.where(fired[:, None], probabilities, uniform).float()
    survivors = draw_survivors(draw_from, rule, generator)
    survivors = torch.where(fired[:, None], survivors, identity)
    after = survivor_log_weights(normalised, probabilities, survivors, rule)
    return fired, survivors, torch.where(fired[:, None], after, log_weights), log_mean
