"""Bounds on the error of the returned sample at lengths where the exact law is out of reach.

The quantity of interest is KL(q_K || pi): the law of the sequence returned by K particles
against the target. From the weights of n independent runs of K particles (no resampling):

    upper_bound       log(mean Z_hat) - mean(log Z_hat). In expectation -E log(Z_hat / Z), the
                      KL between sampler and target on the space of all K particles and the
                      index returned, which bounds KL(q_K || pi) from above (data processing).
                      Biased low when the mean of Z_hat is itself underestimated.
    leading_term      Var_pi(w) / (2 K^2) with w the normalised weight: the first term of the
                      expansion of KL(q_K || pi) for large K, estimated from the weights.
    marginal_lower_bound
                      the largest KL between a marginal of the returned sequences and the same
                      marginal of the target (single positions or pairs): a lower bound by data
                      processing; needs reference marginals of the target.

Checked against the exact law at 8 positions: the leading term is within about 1.05 of the
exact KL at K = 4 and 1.01 at K = 16; the upper bound is loose by 3.6 and 13 at those K.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .exact import kl


def log_mean_exp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    top = values.max(axis=axis, keepdims=True)
    return np.squeeze(top, axis) + np.log(np.exp(values - top).mean(axis=axis))


def upper_bound(log_Z_hat: np.ndarray) -> float:
    """(runs,) log Z_hat -> log(mean Z_hat) - mean(log Z_hat), in nats."""
    return float(log_mean_exp(log_Z_hat) - log_Z_hat.mean())


def leading_term(log_weights: np.ndarray) -> float:
    """(runs, K) final log-weights without resampling -> the plug-in estimate of
    Var_pi(w) / (2 K^2): (m3 - m2^2) / (2 K^2) with m_k the mean of w^k over all weights,
    w scaled to mean one."""
    n_particles = log_weights.shape[1]
    weights = np.exp(log_weights - log_weights.max()).ravel()
    weights = weights / weights.mean()
    second = (weights**2).mean()
    third = (weights**3).mean()
    return float((third - second * second) / (2 * n_particles * n_particles))


def marginal_lower_bound(
    returned: np.ndarray,
    target_sequences: np.ndarray,
    target_law: np.ndarray,
    n_tokens: int,
    *,
    order: int = 1,
    positions: Optional[Iterable[Tuple[int, ...]]] = None,
) -> float:
    """The largest KL between marginals of the returned sequences (n, L) and of the target, over
    single positions (order 1) or pairs (order 2). The target is given as weighted sequences."""
    length = returned.shape[1]
    groups = (
        list(itertools.combinations(range(length), order)) if positions is None else list(positions)
    )
    best = 0.0
    for group in groups:
        code_returned = np.zeros(len(returned), dtype=np.int64)
        code_target = np.zeros(len(target_sequences), dtype=np.int64)
        for position in group:
            code_returned = code_returned * n_tokens + returned[:, position]
            code_target = code_target * n_tokens + target_sequences[:, position]
        size = n_tokens ** len(group)
        observed = np.bincount(code_returned, minlength=size) / len(returned)
        expected = np.bincount(code_target, weights=target_law, minlength=size)
        best = max(best, kl(observed, np.where(expected > 0, expected, 1e-300)))
    return best


def bracket(
    log_Z_hat: np.ndarray, log_weights: np.ndarray, lower: Optional[float] = None
) -> Dict[str, float]:
    """What to report: [lower, upper] with the leading term as the central value (use it for
    K >= 4; at K = 1 the upper bound is the KL itself)."""
    return {
        "lower": float("nan") if lower is None else lower,
        "central": leading_term(log_weights),
        "upper": upper_bound(log_Z_hat),
    }
