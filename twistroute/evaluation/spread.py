"""Spread of the estimate Z_hat of P(x_0 in C) across runs (notes, Figure 2).

A sampler whose weights are even gives nearly the same Z_hat in every run. The standard
deviation of log Z_hat needs no reference; the root mean square of log(Z_hat / P(C)) needs the
exact P(C) (oracle models) and also sees bias. Both in nats, lower is better.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def sd_log_Z_hat(log_Z_hat: np.ndarray) -> float:
    """(runs,) -> standard deviation of log Z_hat over the runs with a valid output."""
    finite = log_Z_hat[np.isfinite(log_Z_hat)]
    return float(finite.std(ddof=1))


def rms_log_ratio(log_Z_hat: np.ndarray, log_p_C: float) -> Tuple[float, float]:
    """(runs,) -> root mean square of log(Z_hat / P(C)) and its jackknife standard error."""
    square = (log_Z_hat[np.isfinite(log_Z_hat)] - log_p_C) ** 2
    n = len(square)
    value = float(np.sqrt(square.mean()))
    leave_one_out = np.sqrt((square.sum() - square) / (n - 1))
    error = float(np.sqrt((n - 1) / n * ((leave_one_out - leave_one_out.mean()) ** 2).sum()))
    return value, error


def mean_ratio(log_Z_hat: np.ndarray, log_p_C: float) -> Tuple[float, float]:
    """(runs,) -> mean of Z_hat / P(C) and its standard error. One for an unbiased sampler;
    below one in a finite sample when rare runs carry much of the mass (hidden mass)."""
    ratio = np.exp(log_Z_hat - log_p_C)
    return float(ratio.mean()), float(ratio.std(ddof=1) / np.sqrt(len(ratio)))


def dead_runs(log_Z_hat: np.ndarray) -> Optional[int]:
    """Number of runs in which every particle had weight zero."""
    return int((~np.isfinite(log_Z_hat)).sum())
