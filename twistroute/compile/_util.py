"""Shared numerics for the CPU reference compiler."""
import numpy as np

NEG = -np.inf


def lse(v):
    """log-sum-exp of a 1-D array; returns -inf for an empty or all -inf input."""
    v = np.asarray(v, float)
    if v.size == 0:
        return NEG
    m = v.max()
    return m + np.log(np.exp(v - m).sum()) if np.isfinite(m) else NEG


def categorical(rng, logits):
    """Draw an index with probability proportional to exp(logits)."""
    logits = np.asarray(logits, float)
    m = logits.max()
    if not np.isfinite(m):
        raise ValueError("categorical: no finite mass")
    p = np.exp(logits - m)
    return int(rng.choice(len(p), p=p / p.sum()))


class CompilerRefuse(Exception):
    """Raised when a construction exceeds its size budget.

    This is the honest tractability boundary: a constraint family whose exact counting is #P-hard
    (e.g. clause conjunctions) has no small circuit, so the compiler refuses instead of silently
    building an exponential object.
    """
