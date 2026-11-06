"""
Reveal schedules: which masked positions a masked-diffusion reverse step uncovers.

Two processes, and they are genuinely different. Our E1-E3 runs and TWISTER (Thimmaiah et al.,
arXiv:2609.35609, Sec. 4.3) uncover exactly N / T positions per step, chosen uniformly among those
still masked. The MDLM reverse step (Sahoo et al., arXiv:2406.07524, Eq. 6) instead uncovers each
masked position independently with probability 1 / (T - t) under the log-linear schedule, which
leaves a random number of reveals per step. With the same model and constraint, h0 differs between
them by 4-10 standard errors (docs/status.md Sec. 6c), so the schedule must match the process the
result is meant to describe. mRNAutilus uses the independent form.

The independent schedule is ragged across a batch, so `reveal_indices` pads to the row maximum and
returns a validity mask; callers must ignore invalid columns rather than treat them as reveals.
Counting a padded column as revealed biases Zhat by roughly 30% (regression test in test_smc.py).
"""
import torch

from ..torch_routes.routes import onehot_log

NEG_INF = float("-inf")
SCHEDULES = ("fixed", "independent")

# ----------------------------------------------------------------------------------------------- primitives
def reveal_sets(masked, n_t, gen=None):
    """Uniform random subset of size n_t among masked positions, per row. Returns (B, n_t) indices."""
    u = torch.rand(masked.shape, device=masked.device, generator=gen)
    u = torch.where(masked, u, torch.full_like(u, 2.0))
    return torch.topk(-u, n_t, dim=-1).indices


SCHEDULES = ("fixed", "independent")


def reveal_indices(masked, t, T, schedule, gen=None):
    """Positions revealed at step t, as (indices (B, k), valid (B, k) or None).

    "fixed"        exactly N / T positions per row, uniform among the masked ones (TWISTER / our E1-E3 runs).
                   Returns valid = None: every column is a real reveal.
    "independent"  each still-masked position reveals on its own with probability 1 / (T - t), the MDLM
                   reverse step for the log-linear schedule (the step from time (T - t)/T to (T - t - 1)/T
                   unmasks a masked position with probability dt / t). The count per row is random, and at
                   t = T - 1 the probability is 1, so every row finishes fully revealed. Columns are padded to
                   the largest count in the batch; `valid` marks the real ones.

    The scheduler cancels in the potential (notes, Lemma "step-exact proposal"), so any scheduler gives an
    exact method provided the proposal and the native kernel use the same one. The choice fixes WHICH native
    process the result is exact for.
    """
    if schedule == "fixed":
        return reveal_sets(masked, masked.shape[-1] // T, gen), None
    if schedule != "independent":
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    p = 1.0 / (T - t)
    hit = masked & (torch.rand(masked.shape, device=masked.device, generator=gen) < p)
    cnt = hit.sum(-1)
    k = int(cnt.max().clamp_min(1))
    order = torch.argsort(hit.to(torch.int8), dim=-1, descending=True, stable=True)
    r = order[:, :k]
    valid = torch.arange(k, device=masked.device)[None] < cnt[:, None]
    return r, valid


def clamp_revealed(lw, x, ycls, mask_id):
    """Replace class log-weights at revealed positions of x by the one-hot of their class."""
    return torch.where((x != mask_id)[..., None], onehot_log(ycls, lw.shape[-1]), lw)
