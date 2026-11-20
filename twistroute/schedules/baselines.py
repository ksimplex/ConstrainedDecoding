"""Schedules from other work, kept for reference."""

from __future__ import annotations

from typing import Optional

import torch

from ..state import State
from .base import Schedule, Unmasking, pad


class Dilated(Schedule):
    """A fixed coarse-to-fine schedule that does not look at the model: the dilated unmasking
    of DUS (Luxembourg, Permuter and Nachmani, arXiv:2506.19037, eq. 8 and 9) with the whole
    sequence as one block. With base a, step t unmasks the masked positions whose index is a
    multiple of floor(L / a^(t + 1)); the sequence is complete after ceil(log_a L) steps."""

    def __init__(self, base: int = 2) -> None:
        if base < 2:
            raise ValueError("base must be >= 2")
        self.base = int(base)

    def unmask(
        self, state: State, step: int, n_steps: int, generator: Optional[torch.Generator] = None
    ) -> Unmasking:
        masked = state.masked
        length = masked.shape[1]
        stride = max(1, length // self.base ** (step + 1))
        if step >= n_steps - 1 or stride == 1:
            return pad(masked)
        return pad(masked & (torch.arange(length, device=masked.device)[None] % stride == 0))
