"""From a pattern over characters to a constraint over a model's tokens.

A tokenizer has ~10^5 tokens; a constraint distinguishes far fewer. `from_regex` groups the tokens
into classes that the pattern treats alike (150 to 200 on the JSON and function-call benchmarks of
the notes) and returns the automaton over those classes.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..compile.tokens import lift_pattern, token_codes
from .automaton import automaton
from .base import Constraint


def from_regex(
    pattern: str,
    token_strings: Sequence[str],
    *,
    end_ids: Sequence[int] = (),
    codes=None,
    device: str = "cpu",
) -> Tuple[Constraint, np.ndarray, dict]:
    """pattern: a regular expression over characters. token_strings[t]: the text of token t.
    end_ids: tokens allowed (only) after the pattern has matched, e.g. end-of-text padding.

    Returns (constraint over token classes, token_class array (V,), info). Pass `codes` from
    `token_codes(token_strings)` when compiling many patterns for one tokenizer.
    """
    token_class, dfa, info = lift_pattern(
        pattern, token_strings, end_ids=tuple(end_ids), codes=codes
    )
    return automaton(dfa, device=device, name="pattern over tokens"), token_class, info


__all__ = ["from_regex", "token_codes"]
