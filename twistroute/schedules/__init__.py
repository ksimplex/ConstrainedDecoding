"""Unmasking schedules (notes, sections 1 and 4).

base        the interface; RandomOrder; helpers shared by the schedules
sequential  LeftToRight, Confidence, Sum (one position per step by default)
parallel    Blocks, BudgetedCoReveal
baselines   Dilated (DUS), for reference
dependence  the estimate of dependence between positions that Sum and BudgetedCoReveal use
"""

from .base import FromPolicy, RandomOrder, Schedule, as_schedule, constrained_marginals, forced, pad
from .baselines import Dilated
from .dependence import DistanceDependence, fit_distance_dependence
from .parallel import Blocks, BudgetedCoReveal
from .sequential import Confidence, FixedOrder, LeftToRight, Ranked, Sum

__all__ = [
    "Schedule",
    "RandomOrder",
    "FromPolicy",
    "as_schedule",
    "constrained_marginals",
    "forced",
    "pad",
    "LeftToRight",
    "FixedOrder",
    "Confidence",
    "Sum",
    "Ranked",
    "Blocks",
    "BudgetedCoReveal",
    "Dilated",
    "DistanceDependence",
    "fit_distance_dependence",
]
