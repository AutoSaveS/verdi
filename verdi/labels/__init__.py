"""Construction of the three proxy labels and their fusion strategies.

The formulas follow Section 3.1 and Appendix A.3. Configurable parameters and
their defaults are listed in ``verdi/config.py``.
"""

from .fusion import (
    LABELS,
    STRATEGIES,
    combine_availability_weighted,
    combine_equal,
    combine_fixed_weights,
    combine_learnable_weights,
)
from .r_a import grid_r_a, harmonise_health, tree_composite
from .r_b import NdviPair, daily_tmax, detect_heat_events, grid_r_b, retention, valid_pair
from .r_c import grid_r_c, species_baseline

__all__ = [
    "LABELS",
    "STRATEGIES",
    "NdviPair",
    "combine_availability_weighted",
    "combine_equal",
    "combine_fixed_weights",
    "combine_learnable_weights",
    "daily_tmax",
    "detect_heat_events",
    "grid_r_a",
    "grid_r_b",
    "grid_r_c",
    "harmonise_health",
    "retention",
    "species_baseline",
    "tree_composite",
    "valid_pair",
]
