"""R*_A: relative cross-environment condition (Appendix A.5, main1.tex L2132).

    P_i = 0.4 h_i + 0.4 n_i + 0.2 d_i

for every tree of a species with at least 500 individuals, where ``h`` is the
harmonised census health grade, ``n`` the within-species NDVI percentile and
``d`` the within-species DBH percentile. Grid cells then receive

    R*_A = rank(mean P) / N_s

with multi-species grids averaged by canopy area.

IMPLEMENTATION CHOICE: the manuscript does not define N_s, the rank direction,
tie handling, the percentile scale, how tree-level NDVI is extracted, or how
species below the 500-individual threshold and cells without a qualifying
species are treated. The choices here are documented on the arguments that
carry them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import HEALTH_GRADES, R_A_MIN_SPECIES_N, R_A_WEIGHTS

#: Columns expected in the per-tree frame.
REQUIRED_COLUMNS = ("grid_id", "species", "health_grade", "ndvi", "dbh_cm")


def harmonise_health(grades: pd.Series, city: str) -> pd.Series:
    """Map census health classes to [0, 1] using the field mapping (Table A.19).

    Paris has no census health field, so ``R*_A`` is unavailable there and no
    substitute weights are used (main1.tex L2132).
    """
    mapping: Dict[str, float] = HEALTH_GRADES[city]
    if not mapping:
        raise ValueError(
            f"no census health field is harmonised for {city!r}; R*_A is "
            "unavailable for this city (Appendix A.5)"
        )
    unknown = set(grades.dropna().unique()) - set(mapping)
    if unknown:
        raise KeyError(
            f"health grades {sorted(unknown)} have no harmonised value for "
            f"{city!r}; add them to HEALTH_GRADES only if the manuscript "
            "defines them"
        )
    return grades.map(mapping)


def within_species_percentile(values: pd.Series) -> pd.Series:
    """Percentile rank within a species, in (0, 1].

    IMPLEMENTATION CHOICE: ``pandas.rank(pct=True)`` with average ties, so a
    species of n trees spans 1/n .. 1. The manuscript writes "percentile" and
    "rank" without fixing the scale.
    """
    return values.rank(pct=True, method="average")


def tree_composite(trees: pd.DataFrame, city: str) -> pd.DataFrame:
    """Add ``health``, ``n_pct``, ``d_pct`` and ``P_i`` to a per-tree frame.

    Only species with at least ``R_A_MIN_SPECIES_N`` individuals are scored;
    trees of smaller species are kept but marked ``eligible=False``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in trees.columns]
    if missing:
        raise KeyError(f"per-tree frame is missing {missing}")

    out = trees.copy()
    out["health"] = harmonise_health(out["health_grade"], city)

    species_n = out.groupby("species")["species"].transform("size")
    out["eligible"] = species_n >= R_A_MIN_SPECIES_N

    scored = out[out["eligible"]]
    if scored.empty:
        out["n_pct"] = np.nan
        out["d_pct"] = np.nan
        out["P_i"] = np.nan
        return out

    grouped = scored.groupby("species")
    out.loc[scored.index, "n_pct"] = grouped["ndvi"].transform(within_species_percentile)
    out.loc[scored.index, "d_pct"] = grouped["dbh_cm"].transform(within_species_percentile)
    out["P_i"] = (
        R_A_WEIGHTS["health"] * out["health"]
        + R_A_WEIGHTS["ndvi_pct"] * out["n_pct"]
        + R_A_WEIGHTS["dbh_pct"] * out["d_pct"]
    )
    out.loc[~out["eligible"], "P_i"] = np.nan
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Canopy-area-weighted mean; falls back to the plain mean without areas."""
    mask = values.notna()
    if not mask.any():
        return np.nan
    v, w = values[mask], weights[mask]
    if w.isna().any() or float(w.sum()) <= 0.0:
        return float(v.mean())
    return float(np.average(v, weights=w))


def grid_r_a(
    trees: pd.DataFrame,
    city: str,
    canopy_area_col: Optional[str] = "canopy_area_m2",
) -> pd.Series:
    """Compute ``R*_A`` per grid cell.

    Multi-species grids are averaged by canopy area (main1.tex L2132). The
    grid-level rank normalization uses ascending average ranks over the cells
    that receive a value, divided by the number of such cells; cells with no
    eligible tree return NaN.

    IMPLEMENTATION CHOICE: ``N_s`` is taken to be the number of scored cells
    and the rank is ascending (higher composite condition ranks higher). The
    manuscript leaves both implicit.
    """
    scored = tree_composite(trees, city)
    area_col = canopy_area_col if canopy_area_col in scored.columns else None

    def _weights(frame: pd.DataFrame) -> pd.Series:
        if area_col is None:
            return pd.Series(np.nan, index=frame.index)
        return frame[area_col]

    cell_mean_values = {}
    for grid_id, frame in scored.groupby("grid_id", sort=True):
        if frame["species"].nunique() == 1:
            cell_mean_values[grid_id] = _weighted_mean(frame["P_i"], _weights(frame))
            continue
        # Multi-species grids: average the species means by canopy area.
        species_means, species_areas = [], []
        for _, group in frame.groupby("species"):
            species_means.append(_weighted_mean(group["P_i"], _weights(group)))
            species_areas.append(float(group[area_col].sum()) if area_col else np.nan)
        cell_mean_values[grid_id] = _weighted_mean(
            pd.Series(species_means, dtype=float), pd.Series(species_areas, dtype=float)
        )
    cell_mean = pd.Series(cell_mean_values, dtype=float)
    cell_mean.index.name = "grid_id"

    n_s = int(cell_mean.notna().sum())
    if n_s == 0:
        return cell_mean.astype(float)
    ranks = cell_mean.rank(method="average", ascending=True, na_option="keep")
    return (ranks / n_s).astype(float)
