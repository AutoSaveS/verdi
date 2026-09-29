"""R*_C: relative deviation from a species baseline.

Eq. A.2:

    R*_C = sigmoid( (NDVI_grid - NDVI~(s)) / sigma_NDVI(s) )

where the baseline and its spread are species-level. This is the only label
available in all three cities and it is the primary target.

The baseline statistic ("median" or "mean") is an argument, and
``apply_minmax`` adds an optional min-max rescaling after the sigmoid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..config import R_C_APPLY_MINMAX, R_C_BASELINE_STATISTIC


def species_baseline(
    ndvi: pd.Series,
    species: pd.Series,
    statistic: str = R_C_BASELINE_STATISTIC,
) -> pd.DataFrame:
    """Per-species baseline and spread of NDVI.

    Returns a frame indexed by species with columns ``baseline``,
    ``spread`` and ``n``. ``spread`` is the population standard deviation,
    falling back to NaN for a species with a single observation.
    """
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    frame = pd.DataFrame({"ndvi": ndvi, "species": species}).dropna()
    grouped = frame.groupby("species")["ndvi"]
    centre = grouped.median() if statistic == "median" else grouped.mean()
    out = pd.DataFrame({"baseline": centre, "spread": grouped.std(ddof=0), "n": grouped.size()})
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def grid_r_c(
    ndvi: pd.Series,
    species: pd.Series,
    statistic: str = R_C_BASELINE_STATISTIC,
    apply_minmax: bool = R_C_APPLY_MINMAX,
    baselines: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Per-cell ``R*_C`` from a cell's NDVI and its dominant species.

    ``ndvi`` and ``species`` are indexed by grid cell. ``baselines`` can be
    supplied to reuse the fit from another split; by default it is fitted on
    the values passed in.

    Cells whose species has a single individual, or a zero spread, return NaN.
    """
    frame = pd.DataFrame({"ndvi": ndvi, "species": species}).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=ndvi.index, dtype=float)

    stats = species_baseline(frame["ndvi"], frame["species"], statistic) \
        if baselines is None else baselines

    joined = frame.join(stats, on="species")
    spread = joined["spread"].replace(0.0, np.nan)
    z = (joined["ndvi"] - joined["baseline"]) / spread
    values = sigmoid(z.to_numpy(dtype=float))
    out = pd.Series(values, index=joined.index, dtype=float)

    if apply_minmax:
        lo, hi = np.nanmin(values), np.nanmax(values)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out = (out - lo) / (hi - lo)
    return out.reindex(ndvi.index)
