"""R*_B: short-term NDVI retention around qualifying heat events.

Appendix A.3. A qualifying heat event is at least three
consecutive days with daily maximum air temperature above the city-specific
95th percentile of the ERA5-Land record for 2018-2023. Retention is computed
from cloud-filtered Sentinel-2 pre/post pairs within 14 days:

    R*_B = 1 - |NDVI_post - NDVI_pre| / max(NDVI_pre, 0.2),  clipped to [0, 1]

Cells without a qualifying event or without a valid image pair are NaN.

The temperature aggregation, percentile base period, window anchor, cloud
filter and multi-event combination are arguments with stated defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import (
    R_B_CLIP,
    R_B_CLOUD_MAX_PCT,
    R_B_EVENT_MIN_DAYS,
    R_B_EVENT_PERCENTILE,
    R_B_NDVI_FLOOR,
    R_B_NDVI_WINDOW_DAYS,
)


def daily_tmax(temperature: pd.Series, hours_per_day: int = 24) -> pd.Series:
    """Daily maximum of an hourly (or sub-daily) temperature series.

    Intended for ERA5-Land ``2m_temperature`` (t2m, K); the aggregation is a
    calendar-day maximum and assumes one entry per hour.
    """
    if not isinstance(temperature.index, pd.DatetimeIndex):
        raise TypeError("temperature series must have a DatetimeIndex")
    return temperature.resample("1D").max().dropna()


def detect_heat_events(
    tmax: pd.Series,
    percentile: float = R_B_EVENT_PERCENTILE,
    min_days: int = R_B_EVENT_MIN_DAYS,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return runs of >= ``min_days`` consecutive days above ``percentile``.

    The percentile is taken over the series passed in: pass a summer-window
    series for a summer percentile, or the full year for an annual one.
    """
    if tmax.empty:
        return []
    threshold = float(np.nanpercentile(tmax.to_numpy(dtype=float), percentile))
    above = (tmax > threshold).to_numpy()
    days = tmax.index

    events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start: Optional[int] = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_days:
                events.append((days[start], days[i - 1]))
            start = None
    if start is not None and len(above) - start >= min_days:
        events.append((days[start], days[-1]))
    return events


def retention(ndvi_pre: float, ndvi_post: float, floor: float = R_B_NDVI_FLOOR) -> float:
    """Eq. A.1, clipped to [0, 1].

    The absolute value in the numerator means a greening response also
    lowers the value.
    """
    if not np.isfinite(ndvi_pre) or not np.isfinite(ndvi_post):
        return float("nan")
    raw = 1.0 - abs(ndvi_post - ndvi_pre) / max(ndvi_pre, floor)
    return float(np.clip(raw, *R_B_CLIP))


@dataclass(frozen=True)
class NdviPair:
    """A pre/post Sentinel-2 pair associated with a heat event."""
    grid_id: str
    event_start: pd.Timestamp
    days_before: int
    days_after: int
    ndvi_pre: float
    ndvi_post: float
    cloud_pct: float = 0.0


def valid_pair(pair: NdviPair, window_days: int = R_B_NDVI_WINDOW_DAYS,
               cloud_max_pct: float = R_B_CLOUD_MAX_PCT) -> bool:
    """A pair is usable when it is within the window and cloud-free enough."""
    return (
        pair.cloud_pct < cloud_max_pct
        and 0 <= pair.days_before <= window_days
        and 0 <= pair.days_after <= window_days
    )


def grid_r_b(
    pairs: Sequence[NdviPair],
    grid_ids: Optional[Sequence[str]] = None,
    case: str = "mean",
) -> pd.Series:
    """Per-cell ``R*_B`` from event/pair records.

    When a cell has several qualifying events, ``case="mean"`` averages the
    per-event retentions, ``"last"`` keeps the most recent event. Cells with
    no usable pair are NaN.
    """
    rows = []
    for pair in pairs:
        if not valid_pair(pair):
            continue
        rows.append(
            {
                "grid_id": pair.grid_id,
                "event_start": pair.event_start,
                "R_B": retention(pair.ndvi_pre, pair.ndvi_post),
            }
        )
    frame = pd.DataFrame(rows, columns=["grid_id", "event_start", "R_B"])
    if grid_ids is not None:
        index = pd.Index(grid_ids, name="grid_id")
    else:
        index = pd.Index(sorted(set(pairs[i].grid_id for i in range(len(pairs)))), name="grid_id")
    if frame.empty:
        return pd.Series(np.nan, index=index, dtype=float)

    if case == "mean":
        out = frame.groupby("grid_id")["R_B"].mean()
    elif case == "last":
        out = (
            frame.sort_values("event_start")
            .groupby("grid_id")["R_B"]
            .last()
        )
    else:
        raise ValueError("case must be 'mean' or 'last'")
    return out.reindex(index).astype(float)
