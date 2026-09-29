"""Per-city z-score standardisation with saved parameters.

Appendix A.7: per-city z-scores with parameters saved for inference;
categorical variables use learnable embeddings. Parameters are fitted on the
training partition by default so that validation and test statistics do not
enter the transform; ``fit_on="all"`` fits on the whole city.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ZScoreParams:
    """Means and standard deviations, one row per standardised column."""
    mean: pd.Series
    std: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"mean": self.mean, "std": self.std})

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ZScoreParams":
        return cls(mean=frame["mean"], std=frame["std"])


def fit(frame: pd.DataFrame, columns: Sequence[str]) -> ZScoreParams:
    """Fit per-column mean and standard deviation (population, ddof=0).

    Columns with zero spread keep a standard deviation of 1 so the transform
    stays finite; the caller can spot them from ``std == 1`` combined with a
    constant column.
    """
    values = frame[list(columns)].astype(float)
    std = values.std(ddof=0).replace(0.0, 1.0)
    return ZScoreParams(mean=values.mean(), std=std)


def transform(frame: pd.DataFrame, params: ZScoreParams,
              columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Apply saved z-score parameters; returns a copy."""
    cols = list(columns) if columns is not None else list(params.mean.index)
    out = frame.copy()
    out[cols] = (frame[cols].astype(float) - params.mean[cols]) / params.std[cols]
    return out


def fit_transform(frame: pd.DataFrame, columns: Sequence[str],
                  split: Optional[pd.Series] = None,
                  fit_on: str = "train") -> tuple:
    """Fit on one partition and transform the frame.

    ``split`` is a per-row partition label ("train", "validation", "test").
    With ``fit_on="all"`` (or no split supplied) the parameters come from
    every row.
    """
    if split is None or fit_on == "all":
        params = fit(frame, columns)
    else:
        mask = split == fit_on
        if not mask.any():
            raise ValueError(f"no rows in the {fit_on!r} partition to fit on")
        params = fit(frame[mask], columns)
    return transform(frame, params, columns), params
