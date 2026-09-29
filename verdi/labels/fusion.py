"""Combining the three proxy labels.

Two sets of strategies are implemented:

* Table A.23 - fusion strategies: fixed
  weights with a grid search (S1), three prediction heads (S2), learnable
  weights conditioned on city and species (S3), and availability-based
  weights from a data-quality score (S4).
* Appendix A.4 - label combinations:
  R*_C only, equal weights over available labels, availability weights with
  indicator weights, and a multi-head variant sharing the generator.

``R*_C`` alone is the primary configuration.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

LABELS = ("R_A", "R_B", "R_C")


def available_matrix(labels: pd.DataFrame, columns: Sequence[str] = LABELS) -> np.ndarray:
    """Boolean availability matrix, one column per label."""
    return labels[list(columns)].notna().to_numpy()


def combine_equal(labels: pd.DataFrame, columns: Sequence[str] = LABELS) -> pd.Series:
    """Equal-weight mean over the labels that are available for the cell."""
    values = labels[list(columns)]
    return values.mean(axis=1, skipna=True)


def combine_availability_weighted(
    labels: pd.DataFrame,
    quality: Optional[pd.DataFrame] = None,
    columns: Sequence[str] = LABELS,
) -> pd.Series:
    """Availability weights ``w_k = 1[available]`` (Appendix A.4), optionally
    multiplied by a per-label data-quality score ``q_k`` in [0, 1].

    ``w_k`` proportional to ``q_k`` is strategy S4 of Table A.23.
    """
    values = labels[list(columns)]
    weights = values.notna().astype(float)
    if quality is not None:
        weights = weights * quality[list(columns)].fillna(0.0)
    numerator = (values.fillna(0.0) * weights).sum(axis=1)
    denominator = weights.sum(axis=1)
    return (numerator / denominator.replace(0.0, np.nan)).astype(float)


def combine_fixed_weights(
    labels: pd.DataFrame,
    weights: Dict[str, float],
    columns: Sequence[str] = LABELS,
) -> pd.Series:
    """S1: R* = sum_k w_k R*_k, with the weights supplied by the caller.

    ``weights`` is a mapping over label names, for example
    ``{"R_A": 0.3, "R_B": 0.3, "R_C": 0.4}``. Weight selection by grid search
    is left to the caller.
    """
    missing = set(columns) - set(weights)
    if missing:
        raise KeyError(f"weights are missing for {sorted(missing)}")
    values = labels[list(columns)]
    weights_series = pd.Series({k: float(weights[k]) for k in columns})
    available = values.notna()
    weighted = (values.fillna(0.0) * weights_series).sum(axis=1)
    total = (available * weights_series).sum(axis=1)
    return (weighted / total.replace(0.0, np.nan)).astype(float)


def combine_learnable_weights(
    labels: pd.DataFrame,
    alpha: pd.DataFrame,
    columns: Sequence[str] = LABELS,
) -> pd.Series:
    """S3: softmax over per-(city, species) logits ``alpha``.

    ``alpha`` is indexed like ``labels`` with one column per label; the
    softmax is taken across the label axis and renormalised over the
    available labels.
    """
    logits = alpha[list(columns)].to_numpy(dtype=float)
    logits = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    weights = exp / exp.sum(axis=1, keepdims=True)
    weights = pd.DataFrame(weights, index=labels.index, columns=list(columns))
    available = labels[list(columns)].notna()
    weights = weights * available
    weights = weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0)
    values = labels[list(columns)].fillna(0.0)
    return (values * weights).sum(axis=1, skipna=False).astype(float)


#: Names of the Appendix A.4 strategies, for scripts and documentation.
STRATEGIES = ("R_C_only", "equal_available", "availability_weighted", "multi_head")
