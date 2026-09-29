"""Grid construction and point-to-cell assignment.

Section 3.1: each study domain is covered by a regular grid
of Delta x Delta cells (Delta = 100 m by default, ablated over
{50, 75, 100, 150, 200} m). Cell counts are computed from the bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from ..config import DELTA_M, STUDY_AREAS


@dataclass(frozen=True)
class Grid:
    """A regular grid over a projected (metre) bounding box."""
    x_min: float
    y_min: float
    n_x: int
    n_y: int
    delta_m: float = DELTA_M

    @property
    def n_cells(self) -> int:
        return self.n_x * self.n_y

    def cell_id(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        return np.asarray([f"{i:05d}_{j:05d}" for i, j in zip(ix, iy)])

    def index_of(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ix = np.floor((x - self.x_min) / self.delta_m).astype(int)
        iy = np.floor((y - self.y_min) / self.delta_m).astype(int)
        inside = (ix >= 0) & (ix < self.n_x) & (iy >= 0) & (iy < self.n_y)
        return np.where(inside, ix, -1), np.where(inside, iy, -1)

    def centres(self) -> Iterator[Tuple[str, float, float]]:
        for i in range(self.n_x):
            for j in range(self.n_y):
                yield (
                    f"{i:05d}_{j:05d}",
                    self.x_min + (i + 0.5) * self.delta_m,
                    self.y_min + (j + 0.5) * self.delta_m,
                )


def grid_from_extent(extent_km: Tuple[float, float], delta_m: float = DELTA_M,
                     origin: Tuple[float, float] = (0.0, 0.0)) -> Grid:
    """Grid covering ``extent_km`` at ``delta_m`` resolution."""
    width_m = extent_km[0] * 1000.0
    height_m = extent_km[1] * 1000.0
    return Grid(
        x_min=origin[0],
        y_min=origin[1],
        n_x=int(np.ceil(width_m / delta_m)),
        n_y=int(np.ceil(height_m / delta_m)),
        delta_m=delta_m,
    )


def city_grid(city: str, delta_m: float = DELTA_M) -> Grid:
    """Grid for one of the three study areas, using Table 3 extents."""
    if city not in STUDY_AREAS:
        raise KeyError(f"unknown city {city!r}; expected one of {sorted(STUDY_AREAS)}")
    return grid_from_extent(STUDY_AREAS[city]["extent_km"], delta_m=delta_m)


def assign_points(grid: Grid, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Assign projected points to cells; points outside the grid are dropped.

    Returns a frame with ``grid_id``, ``ix``, ``iy`` indexed like the input.
    """
    ix, iy = grid.index_of(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    inside = (ix >= 0) & (iy >= 0)
    return pd.DataFrame(
        {"grid_id": grid.cell_id(ix[inside], iy[inside]), "ix": ix[inside], "iy": iy[inside]},
        index=np.flatnonzero(inside),
    )


def cell_count_summary(delta_m: float = DELTA_M) -> Dict[str, int]:
    """Total cells per city at ``delta_m`` resolution."""
    return {city: city_grid(city, delta_m).n_cells for city in STUDY_AREAS}
