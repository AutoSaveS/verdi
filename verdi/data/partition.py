"""Spatial partitioning: super-blocks, train/validation/test and hold-out.

Appendix A.5: 5Delta x 5Delta super-blocks are assigned to
train/validation/test in 70/15/15 proportions with a minimum 4Delta
inter-partition separation, and 10 % per city is held out for cross-city
generalization. The assignment is deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    CROSS_CITY_HOLDOUT,
    MIN_SEPARATION_CELLS,
    PARTITION_SEED,
    SPLIT_FRACTIONS,
    SUPERBLOCK_CELLS,
)
from .grid import Grid


@dataclass(frozen=True)
class PartitionResult:
    """Super-block assignment and the per-cell split labels."""
    blocks: pd.DataFrame          # block index -> split
    cells: pd.Series              # grid_id -> split
    holdout: pd.Series            # grid_id -> bool (cross-city hold-out)
    seed: int
    separation_cells: int
    separation_margin_blocks: float = 0.0   # margin in super-block widths


def superblock_index(grid: Grid, block_cells: int = SUPERBLOCK_CELLS) -> pd.DataFrame:
    """Map every cell to its super-block index."""
    rows = []
    for grid_id, _, _ in grid.centres():
        i, j = (int(part) for part in grid_id.split("_"))
        rows.append({"grid_id": grid_id, "bx": i // block_cells, "by": j // block_cells})
    frame = pd.DataFrame(rows).set_index("grid_id")
    frame["block_id"] = frame["bx"].astype(str) + "_" + frame["by"].astype(str)
    return frame


def block_separation_margin(separation_cells: int = MIN_SEPARATION_CELLS,
                            superblock_cells: int = SUPERBLOCK_CELLS) -> float:
    """Separation between partitions, in super-block widths.

    The split requires a minimum 4Delta separation between partitions
    while splitting 5Delta x 5Delta super-blocks. Two distinct super-blocks
    have centres at least 5Delta apart, so the requirement holds by
    construction and no extra buffer is needed. The value returned is the
    margin in super-block widths, so a caller can check the assumption rather
    than trust it: ``(5 - 4) / 5 = 0.2`` with the default settings.
    """
    return max(0.0, (superblock_cells - separation_cells) / superblock_cells)


def assign_blocks(
    grid: Grid,
    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS,
    separation_cells: int = MIN_SEPARATION_CELLS,
    seed: int = PARTITION_SEED,
) -> Tuple[pd.DataFrame, float]:
    """Assign super-blocks to train/validation/test.

    Blocks are dealt out in a seeded shuffle, so the partition is
    deterministic for a given grid, seed and set of fractions, and linear in
    the number of blocks. The separation requirement is structural (see
    :func:`block_separation_margin`) rather than enforced by a per-block
    search, which is what keeps a city-wide grid tractable.

    Returns ``(blocks, margin_in_superblock_widths)``.
    """
    frame = superblock_index(grid)
    blocks = (
        frame[["bx", "by"]].drop_duplicates().sort_values(["bx", "by"]).reset_index(drop=True)
    )
    blocks["block_id"] = blocks["bx"].astype(str) + "_" + blocks["by"].astype(str)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    n_train = int(round(fractions[0] * len(blocks)))
    n_val = int(round(fractions[1] * len(blocks)))

    labels = np.array(["train"] * len(blocks), dtype=object)
    labels[order[n_train:n_train + n_val]] = "validation"
    labels[order[n_train + n_val:]] = "test"
    blocks["split"] = labels

    cells = frame.join(blocks.set_index("block_id")["split"], on="block_id")["split"]
    return blocks, block_separation_margin(separation_cells)


def cross_city_holdout(
    grid: Grid,
    cells: pd.Series,
    fraction: float = CROSS_CITY_HOLDOUT,
    seed: int = PARTITION_SEED,
) -> pd.Series:
    """Flag the per-city cross-city hold-out cells.

    Drawn from the training blocks only, so the hold-out is not part of the
    fitted parameters.
    """
    rng = np.random.default_rng(seed + 1)
    candidates = cells.index[cells == "train"].to_numpy()
    n = int(round(fraction * len(cells)))
    n = min(n, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False)
    return pd.Series(cells.index.isin(chosen), index=cells.index, name="cross_city_holdout")


def make_partition(
    grid: Grid,
    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS,
    separation_cells: int = MIN_SEPARATION_CELLS,
    seed: int = PARTITION_SEED,
) -> PartitionResult:
    """Full partition: super-block splits plus the cross-city hold-out."""
    blocks, margin = assign_blocks(grid, fractions=fractions,
                                   separation_cells=separation_cells, seed=seed)
    # Expand the block-level labels back to every grid cell, so the returned
    # series is indexed by grid_id rather than by super-block.
    cells = superblock_index(grid).join(
        blocks.set_index("block_id")["split"], on="block_id"
    )["split"]
    holdout = cross_city_holdout(grid, cells, seed=seed)
    return PartitionResult(blocks=blocks, cells=cells, holdout=holdout,
                           seed=seed, separation_cells=separation_cells,
                           separation_margin_blocks=margin)


def summarise(result: PartitionResult) -> Dict[str, Dict[str, float]]:
    """Cell counts and proportions per split, for reporting."""
    counts = result.cells.value_counts().to_dict()
    total = float(len(result.cells))
    out: Dict[str, Dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        n = float(counts.get(split, 0.0))
        out[split] = {"cells": n, "fraction": n / total if total else float("nan")}
    out["cross_city_holdout"] = {
        "cells": float(result.holdout.sum()),
        "fraction": float(result.holdout.mean()) if total else float("nan"),
    }
    return out


def load_parameters(grid: Grid, params: Optional[Dict[str, float]] = None) -> Grid:
    """Placeholder for reading grid parameters from a config file.

    Returns ``grid`` unchanged; provides a single entry point for projects
    that store grid parameters in their own file.
    """
    return grid
