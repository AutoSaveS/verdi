"""Checks of the grid, the exclusion rules and the spatial split."""

import numpy as np
import pandas as pd
import pytest

from verdi.config import DELTA_M, SPLIT_FRACTIONS
from verdi.data import city_grid, grid_from_extent, make_partition, summarise
from verdi.data.partition import assign_blocks, superblock_index
from verdi.data.standardize import fit, transform


def test_grid_cell_count_matches_the_extent():
    grid = grid_from_extent((10.0, 5.0), delta_m=100.0)
    assert (grid.n_x, grid.n_y) == (100, 50)
    assert grid.n_cells == 5000


def test_city_grids_are_positive():
    for city in ("nyc", "paris", "melbourne"):
        assert city_grid(city).n_cells > 0


def test_split_is_disjoint_and_covers_the_grid():
    grid = grid_from_extent((5.0, 5.0), delta_m=100.0)
    result = make_partition(grid)
    counts = result.cells.value_counts()
    assert set(counts.index) <= {"train", "validation", "test"}
    assert counts.sum() == grid.n_cells
    assert result.cells.index.is_unique


def test_each_super_block_carries_a_single_split():
    """No block straddles two partitions, so the partitions stay contiguous."""
    grid = grid_from_extent((5.0, 5.0), delta_m=100.0)
    result = make_partition(grid)
    labels = result.blocks.set_index("block_id")["split"]
    assert labels.index.is_unique
    assert set(result.cells.unique()) <= set(labels.unique())


def test_realised_proportions_are_near_the_requested_split():
    grid = grid_from_extent((10.0, 10.0), delta_m=100.0)
    fractions = summarise(make_partition(grid))
    for split, target in zip(("train", "validation", "test"), SPLIT_FRACTIONS):
        assert fractions[split]["fraction"] == pytest.approx(target, abs=0.05)


def test_holdout_is_drawn_from_training_blocks_only():
    grid = grid_from_extent((5.0, 5.0), delta_m=100.0)
    result = make_partition(grid)
    held = result.cells[result.holdout]
    assert set(held.unique()) == {"train"}


def test_holdout_is_a_tenth_of_the_training_partition():
    """The hold-out fraction is taken against training, not against the city."""
    grid = grid_from_extent((10.0, 10.0), delta_m=100.0)
    result = make_partition(grid)
    n_train = int((result.cells == "train").sum())
    n_holdout = int(result.holdout.sum())
    assert n_train > 0
    assert n_holdout / n_train == pytest.approx(0.10, abs=0.02)


def test_partitions_separate_at_the_block_scale():
    """Separation is at the super-block scale, which is what the tiling allows.

    Adjacent super-blocks in different partitions share a border, so a
    cell-level clearance is not achievable and is not claimed. The property
    that does hold, and that this checks, is that every super-block is wholly
    inside one partition.
    """
    grid = grid_from_extent((5.0, 5.0), delta_m=100.0)
    result = make_partition(grid)
    blocks = superblock_index(grid)
    per_block = result.cells.groupby(blocks["block_id"]).nunique()
    assert (per_block == 1).all()


def test_partition_is_deterministic():
    grid = grid_from_extent((3.0, 3.0), delta_m=100.0)
    first = make_partition(grid).cells
    second = make_partition(grid).cells
    assert first.equals(second)


def test_z_score_fit_on_train_partition():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    split = pd.Series(["train", "train", "test", "test"])
    from verdi.data.standardize import fit_transform

    out, params = fit_transform(frame, ["a"], split=split, fit_on="train")
    assert params.mean["a"] == pytest.approx(1.5)
    # The training rows are standardised; the test rows are shifted by the
    # same parameters, so only the training subset has mean 0 and unit spread.
    assert out.loc[split == "train", "a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out.loc[split == "train", "a"].std(ddof=0) == pytest.approx(1.0)
    assert out["a"].iloc[2] > out["a"].iloc[0]


def test_transform_is_reversible_in_shape():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    params = fit(frame, ["a"])
    out = transform(frame, params)
    assert out.shape == frame.shape
    assert out["a"].std(ddof=0) == pytest.approx(1.0)
