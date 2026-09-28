"""Build the three proxy labels on a synthetic city and write the output table.

Usage:
    python scripts/build_labels.py --city nyc --cells 400 --out data/example_nyc.csv

The city is simulated, so the numbers validate the plumbing and the label
formulas, not any scientific claim. Replace the synthetic inputs with the
provider data listed in docs/data_sources.md to build the real tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verdi.data.schema import columns, write_table          # noqa: E402
from verdi.labels.fusion import combine_availability_weighted  # noqa: E402
from verdi.labels.r_a import grid_r_a                        # noqa: E402
from verdi.labels.r_b import NdviPair, grid_r_b              # noqa: E402
from verdi.labels.r_c import grid_r_c                        # noqa: E402


def synthetic_census(city: str, cells: int, seed: int = 0) -> pd.DataFrame:
    """A synthetic per-tree census: about ten trees per grid cell."""
    rng = np.random.default_rng(seed)
    n = cells * 10
    grid_ids = np.repeat([f"g{k:04d}" for k in range(cells)], 10)
    grades = {
        "nyc": ["Good", "Fair", "Poor"],
        "melbourne": [">20yr", "10-20yr"],
        "paris": [],
    }[city]
    return pd.DataFrame(
        {
            "grid_id": grid_ids,
            "species": rng.choice(
                ["Platanus x acerifolia", "Acer platanoides", "Tilia cordata"], n,
                p=[0.7, 0.2, 0.1],
            ),
            "health_grade": rng.choice(grades, n) if grades else None,
            "ndvi": rng.uniform(0.2, 0.85, n),
            "dbh_cm": rng.uniform(8, 70, n),
            "canopy_area_m2": rng.uniform(4, 60, n),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nyc", choices=["nyc", "paris", "melbourne"])
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    trees = synthetic_census(args.city, args.cells, args.seed)
    rng = np.random.default_rng(args.seed + 1)

    # R*_A is unavailable where the census has no health field (Paris).
    if trees["health_grade"].isna().all():
        r_a = pd.Series(np.nan, index=[f"g{k:04d}" for k in range(args.cells)])
    else:
        r_a = grid_r_a(trees, args.city, canopy_area_col="canopy_area_m2")

    # R*_B: one synthetic heat-event pair per cell where a "cloud-free" flag holds.
    grid_ids = [f"g{k:04d}" for k in range(args.cells)]
    pairs = []
    for grid_id in grid_ids:
        if rng.random() < 0.6:
            pre = float(rng.uniform(0.3, 0.8))
            pairs.append(
                NdviPair(
                    grid_id=grid_id,
                    event_start=pd.Timestamp("2021-07-15"),
                    days_before=int(rng.integers(1, 10)),
                    days_after=int(rng.integers(1, 10)),
                    ndvi_pre=pre,
                    ndvi_post=float(np.clip(pre + rng.normal(0, 0.08), 0.05, 1.0)),
                    cloud_pct=float(rng.uniform(0, 30)),
                )
            )
    r_b = grid_r_b(pairs, grid_ids)

    # R*_C: the dominant species of a cell against the species baseline.
    cell_species = trees.groupby("grid_id")["species"].agg(
        lambda values: values.value_counts().idxmax()
    ).reindex(grid_ids)
    cell_ndvi = trees.groupby("grid_id")["ndvi"].mean().reindex(grid_ids)
    cell_ndvi = cell_ndvi.fillna(pd.Series(rng.uniform(0.2, 0.85, len(grid_ids)), index=grid_ids))
    r_c = grid_r_c(cell_ndvi, cell_species.fillna("Platanus x acerifolia"))

    table = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "city": args.city,
            "R_A": r_a.reindex(grid_ids),
            "R_B": r_b.reindex(grid_ids),
            "R_C": r_c.reindex(grid_ids),
            "dominant_species": cell_species.fillna("Platanus x acerifolia"),
            "tree_count_obs": trees.groupby("grid_id")["ndvi"].size().reindex(grid_ids, fill_value=0),
        }
    )
    labels = table[["R_A", "R_B", "R_C"]]
    table["R_fused"] = combine_availability_weighted(labels)
    table = table.reindex(columns=columns())

    out = args.out or f"data/example_{args.city}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    written = write_table(table, out)
    coverage = {label: f"{table[label].notna().mean():.0%}" for label in ("R_A", "R_B", "R_C")}
    print(f"wrote {written}: {len(table)} cells; label coverage {coverage}")
    print("synthetic data: the numbers exercise the pipeline, not the manuscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
