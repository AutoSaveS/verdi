"""Create the spatial split for a city and print its summary.

Usage:
    python scripts/make_partition.py --city melbourne --delta 100 --out data/partition_melbourne.csv

The split is deterministic for a given seed. See verdi/data/partition.py for the
choices the manuscript leaves open.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verdi.data.grid import city_grid                          # noqa: E402
from verdi.data.partition import make_partition, summarise     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nyc", choices=["nyc", "paris", "melbourne"])
    parser.add_argument("--delta", type=int, default=100, help="cell size in metres")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    grid = city_grid(args.city, delta_m=args.delta)
    result = make_partition(grid) if args.seed is None else _with_seed(grid, args.seed)
    summary = summarise(result)
    print(f"{args.city} at {args.delta} m: {grid.n_cells} cells in a "
          f"{grid.n_x} x {grid.n_y} grid")
    for split, values in summary.items():
        print(f"  {split:>18}: {int(values['cells']):>7} cells "
              f"({values['fraction']:.1%})")

    if args.out:
        frame = result.cells.rename("partition").to_frame()
        frame["cross_city_holdout"] = result.holdout
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        frame.to_csv(args.out)
        print(f"wrote {args.out}")
    return 0


def _with_seed(grid, seed: int):
    from verdi.data.partition import make_partition as build

    return build(grid, seed=seed)


if __name__ == "__main__":
    import os
    raise SystemExit(main())
