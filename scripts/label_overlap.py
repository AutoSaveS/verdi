"""Agreement between the proxy labels that follows from their construction alone.

R*_A includes the within-species NDVI percentile and R*_C is computed from
within-species NDVI, so the two labels agree partly by construction. This
script runs the label code on simulated cities to measure that agreement.

Each simulated cell has a latent condition that sets its greenness. The census
health grade of each tree is linked to that condition with strength ``link``:
at ``link = 0`` the health grade carries no information about the tree, so any
agreement between R*_A and R*_C comes from the shared NDVI component. DBH is
independent of condition. Spearman correlations are averaged over seeds.

    python scripts/label_overlap.py --cells 4000 --seeds 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verdi.config import HEALTH_GRADES  # noqa: E402
from verdi.labels import grid_r_a, grid_r_c, retention  # noqa: E402

SPECIES = "Platanus x acerifolia"


def simulate_city(link: float, shares: dict, n_cells: int, seed: int) -> pd.DataFrame:
    """Cell-level R*_A, R*_C and mean census health for one simulated city."""
    rng = np.random.default_rng(seed)
    condition = rng.normal(size=n_cells)
    ndvi_cell = 0.55 + 0.08 * condition
    cut_poor, cut_fair = shares["Poor"], shares["Poor"] + shares["Fair"]
    rows = []
    for g in range(n_cells):
        for _ in range(rng.integers(1, 6)):
            latent = link * condition[g] + math.sqrt(1.0 - link ** 2) * rng.normal()
            u = 0.5 * (1.0 + math.erf(latent / math.sqrt(2.0)))
            grade = "Poor" if u < cut_poor else ("Fair" if u < cut_fair else "Good")
            rows.append({"grid_id": f"g{g:05d}", "species": SPECIES, "health_grade": grade,
                         "ndvi": ndvi_cell[g] + 0.03 * rng.normal(),
                         "dbh_cm": rng.uniform(10.0, 90.0),
                         "canopy_area_m2": rng.uniform(20.0, 120.0)})
    trees = pd.DataFrame(rows)
    r_a = grid_r_a(trees, "nyc")
    cell_ndvi = trees.groupby("grid_id")["ndvi"].mean()
    r_c = grid_r_c(cell_ndvi, pd.Series(SPECIES, index=cell_ndvi.index))
    health = trees["health_grade"].map(HEALTH_GRADES["nyc"]).groupby(trees["grid_id"]).mean()
    return pd.DataFrame({"R_A": r_a, "R_C": r_c, "health": health}).dropna()


def r_b_vs_r_c(n_cells: int, seed: int) -> float:
    """Spearman(R*_B, R*_C) when the heat-event NDVI loss is unrelated to greenness."""
    rng = np.random.default_rng(seed)
    pre = 0.55 + 0.08 * rng.normal(size=n_cells)
    loss = np.abs(rng.normal(0.04, 0.03, size=n_cells))
    r_b = np.array([retention(p, p - l) for p, l in zip(pre, loss)])
    r_c = grid_r_c(pd.Series(pre), pd.Series(SPECIES, index=range(n_cells)))
    return float(spearmanr(r_b, r_c).correlation)


def mean_sd(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ({values.std(ddof=1):.2f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    seeds = range(args.seeds)
    base = {"Good": 0.70, "Fair": 0.22, "Poor": 0.08}

    print("Spearman correlation, mean (SD) over seeds; grade shares Good/Fair/Poor = 70/22/8 %")
    print(f"{'health-condition link':>22}  {'R*_A vs R*_C':>14}  {'health vs R*_C':>15}")
    for link in (0.0, 0.2, 0.4, 0.6):
        frames = [simulate_city(link, base, args.cells, s) for s in seeds]
        a = [spearmanr(f.R_A, f.R_C).correlation for f in frames]
        h = [spearmanr(f.health, f.R_C).correlation for f in frames]
        print(f"{link:>22.1f}  {mean_sd(a):>14}  {mean_sd(h):>15}")

    print("\nR*_A vs R*_C at link = 0 under other grade shares")
    for shares in ({"Good": 0.90, "Fair": 0.08, "Poor": 0.02},
                   {"Good": 0.50, "Fair": 0.30, "Poor": 0.20}):
        frames = [simulate_city(0.0, shares, args.cells, s) for s in seeds]
        a = [spearmanr(f.R_A, f.R_C).correlation for f in frames]
        label = "/".join(f"{int(round(100 * shares[k]))}" for k in ("Good", "Fair", "Poor"))
        print(f"  {label} %: {mean_sd(a)}")

    print(f"\nR*_B vs R*_C, loss unrelated to greenness: {mean_sd([r_b_vs_r_c(args.cells, s) for s in seeds])}")


if __name__ == "__main__":
    main()
