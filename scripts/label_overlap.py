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
    python scripts/label_overlap.py --figure label_overlap.pdf   # also draw the figure
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


def draw_figure(path: str, example: pd.DataFrame, links, a_stats, h_stats,
                share_range, rb_mean: float) -> None:
    """Two panels: R*_A against R*_C at no link, and correlations against the link."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    blue, green, grey = "#2B6CB0", "#2F855A", "#718096"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})

    sample = example.sample(n=min(1500, len(example)), random_state=0)
    ax1.scatter(sample.R_C, sample.R_A, s=3, alpha=0.35, color=blue, linewidths=0)
    rho = spearmanr(example.R_A, example.R_C).correlation
    ax1.set_xlabel(r"$R^*_C$")
    ax1.set_ylabel(r"$R^*_A$")
    ax1.set_title(r"(a) Health grade unrelated to condition", fontsize=9, loc="left")
    ax1.text(0.04, 0.93, rf"$\rho = {rho:.2f}$", transform=ax1.transAxes, va="top")
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    x = np.asarray(links, dtype=float)
    for stats, colour, label in ((a_stats, blue, r"$R^*_A$ vs. $R^*_C$"),
                                 (h_stats, green, r"Census health vs. $R^*_C$")):
        m = np.array([v[0] for v in stats]); sd = np.array([v[1] for v in stats])
        ax2.errorbar(x, m, yerr=sd, color=colour, marker="o", ms=3.5, lw=1.2, capsize=2, label=label)
    ax2.vlines(0.0, share_range[0], share_range[1], color=blue, lw=4, alpha=0.25,
               label="Range over grade shares")
    ax2.axhline(rb_mean, color=grey, lw=0.9, ls="--", label=r"$R^*_B$ vs. $R^*_C$")
    ax2.set_xlabel("Link between census health grade and tree condition")
    ax2.set_ylabel(r"Spearman $\rho$")
    ax2.set_xticks(x)
    ax2.set_xlim(-0.05, x.max() + 0.05)
    ax2.set_ylim(-0.08, 1.0)
    ax2.set_title("(b) Correlation expected from construction", fontsize=9, loc="left")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"figure written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--figure", help="also draw the figure to this path (PDF or PNG)")
    args = parser.parse_args()
    seeds = range(args.seeds)
    base = {"Good": 0.70, "Fair": 0.22, "Poor": 0.08}

    print("Spearman correlation, mean (SD) over seeds; grade shares Good/Fair/Poor = 70/22/8 %")
    print(f"{'health-condition link':>22}  {'R*_A vs R*_C':>14}  {'health vs R*_C':>15}")
    links = (0.0, 0.2, 0.4, 0.6)
    a_stats, h_stats, example = [], [], None
    for link in links:
        frames = [simulate_city(link, base, args.cells, s) for s in seeds]
        if link == 0.0:
            example = frames[0]
        a = [spearmanr(f.R_A, f.R_C).correlation for f in frames]
        h = [spearmanr(f.health, f.R_C).correlation for f in frames]
        a_stats.append((np.mean(a), np.std(a, ddof=1)))
        h_stats.append((np.mean(h), np.std(h, ddof=1)))
        print(f"{link:>22.1f}  {mean_sd(a):>14}  {mean_sd(h):>15}")

    print("\nR*_A vs R*_C at link = 0 under other grade shares")
    share_means = []
    for shares in ({"Good": 0.90, "Fair": 0.08, "Poor": 0.02},
                   {"Good": 0.50, "Fair": 0.30, "Poor": 0.20}):
        frames = [simulate_city(0.0, shares, args.cells, s) for s in seeds]
        a = [spearmanr(f.R_A, f.R_C).correlation for f in frames]
        share_means.append(float(np.mean(a)))
        label = "/".join(f"{int(round(100 * shares[k]))}" for k in ("Good", "Fair", "Poor"))
        print(f"  {label} %: {mean_sd(a)}")

    rb = [r_b_vs_r_c(args.cells, s) for s in seeds]
    print(f"\nR*_B vs R*_C, loss unrelated to greenness: {mean_sd(rb)}")

    if args.figure:
        draw_figure(args.figure, example, links, a_stats, h_stats,
                    (min(share_means), max(share_means)), float(np.mean(rb)))


if __name__ == "__main__":
    main()
