"""The per-city output table.

Per-city Parquet files with grid identifiers, UTM/WGS84 coordinates,
standardized V and E vectors, the three resilience labels with availability
flags, dominant species, tree count, data completeness and partition
assignment (Appendix A.5). Column names are declared in one place so they can
be renamed without touching the label code.

Parquet output needs ``pyarrow``; :func:`write_table` falls back to CSV and
says so, because the dependency is optional in ``requirements.txt``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..config import LATENT_FACTORS

#: Vegetation configuration vector (Table A.20).
V_COLUMNS: List[str] = [
    "species_diversity",      # v1  Shannon H' over species
    "dominant_spp_embed",     # v2  learnable, d = 8
    "tree_count",             # v3  spatial join + count
    "mean_dbh_cm",            # v4  harmonised units
    "canopy_cover_ratio",     # v5  NDVI > 0.4 area / grid area
    "planting_density",       # v6  trees per hectare
    "ndvi_mean",              # v7  summer median, cloud-masked
    "ndvi_std",               # v8  within-grid pixel standard deviation
    "evi_mean",               # v9  EVI summer median
    "health_score",           # v10 census health grade + NDVI deviation
    "canopy_height_m",        # v11 zonal mean of ETH canopy height
    "lai_mean",               # v12 GEE biophysical processor
]

#: Environmental condition vector (Table A.21).
E_COLUMNS: List[str] = [
    "impervious_ratio",       # e1  WorldCover built-up / grid area
    "building_coverage",      # e2  footprint intersection / grid area
    "building_height_m",      # e3  area-weighted mean
    "sky_view_factor",        # e4  hemispherical viewshed, 200 m
    "hw_ratio",               # e5  length-weighted mean H / W
    "lst_mean_K",             # e6  Landsat 8/9 B10, summer mean
    "lst_max_K",              # e7  Landsat 8/9 B10, summer maximum
    "soil_type_embed",        # e8  WRB class embedding
    "soil_oc_g_kg",           # e9  SoilGrids, 0-30 cm depth-weighted mean
    "soil_clay_pct",          # e10 SoilGrids, 0-30 cm depth-weighted mean
    "solar_rad_kwh",          # e11 SSRD accumulation, annual
    "wind_speed_ms",          # e12 sqrt(u10^2 + v10^2), summer mean
    "dist_green_m",           # e13 Euclidean distance to nearest green space
    "land_use_embed",         # e14 dominant WorldCover class embedding
]

#: Sensing modalities m1-m6 (Table C.29, Panel A).
MODALITIES: Dict[str, Dict[str, str]] = {
    "m1": {"name": "Satellite multispectral", "resolution": "10-30 m", "frequency": "5-16 d"},
    "m2": {"name": "Thermal infrared (Landsat 8/9 B10)", "resolution": "100 m", "frequency": "16 d"},
    "m3": {"name": "Weather stations", "resolution": "point", "frequency": "hourly"},
    "m4": {"name": "ERA5-Land reanalysis", "resolution": "9 km", "frequency": "hourly"},
    "m5": {"name": "SMAP soil moisture", "resolution": "9 km", "frequency": "2-3 d"},
    "m6": {"name": "Static GIS", "resolution": "vector", "frequency": "static"},
}

#: Per-city modality availability (Table C.29, Panel A).
MODALITY_AVAILABILITY: Dict[str, Dict[str, bool]] = {
    "nyc": {"m1": True, "m2": True, "m3": True, "m4": True, "m5": True, "m6": True},
    "paris": {"m1": True, "m2": False, "m3": False, "m4": True, "m5": False, "m6": True},
    "melbourne": {"m1": True, "m2": False, "m3": False, "m4": False, "m5": False, "m6": True},
}

#: Latent factors exported for the typology step.
LATENT_COLUMNS: List[str] = [f"z_{name}" for name in LATENT_FACTORS]

#: Columns present in the per-city table.
def columns() -> List[str]:
    return (
        ["grid_id", "x_utm", "y_utm", "lon", "lat", "city"]
        + V_COLUMNS
        + E_COLUMNS
        + LATENT_COLUMNS
        + ["R_A", "R_B", "R_C",
           "R_A_available", "R_B_available", "R_C_available",
           "dominant_species", "tree_count_obs", "completeness", "partition",
           "cross_city_holdout"]
    )


def coerce(table: pd.DataFrame) -> pd.DataFrame:
    """Return the table with every declared column present and typed.

    Missing columns are added as NaN so downstream code can rely on the
    schema; availability flags default to False for a missing label column.
    """
    out = table.copy()
    for column in columns():
        if column not in out.columns:
            out[column] = pd.NA
    for label in ("R_A", "R_B", "R_C"):
        flag = f"{label}_available"
        if label in table.columns:
            out[flag] = table[label].notna()
        else:
            out[flag] = False
    out["cross_city_holdout"] = out["cross_city_holdout"].fillna(False).astype(bool)
    return out[columns()]


def write_table(table: pd.DataFrame, path: str, prefer_parquet: bool = True) -> str:
    """Write the table as Parquet when ``pyarrow`` is available, else CSV.

    Returns the path actually written.
    """
    if prefer_parquet:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            csv_path = path.rsplit(".", 1)[0] + ".csv"
            table.to_csv(csv_path, index=False)
            return csv_path
        table.to_parquet(path, index=False)
        return path
    table.to_csv(path, index=False)
    return path


def read_table(path: str) -> pd.DataFrame:
    """Read a table written by :func:`write_table`."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
