"""Harmonise the three municipal tree censuses into one tree frame.

Table A.17 maps each city's fields onto the unified
columns used by the pipeline:

======================  =================  ==================  ==========================
Unified field           NYC                Melbourne           Paris
======================  =================  ==================  ==========================
species (Latin)         ``spc_latin``      ``Scientific Name`` ``GENRE`` + ``ESPECE``
DBH (cm)                ``tree_dbh`` x 2.54  ``Diameter Breast   ``CIRCONFERENCE`` / pi
                        (in -> cm)         Height`` (cm)
health (0-1)            Good 1.0 / Fair     ULE >20 yr 1.0 /    unavailable
                        0.6 / Poor 0.2     10-20 yr 0.6
coordinates             ``lat``, ``lon``   GeoJSON centroid    ``geo_point_2d``
======================  =================  ==================  ==========================

The unit of the Paris circumference is an argument, and unmapped health
classes raise rather than silently becoming NaN.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..labels.r_a import harmonise_health

#: Source field names per city, as in Table A.17.
FIELD_MAP: Dict[str, Dict[str, Optional[str]]] = {
    "nyc": {
        "species": "spc_latin",
        "dbh": "tree_dbh",
        "dbh_unit": "in",
        "health": "health",
        "lat": "lat",
        "lon": "lon",
    },
    "melbourne": {
        "species": "Scientific Name",
        "dbh": "Diameter Breast Height",
        "dbh_unit": "cm",
        "health": "ULE",
        "lat": None,   # GeoJSON centroid: supplied by the caller
        "lon": None,
        "geometry": "geometry",
    },
    "paris": {
        "species": None,          # GENRE + ESPECE
        "species_parts": ("GENRE", "ESPECE"),
        "dbh": "CIRCONFERENCE",
        "dbh_unit": "circumference",
        "health": None,           # no health field
        "lat": "geo_point_2d_lat",
        "lon": "geo_point_2d_lon",
        "point": "geo_point_2d",
    },
}

INCH_TO_CM = 2.54


def to_db_cm(values: pd.Series, unit: str) -> pd.Series:
    """Convert a diameter measurement to centimetres.

    ``unit`` is ``"in"``, ``"cm"`` or ``"circumference"`` (a circumference in
    the same unit as the length; the caller passes that unit through
    ``circumference_unit`` in :func:`harmonise`).
    """
    if unit == "in":
        return values.astype(float) * INCH_TO_CM
    if unit == "cm":
        return values.astype(float)
    raise ValueError(f"unsupported DBH unit {unit!r}")


def harmonise(
    frame: pd.DataFrame,
    city: str,
    circumference_unit: str = "cm",
    geometry_centroids: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a frame with the unified columns ``species``, ``dbh_cm``,
    ``health_grade``, ``lat`` and ``lon``.

    ``geometry_centroids`` is an optional frame indexed like ``frame`` giving
    ``lat`` and ``lon`` for the Melbourne GeoJSON records.
    """
    spec = FIELD_MAP[city]
    out = pd.DataFrame(index=frame.index)

    if spec.get("species_parts"):
        genus, epithet = spec["species_parts"]
        out["species"] = (
            frame[genus].astype(str).str.strip()
            + " "
            + frame[epithet].astype(str).str.strip()
        )
    else:
        out["species"] = frame[spec["species"]].astype(str).str.strip()

    unit = spec["dbh_unit"]
    if unit == "circumference":
        out["dbh_cm"] = frame[spec["dbh"]].astype(float) / np.pi
        if circumference_unit != "cm":
            out["dbh_cm"] = to_db_cm(out["dbh_cm"], circumference_unit)
    else:
        out["dbh_cm"] = to_db_cm(frame[spec["dbh"]], unit)

    if spec["health"] is None:
        out["health_grade"] = pd.NA
        out["health"] = np.nan
    else:
        out["health_grade"] = frame[spec["health"]]
        out["health"] = harmonise_health(out["health_grade"], city)

    if spec.get("geometry") and geometry_centroids is not None:
        out["lat"] = geometry_centroids["lat"]
        out["lon"] = geometry_centroids["lon"]
    elif spec.get("point"):
        point = frame[spec["point"]]
        if isinstance(point, pd.Series) and isinstance(point.iloc[0], dict):
            out["lat"] = point.map(lambda p: p["lat"])
            out["lon"] = point.map(lambda p: p["lon"])
        else:
            out["lat"] = frame[spec["lat"]]
            out["lon"] = frame[spec["lon"]]
    else:
        out["lat"] = frame[spec["lat"]]
        out["lon"] = frame[spec["lon"]]

    out["city"] = city
    return out


def completeness(frame: pd.DataFrame, columns: Optional[list] = None) -> pd.Series:
    """Share of non-null values per row over ``columns`` (default: all but id).

    Used for the ``< 70 %`` completeness exclusion.
    """
    cols = columns if columns is not None else [c for c in frame.columns if c not in {"grid_id", "city"}]
    return frame[cols].notna().mean(axis=1)
