"""Checks of the proxy-label formulas against Appendix A.5."""

import numpy as np
import pandas as pd
import pytest

from verdi.labels import (
    NdviPair,
    detect_heat_events,
    grid_r_a,
    grid_r_b,
    grid_r_c,
    harmonise_health,
    retention,
    species_baseline,
)


def _trees(n=600, per_grid=10, seed=0):
    rng = np.random.default_rng(seed)
    grids = [f"g{k:03d}" for k in range(n // per_grid)]
    return pd.DataFrame(
        {
            "grid_id": np.repeat(grids, per_grid),
            "species": "Platanus x acerifolia",
            "health_grade": rng.choice(["Good", "Fair", "Poor"], n),
            "ndvi": rng.uniform(0.2, 0.8, n),
            "dbh_cm": rng.uniform(10, 60, n),
        }
    )


def test_health_mapping_matches_table_a19():
    grades = pd.Series(["Good", "Fair", "Poor"])
    assert harmonise_health(grades, "nyc").tolist() == [1.0, 0.6, 0.2]
    assert harmonise_health(pd.Series([">20yr", "10-20yr"]), "melbourne").tolist() == [1.0, 0.6]


def test_paris_has_no_health_field():
    with pytest.raises(ValueError):
        harmonise_health(pd.Series(["Good"]), "paris")


def test_r_a_is_a_rank_in_the_unit_interval():
    trees = _trees()
    values = grid_r_a(trees, "nyc", canopy_area_col=None)
    assert values.notna().all()
    assert values.min() == pytest.approx(1.0 / len(values))
    assert values.max() == pytest.approx(1.0)


def test_r_a_min_species_threshold():
    """Species below 500 individuals are not scored."""
    small = _trees(n=100, per_grid=10)
    assert grid_r_a(small, "nyc", canopy_area_col=None).isna().all()


def test_retention_formula_and_floor():
    assert retention(0.5, 0.4) == pytest.approx(0.8)          # |0.4-0.5| / 0.5
    assert retention(0.1, 0.4) == pytest.approx(0.0)          # denominator floored at 0.2
    assert retention(0.5, 0.9) == pytest.approx(0.2)          # greening also lowers the value
    assert 0.0 <= retention(0.05, 0.02) <= 1.0


def test_retention_is_clipped():
    assert retention(0.2, 1.0) == pytest.approx(0.0)


def _summer_series(seed: int = 0, days: int = 60):
    """A wavy baseline with one three-day heat spike.

    Sixty days are used because the 95th percentile then sits just above the
    fourth-largest value, so exactly the three spike days qualify: at thirty
    days the percentile lands inside the spike and only two days remain above
    it. The spike sits well above every baseline value.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-06-01", periods=days, freq="D")
    baseline = 24.0 + 3.0 * np.sin(np.arange(days) / 3.0) + rng.normal(0, 0.5, days)
    baseline[30:33] = [45.0, 46.0, 45.5]
    return pd.Series(baseline, index=index)


def test_heat_events_need_consecutive_days():
    tmax = _summer_series()
    threshold = float(np.percentile(tmax, 95))
    above = tmax[tmax > threshold]
    assert len(above) == 3, "the fixture must leave three days above the threshold"

    events = detect_heat_events(tmax, percentile=95, min_days=3)
    assert events == [(pd.Timestamp("2020-07-01"), pd.Timestamp("2020-07-03"))]
    assert detect_heat_events(tmax, percentile=95, min_days=5) == []


def test_heat_events_percentile_and_min_days_are_parameters():
    tmax = _summer_series()
    days = lambda evs: sum((end - start).days + 1 for start, end in evs)
    assert days(detect_heat_events(tmax, percentile=80, min_days=1)) > 0
    assert days(detect_heat_events(tmax, percentile=99, min_days=3)) == 0


def test_grid_r_b_uses_only_usable_pairs():
    good = NdviPair("c1", pd.Timestamp("2020-06-08"), 2, 3, 0.55, 0.48, 4.0)
    cloudy = NdviPair("c2", pd.Timestamp("2020-06-08"), 2, 3, 0.55, 0.48, 40.0)
    late = NdviPair("c3", pd.Timestamp("2020-06-08"), 30, 30, 0.55, 0.48, 1.0)
    out = grid_r_b([good, cloudy, late], ["c1", "c2", "c3"])
    assert out["c1"] == pytest.approx(retention(0.55, 0.48))
    assert np.isnan(out["c2"]) and np.isnan(out["c3"])


def test_species_baseline_statistic_is_explicit():
    ndvi = pd.Series([0.2, 0.4, 0.9])
    species = pd.Series(["s"] * 3)
    median = species_baseline(ndvi, species, "median")
    mean = species_baseline(ndvi, species, "mean")
    assert median.loc["s", "baseline"] == pytest.approx(0.4)
    assert mean.loc["s", "baseline"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        species_baseline(ndvi, species, "mode")


def test_r_c_is_a_sigmoid_ratio():
    index = list("abcdef")
    ndvi = pd.Series([0.5, 0.4, 0.3, 0.6, 0.55, 0.45], index=index)
    values = grid_r_c(ndvi, pd.Series(["s"] * 6, index=index))
    assert values.notna().all()
    assert values.between(0.0, 1.0).all()
    # Above the baseline gives > 0.5, below gives < 0.5.
    assert (values[ndvi > ndvi.median()] > 0.5).all()
