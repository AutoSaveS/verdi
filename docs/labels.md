# Proxy labels

The three labels are operational indicators, not measurements of ecological
resilience. They are built in `verdi/labels/`.

| Label | What it is | Built from | Availability |
|---|---|---|---|
| `R*_A` | Relative cross-environment condition | Census health, within-species NDVI and DBH percentiles | NYC, Melbourne (Paris has no census health field) |
| `R*_B` | Short-term NDVI retention around qualifying heat events | ERA5-Land daily maximum temperature, Sentinel-2 pre/post pairs | Where a qualifying event and a cloud-free pair exist |
| `R*_C` | Relative deviation from a species baseline | Sentinel-2 NDVI | All cells in all three cities; the primary reported target |

`R*_C` is the only label complete across cities and is the target of the main
prediction tables. The other combinations in the manuscript are design
descriptions and are implemented in `verdi/labels/fusion.py` for completeness.

## R*_A

```
P_i  = 0.4 * health_i + 0.4 * ndvi_percentile_i + 0.2 * dbh_percentile_i
R*_A = rank(mean P over the cell) / N_s
```

* Only species with at least 500 individuals are scored; a tree of a smaller
  species stays in the frame with `eligible = False`.
* Multi-species cells are averaged by canopy area.
* Health grades are harmonised from the census classes: NYC Good / Fair / Poor
  = 1.0 / 0.6 / 0.2; Melbourne ULE > 20 yr = 1.0 and 10-20 yr = 0.6.

Left open by the manuscript, decided here and exposed as arguments:

* `N_s` is the number of scored cells, and the rank is ascending with average
  ties.
* The percentile scale is `pandas.rank(pct=True)` within species.
* The Melbourne ULE value below 10 years is not defined; such trees raise a
  `KeyError` rather than being silently dropped.
* Tree-level NDVI extraction (point or buffer) and the canopy-area source are
  caller-supplied through `canopy_area_col`.

## R*_B

```
R*_B = 1 - |NDVI_post - NDVI_pre| / max(NDVI_pre, 0.2)   clipped to [0, 1]
```

* A qualifying heat event is at least three consecutive days above the
  city-specific 95th percentile of the ERA5-Land record for 2018-2023.
* Pre/post Sentinel-2 pairs must fall within 14 days and have cloud cover below
  10 %.
* Cells without a qualifying event or a usable pair are `NaN`.

Note the absolute value: a greening response also lowers the value. That is
what the manuscript writes, and `verdi.labels.retention` reproduces it.

Left open by the manuscript, decided here: the temperature variable and the
hourly-to-daily aggregation (a calendar-day maximum), the base period of the
percentile (the series passed in), the anchor of the 14-day window (event
start), whether the cloud threshold is per scene or per pixel (per record), and
how several events in one cell are combined (`case="mean"` or `"last"`).

## R*_C

```
R*_C = sigmoid( (NDVI_grid - baseline(species)) / spread(species) )
```

The baseline statistic is a parameter: the manuscript writes the species
"baseline" without saying median or mean, so `statistic="median"` or `"mean"`
is passed explicitly and can be recorded with the run. Cells whose species has
a single individual or zero spread return `NaN`.

The manuscript also describes `R*_C` once as "min-max standardized to [0, 1]"
while the label definition is a sigmoid. `apply_minmax=True` reproduces that
other reading; the sigmoid is the default.

## What the labels do not measure

Recovery capacity, persistence, functional stability and post-disturbance
regeneration. They are also city-relative rather than a common absolute scale,
and `R*_C` shares NDVI with several input variables, so the evaluation
demonstrates predictive agreement under the specified inputs, not independent
ecological validation.
