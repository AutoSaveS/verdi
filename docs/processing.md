# Processing pipeline

`verdi/data/` implements the data-processing steps of Appendix A.7.

## Steps

1. Generate the Delta x Delta grid for the city (`verdi.data.grid.city_grid`).
2. Census spatial join producing `v1`-`v4`, `v6`, `v10`.
3. Raster zonal statistics producing `v5`, `v7`-`v9`, `v11`-`v12`, `e1`,
   `e6`-`e7`.
4. Building footprint overlay producing `e2`-`e5`.
5. SoilGrids WCS and ERA5-Land bilinear interpolation producing `e8`-`e12`.
6. Proximity, land-use embedding and the three proxy labels (`e13`, `e14`,
   `R*_A`, `R*_B`, `R*_C`).

Satellite-derived variables use identical Google Earth Engine scripts
parameterised only by city boundary, summer window and output CRS.
Collections: `S2_SR_HARMONIZED`, `LC08/C02/T1_L2`, WorldCover `v100` (2020).
Cloud threshold below 10 %, pixel-wise median composite.

## Exclusion criteria

A cell is dropped when any of the following holds:

* water-body fraction above 80 %;
* no vegetation: zero trees and NDVI below 0.15;
* data completeness below 70 %;
* physical outliers: NDVI outside [0, 1], DBH above 200 cm, building height
  above 300 m.

## Standardisation

Per-city z-scores are fitted and the parameters saved for inference.
`verdi.data.standardize.fit_transform` fits on the training partition by
default, so validation and test statistics do not leak into the transform;
`fit_on="all"` fits on the whole city. Categorical variables use learnable embeddings
rather than z-scores.

## Spatial partitioning

* 5Delta x 5Delta super-blocks are dealt to train / validation / test in
  70 / 15 / 15 proportions.
* Test blocks are moved to positions that keep at least a 4Delta separation
  from the other partitions.
* A further 10 % per city is held out for cross-city generalisation; it is
  drawn from the training blocks.

The assignment is deterministic for a given seed; `PARTITION_SEED` in
`verdi/config.py` fixes it.

## Output table

Per-city Parquet files (CSV when `pyarrow` is absent) with: grid identifiers,
UTM and WGS84 coordinates, standardised V and E vectors, latent factors, the
three labels with availability flags, dominant species, tree count, data
completeness, partition assignment and the cross-city hold-out flag. Column
names are declared in `verdi/data/schema.py`.
