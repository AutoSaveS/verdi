# Data sources

Raw provider data is not distributed with this repository. Download it from
the providers under their own licences; the tables below list what each
variable needs.

## Study areas

| | New York City | Paris | Melbourne |
|---|---|---|---|
| Centre | 40.71 N, 74.01 W | 48.86 N, 2.35 E | 37.81 S, 144.96 E |
| Climate (Koppen) | Cfa | Cfb | Cfb |
| Summer window | Jun-Aug | Jun-Aug | Dec-Feb |
| Projected CRS | EPSG:32618 | EPSG:32631 | EPSG:32755 |
| Tree census | TreesCount! 2015 | Les Arbres de Paris | Urban Forest Visual |
| Census health field | Yes | No | Yes |
| Modalities retained | 5-6 | 3 | 2 |

Grid resolution is 100 m by default, ablated over 50-200 m.

## Sensing modalities

| ID | Modality | Resolution | Frequency | NYC | Paris | Melbourne |
|---|---|---|---|---|---|---|
| m1 | Satellite multispectral (Sentinel-2) | 10 m | 5-day | yes | yes | yes |
| m2 | Thermal infrared (Landsat 8/9 B10) | 100 m | 16-day | yes | no | no |
| m3 | Weather stations | point | hourly | yes | partial | no |
| m4 | ERA5-Land reanalysis | 9 km | hourly | yes | yes | no |
| m5 | SMAP soil moisture | 9 km | 2-3 day | yes | no | no |
| m6 | Static GIS | vector | static | yes | yes | yes |

Per-city configurations: NYC six modalities, Paris three, Melbourne two.

## Vegetation variables (V)

| ID | Name | Source | Derivation |
|---|---|---|---|
| v1 | species_diversity | Census | Shannon H' over species |
| v2 | dominant_spp_embed | Census | learnable embedding, d = 8 |
| v3 | tree_count | Census | spatial join and count |
| v4 | mean_dbh_cm | Census | harmonised units |
| v5 | canopy_cover_ratio | Sentinel-2 + ETH | NDVI > 0.4 area / grid area |
| v6 | planting_density | Census | trees per hectare |
| v7 | ndvi_mean | Sentinel-2 10 m | summer median, cloud-masked |
| v8 | ndvi_std | Sentinel-2 | within-grid pixel standard deviation |
| v9 | evi_mean | Sentinel-2 | EVI summer median |
| v10 | health_score | Census + Sentinel-2 | health grade combined with NDVI deviation |
| v11 | canopy_height_m | ETH canopy height 1 m | zonal mean |
| v12 | lai_mean | Sentinel-2 GEE | biophysical processor |

## Environmental variables (E)

| ID | Name | Source | Derivation |
|---|---|---|---|
| e1 | impervious_ratio | ESA WorldCover v100 | built-up / grid area |
| e2 | building_coverage | City footprints | footprint intersection / grid area |
| e3 | building_height_m | City + GHSL | area-weighted mean |
| e4 | sky_view_factor | Derived | hemispherical viewshed, 200 m |
| e5 | hw_ratio | Heights + OSM | length-weighted mean height / width |
| e6 | lst_mean_K | Landsat 8/9 B10 | atmospherically corrected, summer mean |
| e7 | lst_max_K | Landsat 8/9 B10 | summer maximum |
| e8 | soil_type_embed | WRB class | learnable embedding |
| e9 | soil_oc_g_kg | SoilGrids v2.0, 0-30 cm | depth-weighted mean |
| e10 | soil_clay_pct | SoilGrids v2.0, 0-30 cm | depth-weighted mean |
| e11 | solar_rad_kwh | ERA5-Land SSRD | accumulation to kWh/m2 per year |
| e12 | wind_speed_ms | ERA5-Land u10, v10 | sqrt(u^2 + v^2), summer mean |
| e13 | dist_green_m | Parks and OSM | Euclidean distance from the cell centre |
| e14 | land_use_embed | WorldCover | dominant class embedding |

Eight of the fourteen environmental variables come from globally uniform
products; `e1`-`e5` and `e13` need city-specific sources.

## Census field mapping

| Unified field | NYC | Melbourne | Paris |
|---|---|---|---|
| Species | `spc_latin` | `Scientific Name` | `GENRE` + `ESPECE` |
| DBH (cm) | `tree_dbh` x 2.54 | `Diameter Breast Height` | `CIRCONFERENCE` / pi |
| Health (0-1) | Good / Fair / Poor | ULE > 20 yr / 10-20 yr | not available |
| Coordinates | `lat`, `lon` | GeoJSON centroid | `geo_point_2d` |

Pass the unit of the Paris circumference to
`verdi.data.census.harmonise(circumference_unit=...)`.

## Licensing

Each provider's terms apply. Sentinel-2 and Landsat imagery are open
(Copernicus and USGS public domain); ERA5-Land and SMAP are free with
attribution; SoilGrids and ESA WorldCover are CC-BY; OSM data is ODbL; GHSL is
open; the three municipal tree censuses are published by their cities under
their own open-data terms.
