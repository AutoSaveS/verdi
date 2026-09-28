<div align="center">

![VERDI](assets/banner.png)

![Tests](https://img.shields.io/badge/tests-26%20passing-8CC85A?labelColor=10221C)
![Python](https://img.shields.io/badge/python-3.9%2B-5AAAC8?labelColor=10221C)
![PyTorch](https://img.shields.io/badge/backend-PyTorch-8CC85A?labelColor=10221C)
![License](https://img.shields.io/badge/license-MIT-5AAAC8?labelColor=10221C)
![Status](https://img.shields.io/badge/manuscript-under%20review-9FB8AE?labelColor=10221C)

**V**egetation–**E**nvironment **R**esilience **D**iagnostic **I**ntelligence —
proxy-label construction, data processing and a reference implementation of the
three-stage framework.

</div>

VERDI couples vegetation state and environmental conditions to assess three
operational, proxy-based resilience indicators for urban vegetation, and to
translate the resulting factor readouts into diagnostic narratives. The
implementation covers municipal tree censuses, Sentinel-2 and Landsat imagery,
ERA5-Land reanalysis, SMAP soil moisture and static GIS layers for three
metropolitan areas (New York City, Paris, Melbourne) on a 100 m grid.

> Companion code for the manuscript *"From observation to evaluation: Coupling
> vegetation state and environmental conditions for near-real-time assessment of
> urban vegetation resilience indicators"* (Urban Forestry & Urban Greening,
> under review).

## Scope of this repository

The label-construction and data-processing code implements the operational
definitions of Section 3.2 and Appendix A of the manuscript. The Stage 1–3
model code is a **reference implementation written from the architecture
description** in Section 3.3 and Appendix G: it is not the code that produced
the manuscript's reported numbers, it ships no trained weights, and Stage 3
stops at the prompt — no language model is called and no fine-tuned adapters are
distributed. No processed city datasets are included; see [Data](#data).

The three labels are operational indicators. They are **not** measurements of
complete ecological resilience, they do not capture recovery, persistence,
functional stability or post-disturbance regeneration, and the model readouts
are associative rather than causal. Which analyses the manuscript reports, and
which are protocol only, is recorded in
[docs/reporting_status.md](docs/reporting_status.md).

## Architecture

```mermaid
flowchart LR
    A["Observed and externally<br/>derived inputs"]:::data --> B["Stage 1<br/>Spatial World Model<br/>coupled latent, K=6"]:::model
    B --> C["Stage 2<br/>Masked Sensor Transformer<br/>fusion under missing modalities"]:::model
    C --> D["Stage 3<br/>attribution, typology,<br/>diagnostic prompts"]:::stage3
    D --> E["Proposed decision support<br/>retrospective scenarios"]:::use
    L["Proxy labels<br/>R*A, R*B, R*C"]:::label -.-> B
    L -.-> C
    classDef data fill:#10221C,stroke:#5AAAC8,color:#E9F4EE
    classDef model fill:#10221C,stroke:#8CC85A,color:#E9F4EE
    classDef label fill:#10221C,stroke:#9FB8AE,color:#E9F4EE
    classDef stage3 fill:#10221C,stroke:#8CC85A,color:#E9F4EE
    classDef use fill:#10221C,stroke:#9FB8AE,color:#E9F4EE
```

## Repository layout

| Path | Purpose |
|---|---|
| `verdi/config.py` | Every manuscript parameter in one place; parameters the manuscript leaves open are marked `IMPLEMENTATION CHOICE` |
| `verdi/labels/` | `R*_A`, `R*_B`, `R*_C` and the label-fusion strategies |
| `verdi/data/` | Census harmonisation, grid construction, exclusion rules, z-scores, spatial partitioning, output schema |
| `verdi/model/` | Reference implementation of Stages 1–3 and their losses |
| `verdi/eval/` | Prediction, representation and calibration metrics |
| `scripts/` | Runnable entry points for the labels and the spatial split |
| `docs/` | Label definitions, data sources, processing rules, model reference, reporting status |
| `tests/` | Synthetic-data checks of the formulas, the split and the model shapes |

## Proxy labels

| Label | Meaning | Built from | Availability |
|---|---|---|---|
| `R*_A` | Relative cross-environment condition | Census health and within-species NDVI and DBH percentiles | NYC, Melbourne (Paris has no census health field) |
| `R*_B` | Short-term NDVI retention around qualifying heat events | ERA5-Land daily maximum temperature, Sentinel-2 pre/post pairs | Cells with a qualifying event and a cloud-free pair |
| `R*_C` | Relative deviation from a species baseline | Sentinel-2 NDVI | All cells in all three cities; the primary reported target |

Formulas, thresholds and the choices the manuscript leaves open are documented
in [docs/labels.md](docs/labels.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the three labels on a synthetic city and write the output table
python scripts/build_labels.py --city nyc --cells 400

# Create and summarise the spatial split for a city
python scripts/make_partition.py --city melbourne

# Run the checks
python -m pytest tests/
```

The scripts find the package from the repository root. `build_labels.py`
simulates the city, so its output exercises the plumbing and the label
formulas; replace the synthetic inputs with the provider data listed in
[docs/data_sources.md](docs/data_sources.md) to build the real tables.

## Data

Raw provider data is **not** distributed here. The censuses (TreesCount! 2015,
Les Arbres de Paris, Melbourne Urban Forest Visual), imagery and derived
products are obtained from their original providers under their own licences:
Sentinel-2 and Landsat are open (Copernicus, USGS public domain), ERA5-Land and
SMAP are free with attribution, SoilGrids and ESA WorldCover are CC-BY, OSM data
is ODbL, and GHSL is open. Per-variable sources, resolutions and derivations are
in [docs/data_sources.md](docs/data_sources.md); the processing rules, exclusion
criteria and partition scheme are in
[docs/processing.md](docs/processing.md).

## License

MIT — see [LICENSE](LICENSE). Provider data retains the licence of its source.
