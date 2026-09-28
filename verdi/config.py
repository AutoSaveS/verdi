"""Parameters of the VERDI workflow, each with the place it comes from.

Every entry carries a ``source`` string pointing at the manuscript (section,
table, equation or appendix) and, where the manuscript does not fix a value,
``implemented`` states the choice this repository makes. Those choices are
marked ``IMPLEMENTATION CHOICE`` on purpose: they are needed to run the code
and are *not* manuscript parameters. Where the manuscript is internally
inconsistent the two readings are both recorded and the conflict is noted.

Manuscript: main1.tex of the UFUG revision (line numbers refer to that file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# --------------------------------------------------------------------------
# Proxy labels (Section 3.2; Appendix A.5, main1.tex L2107-L2150)
# --------------------------------------------------------------------------

#: R*_A composite weights P_i = 0.4 h_i + 0.4 n_i + 0.2 d_i (L2132).
R_A_WEIGHTS: Dict[str, float] = {"health": 0.4, "ndvi_pct": 0.4, "dbh_pct": 0.2}

#: Minimum number of conspecific individuals for a species to be ranked (L2132).
R_A_MIN_SPECIES_N = 500

#: Census health grades harmonised to [0, 1] (Table A.19, L1961).
#: NYC: Good/Fair/Poor. Melbourne: ULE classes. Paris: no health field.
HEALTH_GRADES: Dict[str, Dict[str, float]] = {
    "nyc": {"Good": 1.0, "Fair": 0.6, "Poor": 0.2},
    "melbourne": {">20yr": 1.0, "10-20yr": 0.6},
    # "melbourne": {"<10yr": ???}  # IMPLEMENTATION CHOICE: the manuscript
    # fixes ULE >20 yr and 10-20 yr only (L1961); the <10 yr class is not
    # given a value.
    "paris": {},  # R*_A is unavailable: no census health field (L2132).
}

#: R*_B heat-event definition and NDVI retention window (L2136-L2141).
R_B_EVENT_MIN_DAYS = 3          # >=3 consecutive days above the percentile
R_B_EVENT_PERCENTILE = 95.0     # city-specific 95th percentile of T_max
R_B_ERA5_YEARS = (2018, 2023)   # ERA5-Land record used for the events
R_B_NDVI_WINDOW_DAYS = 14       # pre/post Sentinel-2 pairs within 14 days
R_B_CLOUD_MAX_PCT = 10.0        # cloud < 10 %
R_B_NDVI_FLOOR = 0.2            # denominator floor in Eq. A.1
R_B_CLIP = (0.0, 1.0)           # "clipped to [0, 1]"

#: R*_C species-baseline deviation (Eq. A.2, L2145-L2150).
#: IMPLEMENTATION CHOICE: the manuscript writes the species baseline only as
#: the tilde notation \widetilde{NDVI}(s) with \sigma_NDVI(s) and describes it
#: as a "species-level baseline"; median is not stated. We expose the choice
#: so a run can record which statistic it used.
R_C_BASELINE_STATISTIC = "median"   # IMPLEMENTATION CHOICE ("median" | "mean")
#: L2673 says R*_C is min-max standardised to [0, 1]; Eq. A.2 says sigmoid.
#: CONFLICT in the manuscript. The label definition (Eq. A.2) is the one used
#: here; set True to apply the additional min-max step of L2673.
R_C_APPLY_MINMAX = False

#: Label fusion (Table A.25, L2158-L2176; Table B.26, L2204-L2225).
#: R*_C is the complete primary target of the main reported tables (L2178);
#: the other combinations are design descriptions only.
PRIMARY_LABEL = "R_C"

# --------------------------------------------------------------------------
# Study areas and grid (Section 3.1, Table 2, Appendix A.1/A.2)
# --------------------------------------------------------------------------

#: Grid cell size in metres (L670: "~2,500 km2 at 100 m resolution";
#: ablated over {50, 75, 100, 150, 200} m at L1998).
DELTA_M = 100
DELTA_ABLATION_M = (50, 75, 100, 150, 200)

#: Metropolitan extents and CRS (Table A.18, L1919-L1943).
STUDY_AREAS: Dict[str, Dict[str, object]] = {
    "nyc": {"centre": (40.71, -74.01), "koppen": "Cfa", "extent_km": (46.3, 53.1),
            "area_km2": 2458, "crs": "EPSG:32618", "summer": "Jun-Aug"},
    "paris": {"centre": (48.86, 2.35), "koppen": "Cfb", "extent_km": (54.2, 49.7),
              "area_km2": 2694, "crs": "EPSG:32631", "summer": "Jun-Aug"},
    "melbourne": {"centre": (-37.81, 144.96), "koppen": "Cfb", "extent_km": (50.2, 53.1),
                  "area_km2": 2666, "crs": "EPSG:32755", "summer": "Dec-Feb"},
}

#: Cell exclusion rules (L2188).
EXCLUDE_WATER_FRACTION = 0.80          # water-body fraction > 80 %
EXCLUDE_NDVI_IF_NO_TREES = 0.15        # n_trees = 0 and NDVI < 0.15
EXCLUDE_MIN_COMPLETENESS = 0.70        # data completeness < 70 %
OUTLIER_BOUNDS = {"ndvi": (0.0, 1.0), "dbh_cm": (0.0, 200.0),
                  "building_height_m": (0.0, 300.0)}

# --------------------------------------------------------------------------
# Spatial partitioning (L2192, L774)
# --------------------------------------------------------------------------

SUPERBLOCK_CELLS = 5                  # 5Δ x 5Δ super-blocks
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation / test
MIN_SEPARATION_CELLS = 4              # minimum 4Δ inter-partition separation
CROSS_CITY_HOLDOUT = 0.10             # 10 % per city, "design description only"
#: IMPLEMENTATION CHOICE: the manuscript does not give the block-assignment
#: method or its seed; the seed is exposed so runs are reproducible.
PARTITION_SEED = 20260101

# --------------------------------------------------------------------------
# Model reference implementation (Section 3.3; Appendix G, Table G.40)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage1Config:
    """Stage 1, Spatial World Model (L804-L859, Table G.40 L2723-L2727)."""
    hidden: Tuple[int, int] = (128, 128)     # d -> 128 -> 128, LayerNorm, ReLU
    attn_layers: int = 2                     # "2 layers, 4 heads, d_model=128"
    attn_heads: int = 4
    attn_dim: int = 128
    latent_dim: int = 6                      # K = 6 named factors (L831)
    decoder_hidden: Tuple[int, int] = (128, 256)
    head_hidden: Tuple[int, int] = (64, 32)  # H_psi: K -> 64 -> 32 -> 1
    beta_start: float = 1.0                  # beta-anneal 1 -> 4 (L2341)
    beta_end: float = 4.0
    lambda_tc: float = 1.0                   # "TC lambda = 1"
    lambda_align: float = 0.5                # "weak sup. lambda_a = 0.5"
    #: IMPLEMENTATION CHOICE: lambda_r of L_S1 (L858) is never given.
    lambda_resilience: float = 1.0


#: Names of the K = 6 latent factors (L831).
LATENT_FACTORS = ("thermal", "water", "wind", "soil", "competition", "residual")


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2, Masked Sensor Transformer (L862-L923, Table G.40)."""
    d_model: int = 256                       # modality projection -> d = 256
    layers: int = 6                          # "6 layers, 8 heads, FFN 1024"
    heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    mask_ratio: float = 0.4                  # |M_mask| = floor(0.4 N_i)
    query_tokens: int = 3                    # q_V, q_E, q_R
    #: IMPLEMENTATION CHOICE: Fourier-feature frequencies and the staleness
    #: embedding time unit are not specified (L878).
    fourier_bands: int = 8
    staleness_half_life_days: float = 5.0
    #: IMPLEMENTATION CHOICE: L_S2 (L920) has no likelihood term, so how the
    #: reported sigma is trained is not specified. We use a Gaussian NLL on
    #: the reconstructed state, which is the standard reading of
    #: "V_i ~ N(mu_V, sigma_V)" (L893).
    use_gaussian_nll: bool = True


@dataclass(frozen=True)
class Stage3Config:
    """Stage 3, diagnostic reasoning (L926-L979, Table G.40 L2735-L2738)."""
    n_clusters: int = 0                      # IMPLEMENTATION CHOICE: C is not
    #   specified; the case studies show 4/3/2 clusters (L1834). 0 = fit by BIC.
    covariance_type: str = "full"
    #: IMPLEMENTATION CHOICE: tau (baseline threshold) and tau_vuln (vulnerable
    #: units) are never given (L938, L946).
    tau_baseline: float = 0.5
    tau_vuln: float = 0.4
    reasoning_levels: int = 3                # r1, r2, r3 (L979)
    #: Documented fine-tuning configuration. No weights and no LLM call are
    #: included in this repository.
    base_model: str = "LLaMA-3-8B"           # L1849, L2551, L2736
    lora_rank: int = 16                      # r = 16, alpha = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str = "4-bit GPTQ (~6 GB VRAM)"


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings (L985, L993-L996, Table 10 L1434-L1438)."""
    stage1_epochs: int = 150                 # Table 10 (L985 says beta-anneal
    #   "over 50 epochs": CONFLICT, recorded but not used as the epoch count).
    stage1_beta_anneal_epochs: int = 50
    stage2_pretrain_epochs: int = 100
    stage2_finetune_epochs: int = 50
    stage3_epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-4                         # AdamW, lr = 1e-4
    weight_decay: float = 1e-2
    early_stop_patience: int = 20
    seeds: int = 5                           # five seeds (L2770)
    loss_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)  # lambda1..3
    #: IMPLEMENTATION CHOICE: eta of L_cons (L996) is introduced but never
    #: defined; lambda_z of L_S2 (L920) is likewise unspecified.
    lambda_consistency: float = 1e-5
    lambda_latent: float = 1.0
    device: str = "cpu"


@dataclass(frozen=True)
class MetricsConfig:
    """Evaluation settings (Tables 5-9; Appendix H L2763-L2776)."""
    ece_bins: int = 10                       # IMPLEMENTATION CHOICE: the ECE
    #   name and target (< 0.10) are given (L1189, L2520) but no binning.
    bootstrap_n: int = 10_000                # paired bootstrap (L1298)
    alpha_adjusted: float = 0.0033           # Bonferroni (L2474)
    ci_level: float = 0.95
    retention_threshold: float = 0.80        # C2 >= 80 %
    c3_gap_threshold: float = 0.05           # Delta R^2 < 5 %
    deepeval_bertscore_model: str = "deberta-xlarge-mnli"  # L2534


#: Convenience bundle, so a script can pass one object around.
@dataclass(frozen=True)
class Config:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


CONFIG = Config()
