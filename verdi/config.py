"""Parameters of the VERDI workflow.

Values follow Section 3 and Appendices A and E of the accompanying paper.
Parameters that have no fixed value in the paper are exposed here with a
default so that each run can record the setting it used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# --------------------------------------------------------------------------
# Proxy labels (Section 3.1; Appendix A.3)
# --------------------------------------------------------------------------

#: R*_A composite weights P_i = 0.4 h_i + 0.4 n_i + 0.2 d_i.
R_A_WEIGHTS: Dict[str, float] = {"health": 0.4, "ndvi_pct": 0.4, "dbh_pct": 0.2}

#: Minimum number of conspecific individuals for a species to be ranked.
R_A_MIN_SPECIES_N = 500

#: Census health grades harmonised to [0, 1] (Table A.17).
#: NYC: Good/Fair/Poor. Melbourne: ULE classes. Paris: no health field.
HEALTH_GRADES: Dict[str, Dict[str, float]] = {
    "nyc": {"Good": 1.0, "Fair": 0.6, "Poor": 0.2},
    "melbourne": {">20yr": 1.0, "10-20yr": 0.6},
    # The Melbourne ULE < 10 yr class has no grade; such trees raise KeyError.
    "paris": {},  # R*_A is unavailable: no census health field.
}

#: R*_B heat-event definition and NDVI retention window.
R_B_EVENT_MIN_DAYS = 3          # >=3 consecutive days above the percentile
R_B_EVENT_PERCENTILE = 95.0     # city-specific 95th percentile of T_max
R_B_ERA5_YEARS = (2018, 2023)   # ERA5-Land record used for the events
R_B_NDVI_WINDOW_DAYS = 14       # pre/post Sentinel-2 pairs within 14 days
R_B_CLOUD_MAX_PCT = 10.0        # cloud < 10 %
R_B_NDVI_FLOOR = 0.2            # denominator floor in Eq. A.1
R_B_CLIP = (0.0, 1.0)           # "clipped to [0, 1]"

#: R*_C species-baseline deviation (Eq. A.2).
#: Statistic used for the species baseline: "median" or "mean".
R_C_BASELINE_STATISTIC = "median"
#: Apply an additional min-max rescaling to [0, 1] after the sigmoid.
R_C_APPLY_MINMAX = False

#: Label fusion (Appendix A.4; Table A.21).
#: R*_C is the primary target; it is available in all three cities.
PRIMARY_LABEL = "R_C"

# --------------------------------------------------------------------------
# Study areas and grid (Section 3.1, Table 3)
# --------------------------------------------------------------------------

#: Grid cell size in metres, and the sizes used for the resolution ablation.
DELTA_M = 100
DELTA_ABLATION_M = (50, 75, 100, 150, 200)

#: Metropolitan extents (Table 3) and projected CRS.
STUDY_AREAS: Dict[str, Dict[str, object]] = {
    "nyc": {"centre": (40.71, -74.01), "koppen": "Cfa", "extent_km": (46.3, 53.1),
            "area_km2": 2458, "crs": "EPSG:32618", "summer": "Jun-Aug"},
    "paris": {"centre": (48.86, 2.35), "koppen": "Cfb", "extent_km": (54.2, 49.7),
              "area_km2": 2694, "crs": "EPSG:32631", "summer": "Jun-Aug"},
    "melbourne": {"centre": (-37.81, 144.96), "koppen": "Cfb", "extent_km": (50.2, 53.1),
                  "area_km2": 2666, "crs": "EPSG:32755", "summer": "Dec-Feb"},
}

#: Cell exclusion rules (Appendix A.5).
EXCLUDE_WATER_FRACTION = 0.80          # water-body fraction > 80 %
EXCLUDE_NDVI_IF_NO_TREES = 0.15        # n_trees = 0 and NDVI < 0.15
EXCLUDE_MIN_COMPLETENESS = 0.70        # data completeness < 70 %
OUTLIER_BOUNDS = {"ndvi": (0.0, 1.0), "dbh_cm": (0.0, 200.0),
                  "building_height_m": (0.0, 300.0)}

# --------------------------------------------------------------------------
# Spatial partitioning (Appendix A.5)
# --------------------------------------------------------------------------

SUPERBLOCK_CELLS = 5                  # 5Δ x 5Δ super-blocks
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / validation / test
CROSS_CITY_HOLDOUT = 0.10             # 10 % of the training partition
#: Seed of the block assignment, fixed for reproducible splits.
PARTITION_SEED = 20260101

# --------------------------------------------------------------------------
# Model (Section 3.2; Appendix E, Table E.31)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage1Config:
    """Stage 1, Spatial World Model (Section 3.2.1, Table E.31)."""
    hidden: Tuple[int, int] = (128, 128)     # d -> 128 -> 128, LayerNorm, ReLU
    attn_layers: int = 2                     # "2 layers, 4 heads, d_model=128"
    attn_heads: int = 4
    attn_dim: int = 128
    latent_dim: int = 6                      # K = 6 named factors
    decoder_hidden: Tuple[int, int] = (128, 256)
    head_hidden: Tuple[int, int] = (64, 32)  # H_psi: K -> 64 -> 32 -> 1
    beta_start: float = 1.0                  # beta-anneal 1 -> 4
    beta_end: float = 4.0
    lambda_tc: float = 1.0                   # "TC lambda = 1"
    lambda_align: float = 0.5                # "weak sup. lambda_a = 0.5"
    #: Weight of the resilience-prediction term of L_S1.
    lambda_resilience: float = 1.0


#: Names of the K = 6 latent factors.
LATENT_FACTORS = ("thermal", "water", "wind", "soil", "competition", "residual")


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2, Masked Sensor Transformer (Section 3.2.2, Table E.31)."""
    d_model: int = 256                       # modality projection -> d = 256
    layers: int = 6                          # "6 layers, 8 heads, FFN 1024"
    heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    mask_ratio: float = 0.4                  # |M_mask| = floor(0.4 N_i)
    query_tokens: int = 3                    # q_V, q_E, q_R
    #: Number of Fourier-feature bands and staleness half-life in days.
    fourier_bands: int = 8
    staleness_half_life_days: float = 5.0
    #: Train the predicted sigma with a Gaussian negative log-likelihood on
    #: the reconstructed state, V_i ~ N(mu_V, sigma_V).
    use_gaussian_nll: bool = True


@dataclass(frozen=True)
class Stage3Config:
    """Stage 3, diagnostic reasoning (Section 3.2.3, Table E.31)."""
    n_clusters: int = 0                      # 0 = select C by BIC
    covariance_type: str = "full"
    #: Healthy-baseline threshold tau and vulnerability threshold tau_vuln.
    tau_baseline: float = 0.5
    tau_vuln: float = 0.4
    reasoning_levels: int = 3                # r1, r2, r3
    #: Fine-tuning configuration of the Stage 3 language model.
    base_model: str = "LLaMA-3-8B"
    lora_rank: int = 16                      # r = 16, alpha = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str = "4-bit GPTQ (~6 GB VRAM)"


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings (Section 3.3, Table E.32)."""
    stage1_epochs: int = 150
    stage1_beta_anneal_epochs: int = 50
    stage2_pretrain_epochs: int = 100
    stage2_finetune_epochs: int = 50
    stage3_epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-4                         # AdamW, lr = 1e-4
    weight_decay: float = 1e-2
    early_stop_patience: int = 20
    seeds: int = 5                           # five random seeds
    loss_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)  # lambda1..3
    #: Weights of the consistency term (eta) and of the latent term of L_S2.
    lambda_consistency: float = 1e-5
    lambda_latent: float = 1.0
    device: str = "cpu"


@dataclass(frozen=True)
class MetricsConfig:
    """Evaluation settings (Tables 4-8; Appendix F)."""
    ece_bins: int = 10                       # equal-width bins for ECE
    bootstrap_n: int = 10_000                # paired bootstrap resamples
    alpha_adjusted: float = 0.0033           # Bonferroni-adjusted alpha
    ci_level: float = 0.95
    retention_threshold: float = 0.80        # C2 >= 80 %
    c3_gap_threshold: float = 0.05           # Delta R^2 < 5 %
    deepeval_bertscore_model: str = "deberta-xlarge-mnli"


#: Convenience bundle, so a script can pass one object around.
@dataclass(frozen=True)
class Config:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


CONFIG = Config()
