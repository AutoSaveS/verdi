"""The three VERDI stages (Section 3.3; Appendix G).

Stage 3 builds the diagnostic prompts; language-model inference and trained
weights are not part of this package.
"""

from .losses import (
    CrossStageConsistency,
    StageOneLoss,
    StageTwoLoss,
    end_to_end_loss,
    gaussian_nll,
    kl_to_standard_normal,
    masked_sensor_loss,
    total_correlation,
    weak_supervision_align,
)
from .stage1 import CrossAttention, Stage1
from .stage2 import FourierFeatures, Stage2, StalenessEmbedding, make_mask
from .stage3 import (
    LEVEL_TASKS,
    ClusterContext,
    Typology,
    build_prompt,
    compile_context,
    factor_attribution,
    fit_typology,
    reasoning_cascade,
    vulnerable_units,
)

__all__ = [
    "CrossAttention",
    "CrossStageConsistency",
    "ClusterContext",
    "FourierFeatures",
    "LEVEL_TASKS",
    "Stage1",
    "Stage2",
    "StageOneLoss",
    "StageTwoLoss",
    "StalenessEmbedding",
    "Typology",
    "build_prompt",
    "compile_context",
    "end_to_end_loss",
    "factor_attribution",
    "fit_typology",
    "gaussian_nll",
    "kl_to_standard_normal",
    "make_mask",
    "masked_sensor_loss",
    "reasoning_cascade",
    "total_correlation",
    "vulnerable_units",
    "weak_supervision_align",
]
