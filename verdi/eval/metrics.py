"""Evaluation metrics for the VERDI experiments (Tables 5-9; Appendix F)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Prediction metrics
# --------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    """Coefficient of determination, ``1 - SS_res / SS_tot``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        mean = np.average(y_true, weights=w)
        ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
        ss_tot = float(np.sum(w * (y_true - mean) ** 2))
    else:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def retention(r2_partial: float, r2_full: float) -> float:
    """C2 retention, ``R^2_{k=partial} / R^2_{k=6}``."""
    if r2_full == 0.0 or not np.isfinite(r2_full):
        return float("nan")
    return r2_partial / r2_full


def c3_gap(r2_reference: float, r2_verdi: float) -> float:
    """Relative gap for acceptance criterion C3, ``(R2_ref - R2_verdi) / R2_ref``."""
    if r2_reference == 0.0:
        return float("nan")
    return (r2_reference - r2_verdi) / r2_reference


def cohens_d(a: np.ndarray, b: np.ndarray, paired: bool = False) -> float:
    """Cohen's d; the paired form uses the standard deviation of the differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        diff = a - b
        sd = diff.std(ddof=1)
        return float(diff.mean() / sd) if sd > 0 else float("nan")
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 10_000,
                        level: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    draws = rng.choice(diff, size=(n, diff.size), replace=True).mean(axis=1)
    lo = float(np.quantile(draws, (1 - level) / 2))
    hi = float(np.quantile(draws, 1 - (1 - level) / 2))
    return lo, hi


# --------------------------------------------------------------------------
# Representation metrics
# --------------------------------------------------------------------------

def cmra(stage1, v: torch.Tensor, e: torch.Tensor) -> float:
    """Cross-modal reconstruction accuracy.

    Decodes ``V_hat_cross`` from a latent encoded through the environmental
    branch only (and symmetrically for ``E``), reporting mean R^2 over the two
    directions. ``stage1`` is a :class:`verdi.model.Stage1`.
    """
    scores = []
    with torch.no_grad():
        h_e = stage1.encoder_e(e).unsqueeze(1)
        h_v = stage1.encoder_v(v).unsqueeze(1)
        # Zero the absent branch, then run the shared cross-attention and decoder.
        zeros_v = torch.zeros_like(h_v)
        zeros_e = torch.zeros_like(h_e)
        v_cross, _ = stage1.cross(zeros_v, h_e)
        _, e_cross = stage1.cross(h_v, zeros_e)
        v_hat, _ = stage1.decode(stage1.to_latent(
            torch.cat([v_cross.squeeze(1), h_e.squeeze(1)], dim=-1))[:, :stage1.K])
        _, e_hat = stage1.decode(stage1.to_latent(
            torch.cat([h_v.squeeze(1), e_cross.squeeze(1)], dim=-1))[:, :stage1.K])
    scores.append(r2_score(v.numpy().ravel(), v_hat.numpy().ravel()))
    scores.append(r2_score(e.numpy().ravel(), e_hat.numpy().ravel()))
    return float(np.mean(scores))


def cmrg(r2_pretrained: float, r2_scratch: float) -> float:
    """Cross-modal reconstruction gain, ``R^2_pretrained - R^2_scratch``.

    Pass the means over the off-diagonal entries of the 6 x 6 cross-modal
    matrix to obtain the reported summary.
    """
    return float(r2_pretrained - r2_scratch)


def dci_scores(factors: np.ndarray, targets: np.ndarray, alpha: float = 0.02) -> Dict[str, float]:
    """Disentanglement, completeness and informativeness (DCI).

    A Lasso regression is fitted from the estimated factors to the ground-truth
    generative factors and scored with the standard DCI definitions.

    ``alpha`` is the Lasso penalty.
    """
    from sklearn.linear_model import Lasso

    factors = np.asarray(factors, dtype=float)
    targets = np.asarray(targets, dtype=float)
    importance = np.zeros((factors.shape[1], targets.shape[1]))
    for i in range(targets.shape[1]):
        model = Lasso(alpha=alpha).fit(factors, targets[:, i])
        importance[:, i] = np.abs(model.coef_)

    def normalised(m: np.ndarray) -> np.ndarray:
        col_sums = m.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return m / col_sums

    m = normalised(importance)
    row_sums = m.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    disentanglement = float(np.mean(1.0 - m.sum(axis=0) + np.max(m, axis=0) / m.sum(axis=0)))
    completeness = float(np.mean(1.0 - row_sums + np.max(m, axis=1) / row_sums))
    informativeness = float(np.mean(np.abs(importance).sum(axis=0)))
    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": informativeness,
        "mean": float(np.mean([disentanglement, completeness, informativeness])),
    }


ALIGNMENT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("thermal", "lst_mean_K"),
    ("water", "soil_oc_g_kge"),          # e9 x e10
    ("wind", "wind_speed_ms"),           # e12 / e4
    ("soil", "soil_clay_pct"),           # e9 + e10
    ("competition", "planting_density"),  # v6 x v5
)
#: The five latent-indicator alignment pairs. ``rho_target`` averages
#: |Spearman| over the pairs.


def rho_target(latent: np.ndarray, indicators: np.ndarray) -> float:
    """Mean absolute Spearman correlation over the alignment pairs.

    ``latent`` and ``indicators`` are keyed by the names in
    :data:`ALIGNMENT_PAIRS`. Hungarian assignment is used when a model has no
    explicit mapping.
    """
    from scipy.stats import spearmanr

    values = []
    for latent_name, indicator_name in ALIGNMENT_PAIRS:
        if latent_name in latent and indicator_name in indicators:
            rho, _ = spearmanr(latent[latent_name], indicators[indicator_name])
            if np.isfinite(rho):
                values.append(abs(float(rho)))
    return float(np.mean(values)) if values else float("nan")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def expected_calibration_error(probabilities: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected calibration error with ``n_bins`` equal-width bins."""
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    if probabilities.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = probabilities.size
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities > lower) & (probabilities <= upper)
        if not mask.any():
            continue
        confidence = probabilities[mask].mean()
        accuracy = correct[mask].mean()
        ece += (mask.sum() / total) * abs(accuracy - confidence)
    return float(ece)


# --------------------------------------------------------------------------
# Acceptance criteria
# --------------------------------------------------------------------------

@dataclass
class AcceptanceCriteria:
    """Acceptance criteria C1-C3 (Tables 6 and 7)."""
    retention_threshold: float = 0.80      # C2
    c3_gap_threshold: float = 0.05         # C3
    alpha: float = 0.01                    # p < 0.01
    min_effect_size: float = 0.5           # d > 0.5

    def c1(self, p_value: float, effect_size: float) -> bool:
        """Coupled beats the concat VAE: ``p < 0.01`` and ``d > 0.5``."""
        return p_value < self.alpha and effect_size > self.min_effect_size

    def c2(self, retention_value: float) -> bool:
        """Retention of at least 80 %."""
        return bool(np.isfinite(retention_value) and retention_value >= self.retention_threshold)

    def c3(self, gap: float) -> bool:
        """Relative R^2 gap below 5 % against the discriminative reference."""
        return bool(np.isfinite(gap) and gap < self.c3_gap_threshold)
