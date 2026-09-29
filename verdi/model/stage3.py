"""Stage 3, diagnostic reasoning: factor attribution, typology, prompt building.

Section 3.3.3:

* attribution ``Phi_i = z_i - z_base(s_i)``;
* a Gaussian-mixture typology over ``Phi_i`` for units below a vulnerability
  threshold;
* a three-level prompt cascade::

      Prompt_l = Evidence(D_c) (+) History(r_<l) (+) Instruction_l

  with the cluster context ``D_c``.

The module returns the prompts. The language model (LLaMA-3-8B with LoRA,
r = 16, alpha = 32, 4-bit quantization) is configured in ``Stage3Config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import LATENT_FACTORS, Stage3Config


def factor_attribution(z: torch.Tensor, z_base: torch.Tensor) -> torch.Tensor:
    """Factor attribution ``Phi_i = z_i - z_base(s_i)``."""
    return z - z_base


def vulnerable_units(r_hat: torch.Tensor, tau_vuln: float = 0.4) -> torch.Tensor:
    """Indices of units below the vulnerability threshold ``tau_vuln``."""
    return torch.nonzero(r_hat.squeeze(-1) < tau_vuln, as_tuple=False).squeeze(-1)


@dataclass
class Typology:
    """A fitted Gaussian mixture over the attribution vectors."""
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    n_components: int
    covariance_type: str

    def density(self, phi: np.ndarray) -> np.ndarray:
        """Per-component weighted density, shape ``(n_units, n_components)``."""
        from scipy.stats import multivariate_normal

        values = np.asarray(phi, dtype=float)
        scores = []
        for c in range(self.n_components):
            covariance = self.covariances[c]
            scores.append(
                self.weights[c]
                * multivariate_normal.pdf(values, mean=self.means[c], cov=covariance)
            )
        return np.stack(scores, axis=-1)

    def assign(self, phi: np.ndarray) -> np.ndarray:
        """Hard cluster assignment by the component with the highest density."""
        return np.argmax(self.density(phi), axis=-1)

    def extent(self, labels: np.ndarray,
               coordinates: Optional[np.ndarray] = None) -> Dict[int, Dict[str, float]]:
        """``extent(c)`` of the context ``D_c``.

        Reported as the number of units per cluster and, when coordinates are
        supplied, the larger side of the bounding box in metres.
        """
        out: Dict[int, Dict[str, float]] = {}
        for c in range(self.n_components):
            mask = labels == c
            entry = {"n_units": float(mask.sum())}
            if coordinates is not None and mask.any():
                xy = np.asarray(coordinates)[mask]
                entry["extent_m"] = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
            out[c] = entry
        return out


def fit_typology(phi, config: Stage3Config = Stage3Config(), seed: int = 0) -> Typology:
    """Fit the GMM typology.

    ``n_clusters = 0`` selects the number of components by BIC over
    ``range(2, 10)``.
    """
    from sklearn.mixture import GaussianMixture

    values = phi.detach().cpu().numpy() if isinstance(phi, torch.Tensor) else np.asarray(phi)
    if values.ndim != 2:
        raise ValueError("phi must be a 2-D array of attribution vectors")

    n_components = config.n_clusters
    if n_components <= 0:
        best, best_bic = None, np.inf
        for k in range(2, min(10, len(values))):
            model = GaussianMixture(n_components=k, covariance_type=config.covariance_type,
                                    random_state=seed).fit(values)
            bic = model.bic(values)
            if bic < best_bic:
                best, best_bic = model, bic
        if best is None:
            raise ValueError("not enough units to fit a typology")
        model = best
    else:
        model = GaussianMixture(n_components=n_components,
                                covariance_type=config.covariance_type,
                                random_state=seed).fit(values)

    return Typology(
        weights=model.weights_,
        means=model.means_,
        covariances=model.covariances_,
        n_components=model.n_components,
        covariance_type=config.covariance_type,
    )


@dataclass
class ClusterContext:
    """Context ``D_c`` for one cluster."""
    cluster: int
    weight: float
    mean: Sequence[float]
    covariance: Sequence[Sequence[float]]
    extent: Dict[str, float]
    units: List[Dict[str, float]] = field(default_factory=list)


def compile_context(
    typology: Typology,
    cluster: int,
    phi,
    r_hat,
    sigma,
    species: Sequence[str],
    v,
    e,
    coordinates: Optional[np.ndarray] = None,
    max_units: int = 20,
) -> ClusterContext:
    """Assemble ``D_c = {mu_c, Sigma_c, pi_c, extent(c), {R_i, Phi_i, sigma_i, s_i, V_i, E_i}}``.

    ``max_units`` caps the per-unit listing so a prompt stays a reasonable
    size; the units kept are the most vulnerable in the cluster.
    """
    def _array(value):
        # Flatten trailing singleton dimensions (r_hat and sigma arrive as
        # (N, 1)) so that indexing below yields scalars, not length-1 arrays.
        return np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value,
                          dtype=float).reshape(-1)

    phi_np = np.asarray(phi.detach().cpu() if isinstance(phi, torch.Tensor) else phi, dtype=float)
    r_np = _array(r_hat)
    sigma_np = _array(sigma)
    v_np = np.asarray(v.detach().cpu() if isinstance(v, torch.Tensor) else v, dtype=float)
    e_np = np.asarray(e.detach().cpu() if isinstance(e, torch.Tensor) else e, dtype=float)

    labels = typology.assign(phi_np)
    idx = np.flatnonzero(labels == cluster)
    if idx.size:
        idx = idx[np.argsort(r_np[idx].reshape(-1))][:max_units]

    units = [
        {
            "index": int(i),
            "R": float(r_np[i]),
            "sigma": float(sigma_np[i]),
            "species": species[int(i)],
            "phi": phi_np[i].tolist(),
            "V": v_np[i].tolist(),
            "E": e_np[i].tolist(),
        }
        for i in idx
    ]
    extent = typology.extent(labels, coordinates)[cluster]
    return ClusterContext(
        cluster=cluster,
        weight=float(typology.weights[cluster]),
        mean=np.asarray(typology.means[cluster]).tolist(),
        covariance=np.asarray(typology.covariances[cluster]).tolist(),
        extent=extent,
        units=units,
    )


LEVEL_TASKS: Dict[int, str] = {
    1: "Explain the dominant stress factors for this cluster, conditional on the evidence above.",
    2: "Synthesise the spatial pattern of the cluster across the units listed above.",
    3: "Propose a management action for this cluster and state what would need to be verified first.",
}
#: Instructions for levels r1 (explanation), r2 (synthesis) and r3
#: (management suggestion).


def build_prompt(context: ClusterContext, level: int,
                 history: Optional[Sequence[str]] = None) -> str:
    """``Prompt_l = Evidence(D_c) (+) History(r_<l) (+) Instruction_l``.

    The output is a plain string so it can be inspected, logged, or handed to
    any model; nothing here calls one.
    """
    if level not in LEVEL_TASKS:
        raise ValueError(f"level must be one of {sorted(LEVEL_TASKS)}")

    factor_names = ", ".join(LATENT_FACTORS)
    lines = [
        f"Cluster {context.cluster} (weight {context.weight:.2f}); extent {context.extent}.",
        f"Attribution mean over the {len(LATENT_FACTORS)} factors [{factor_names}]: "
        + ", ".join(f"{value:+.2f}" for value in context.mean),
    ]
    for unit in context.units:
        lines.append(
            f"- unit {unit['index']}: R={unit['R']:.2f}, sigma={unit['sigma']:.2f}, "
            f"species={unit['species']}, attribution="
            + ", ".join(f"{value:+.2f}" for value in unit["phi"])
        )

    parts = ["Evidence:\n" + "\n".join(lines)]
    if history:
        parts.append("Prior reasoning:\n" + "\n".join(history))
    parts.append("Task:\n" + LEVEL_TASKS[level])
    parts.append(
        "Report associations supported by the evidence above. These are "
        "decision-support suggestions, not causal diagnoses or validated "
        "interventions."
    )
    return "\n\n".join(parts)


def reasoning_cascade(context: ClusterContext, levels: int = 3) -> List[Dict[str, str]]:
    """Build the r1..r3 prompts, passing earlier outputs as history.

    Returns ``{"level", "prompt"}`` records. The responses themselves are
    produced by the language model outside this function.
    """
    out: List[Dict[str, str]] = []
    history: List[str] = []
    for level in range(1, levels + 1):
        out.append({"level": str(level), "prompt": build_prompt(context, level,
                                                               history=history or None)})
        history.append(f"(r{level} response omitted: no model is called here)")
    return out
