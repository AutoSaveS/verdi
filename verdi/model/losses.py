"""Loss terms of the three stages.

Stage 1::

    L_S1 = ||V - V_hat||^2 + ||E - E_hat||^2 + L_dis + lambda_r * ||R* - H_psi(z)||^2
    L_dis = beta * KL(q(z) || N(0, I))
          + lambda_TC * KL(q(z) || prod_k q(z^k))
          + lambda_a * sum_{(i,j) in P_k} L_align^k(z_i, z_j)

Stage 2::

    L_MSM = E[ sum_{n in M_mask} ||x_n - x_hat_n||^2 ],  |M_mask| = floor(0.4 N_i)
    L_S2  = ||V_hat - V||^2 + ||E_hat - E||^2 + lambda_z * ||z_gt - z_sens||^2
            + lambda_r * ||R* - R_hat||^2

``GaussianNLL`` trains the predicted sigma under ``V_hat ~ N(mu_V, sigma_V)``
and can be switched off. The total-correlation term is the density-ratio
estimate of Chen et al. (2018).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_to_standard_normal(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL( q(z) || N(0, I) ) for a diagonal Gaussian q, summed over factors."""
    return 0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - 1.0 - log_var, dim=-1)


def total_correlation(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Density-ratio estimate of total correlation, per sample.

    ``log q(z) - sum_k log q(z_k)``, with q and q_k Gaussian. The log densities
    are computed under the aggregate posterior, so this is the standard
    minibatch estimate (Chen et al. 2018, Eq. 6).
    """
    # log q(z) under the minibatch aggregate posterior.
    log_qz = _log_density(z, mu, log_var, batch_mean=True)
    # sum_k log q(z_k) under the marginal of each factor.
    log_qz_factor = _log_density(z, mu, log_var, batch_mean=True, per_factor=True).sum(dim=-1)
    return log_qz - log_qz_factor


def _log_density(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
                 batch_mean: bool = True, per_factor: bool = False) -> torch.Tensor:
    """Log N(z; mu_i, var_i) for every i, summed over the latent axes."""
    z = z.unsqueeze(0)                      # 1 x B x K
    mu = mu.unsqueeze(1)                    # B x 1 x K
    log_var = log_var.unsqueeze(1)          # B x 1 x K
    var = torch.exp(log_var)
    log2pi = float(torch.log(torch.tensor(2.0 * torch.pi)))
    value = -0.5 * (log2pi + log_var + (z - mu) ** 2 / var)
    if per_factor:
        return value.squeeze(0)             # B x B x K
    return value.sum(dim=-1).squeeze(0)     # B x B


def weak_supervision_align(z: torch.Tensor, pairs_i: torch.Tensor,
                           pairs_j: torch.Tensor) -> torch.Tensor:
    """Weak-supervision alignment over spatial pairs that differ in one factor.

    ``pairs_i`` / ``pairs_j`` index rows of ``z``; the term pulls those
    embeddings together, which is the "weak supervision" component of L_dis.
    ``L_align^k`` is the squared distance.
    """
    if pairs_i.numel() == 0:
        return z.new_zeros(())
    return F.mse_loss(z[pairs_i], z[pairs_j])


class StageOneLoss(nn.Module):
    """L_S1: reconstruction, disentanglement and resilience terms."""

    def __init__(self, lambda_tc: float = 1.0, lambda_align: float = 0.5,
                 lambda_resilience: float = 1.0, beta: float = 1.0) -> None:
        super().__init__()
        self.lambda_tc = lambda_tc
        self.lambda_align = lambda_align
        self.lambda_resilience = lambda_resilience
        self.beta = beta

    def set_beta(self, beta: float) -> None:
        self.beta = float(beta)

    def forward(
        self,
        v: torch.Tensor,
        v_hat: torch.Tensor,
        e: torch.Tensor,
        e_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        z: torch.Tensor,
        r_star: Optional[torch.Tensor] = None,
        r_hat: Optional[torch.Tensor] = None,
        pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        reconstruction = F.mse_loss(v_hat, v) + F.mse_loss(e_hat, e)
        kl = kl_to_standard_normal(mu, log_var).mean()
        tc = total_correlation(z, mu, log_var).mean()
        align = weak_supervision_align(z, pairs[0], pairs[1]) if pairs else z.new_zeros(())
        disentanglement = self.beta * kl + self.lambda_tc * tc + self.lambda_align * align

        if r_star is not None and r_hat is not None:
            resilience = F.mse_loss(r_hat.squeeze(-1), r_star)
        else:
            resilience = z.new_zeros(())

        total = reconstruction + disentanglement + self.lambda_resilience * resilience
        parts = {
            "reconstruction": float(reconstruction.detach()),
            "kl": float(kl.detach()),
            "total_correlation": float(tc.detach()),
            "align": float(align.detach()),
            "resilience": float(resilience.detach()),
            "beta": self.beta,
        }
        return total, parts


def masked_sensor_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L_MSM over the masked positions."""
    if mask.sum() == 0:
        return x.new_zeros(())
    return F.mse_loss(x_hat[mask], x[mask])


def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a diagonal Gaussian (see module docstring)."""
    return F.gaussian_nll_loss(mu, target, torch.exp(log_var), reduction="mean")


class StageTwoLoss(nn.Module):
    """L_S2: state reconstruction, latent consistency and resilience fidelity."""

    def __init__(self, lambda_latent: float = 1.0, lambda_resilience: float = 1.0,
                 use_gaussian_nll: bool = True) -> None:
        super().__init__()
        self.lambda_latent = lambda_latent
        self.lambda_resilience = lambda_resilience
        self.use_gaussian_nll = use_gaussian_nll

    def forward(
        self,
        v: torch.Tensor,
        v_mu: torch.Tensor,
        v_log_var: torch.Tensor,
        e: torch.Tensor,
        e_mu: torch.Tensor,
        e_log_var: torch.Tensor,
        z_gt: Optional[torch.Tensor] = None,
        z_sens: Optional[torch.Tensor] = None,
        r_star: Optional[torch.Tensor] = None,
        r_hat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.use_gaussian_nll:
            state = gaussian_nll(v, v_mu, v_log_var) + gaussian_nll(e, e_mu, e_log_var)
        else:
            state = F.mse_loss(v_mu, v) + F.mse_loss(e_mu, e)

        if z_gt is not None and z_sens is not None:
            consistency = F.mse_loss(z_sens, z_gt)
        else:
            consistency = state.new_zeros(())

        if r_star is not None and r_hat is not None:
            resilience = F.mse_loss(r_hat.squeeze(-1), r_star)
        else:
            resilience = state.new_zeros(())

        total = state + self.lambda_latent * consistency + self.lambda_resilience * resilience
        return total, {
            "state": float(state.detach()),
            "latent_consistency": float(consistency.detach()),
            "resilience": float(resilience.detach()),
        }


class CrossStageConsistency(nn.Module):
    """L_cons of the end-to-end objective: align the two latent estimates.

    ``L_E2E = lambda_1 L_S1 + lambda_2 L_S2 + lambda_3 L_cons``, with the
    consistency weight exposed as ``eta``.
    """

    def __init__(self, eta: float = 1e-5) -> None:
        super().__init__()
        self.eta = eta

    def forward(self, z_world: torch.Tensor, z_sensor: torch.Tensor) -> torch.Tensor:
        return self.eta * F.mse_loss(z_sensor, z_world)


def end_to_end_loss(loss_s1: torch.Tensor, loss_s2: torch.Tensor,
                    loss_cons: torch.Tensor,
                    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> torch.Tensor:
    """lambda_1 L_S1 + lambda_2 L_S2 + lambda_3 L_cons with the 0.4/0.4/0.2 weights."""
    w1, w2, w3 = weights
    return w1 * loss_s1 + w2 * loss_s2 + w3 * loss_cons
