"""Stage 1, Spatial World Model (Section 3.2.1; Table E.30):

* dual encoders ``F_V`` and ``F_E``, ``d -> 128 -> 128`` with LayerNorm and ReLU;
* bidirectional cross-attention, 2 layers, 4 heads, ``d_model = 128``;
* ``G_phi: 256 -> 2K`` emitting ``mu`` and ``log sigma^2`` with ``K = 6``;
* a decoder ``K -> 128 -> 256 -> (d_V + d_E)``;
* ``H_psi: K -> 64 -> 32 -> 1`` with a sigmoid output.

The six latent factors are thermal stress, water stress, wind exposure, soil
quality, competition and a residual term. The resilience head ``H_psi`` is a
lightweight MLP trained jointly with the encoder.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import LATENT_FACTORS, Stage1Config


def mlp(sizes, norm: bool = True, activation: bool = True) -> nn.Sequential:
    """Feed-forward stack with optional LayerNorm and ReLU on hidden layers."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        last = i == len(sizes) - 2
        if not last and norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if not last and activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CrossAttention(nn.Module):
    """Bidirectional cross-attention block (2 layers, 4 heads, d_model = 128)."""

    def __init__(self, dim: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.v_to_e = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
             for _ in range(layers)]
        )
        self.e_to_v = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
             for _ in range(layers)]
        )
        self.norm_v = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.norm_e = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])

    def forward(self, h_v: torch.Tensor, h_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i in range(len(self.v_to_e)):
            v_attended, _ = self.v_to_e[i](h_v, h_e, h_e, need_weights=False)
            h_v = self.norm_v[i](h_v + v_attended)
            e_attended, _ = self.e_to_v[i](h_e, h_v, h_v, need_weights=False)
            h_e = self.norm_e[i](h_e + e_attended)
        return h_v, h_e


class Stage1(nn.Module):
    """Coupled vegetation-environment world model."""

    def __init__(self, d_v: int, d_e: int, config: Stage1Config = Stage1Config()) -> None:
        super().__init__()
        self.config = config
        self.d_v = d_v
        self.d_e = d_e
        self.K = config.latent_dim

        self.encoder_v = mlp([d_v, *config.hidden])
        self.encoder_e = mlp([d_e, *config.hidden])
        self.cross = CrossAttention(dim=config.attn_dim, heads=config.attn_heads,
                                    layers=config.attn_layers)
        merged = 2 * config.attn_dim
        self.to_latent = nn.Linear(merged, 2 * self.K)
        self.decoder = mlp([self.K, *config.decoder_hidden, d_v + d_e])
        self.resilience_head = nn.Sequential(
            mlp([self.K, *config.head_hidden, 1]),
            nn.Sigmoid(),
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def encode(self, v: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_v = self.encoder_v(v).unsqueeze(1)
        h_e = self.encoder_e(e).unsqueeze(1)
        h_v, h_e = self.cross(h_v, h_e)
        merged = torch.cat([h_v.squeeze(1), h_e.squeeze(1)], dim=-1)
        mu, log_var = self.to_latent(merged).chunk(2, dim=-1)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(z)
        return out[:, :self.d_v], out[:, self.d_v:]

    def resilience(self, z: torch.Tensor) -> torch.Tensor:
        return self.resilience_head(z)

    def forward(self, v: torch.Tensor, e: torch.Tensor,
                sample: bool = True) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encode(v, e)
        z = self.reparameterise(mu, log_var) if sample else mu
        v_hat, e_hat = self.decode(z)
        return {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "v_hat": v_hat,
            "e_hat": e_hat,
            "r_hat": self.resilience(z),
        }

    def latent_dict(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Named latent factors, for reporting and the typology step."""
        return {name: z[:, i] for i, name in enumerate(LATENT_FACTORS)}

    def z_base(self, z: torch.Tensor, r_hat: torch.Tensor, species: torch.Tensor,
               tau: float = 0.5) -> torch.Tensor:
        """Per-species baseline ``z_base(s) = E[z | H_psi(z) > tau, species = s]`` over the batch."""
        out = torch.zeros_like(z)
        for s in torch.unique(species):
            mask = (species == s) & (r_hat.squeeze(-1) > tau)
            if mask.any():
                out[species == s] = z[mask].mean(dim=0, keepdim=True)
            else:
                out[species == s] = z[species == s].mean(dim=0, keepdim=True)
        return out
