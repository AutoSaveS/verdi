"""Stage 2, Masked Sensor Transformer (Section 3.2.2; Table E.31):

* token assembly::

      T_i^(t) = (+)_{n in M_i} ( W_m(x_n) + e_m + gamma(x_n^loc, y_n^loc) + tau(t_n - t_ref) )

  with ``W_m`` a modality-specific projection to ``d = 256``, ``e_m`` a
  learnable modality embedding, ``gamma`` Fourier features of the location and
  ``tau`` a sinusoidal staleness embedding; ``(+)`` concatenates
  variable-length sequences with no padding and no imputation.

* a Transformer with 6 layers, 8 heads, FFN width 1024 and dropout 0.1,
  attending over three learnable queries ``q_V``, ``q_E`` and ``q_R``;
* Gaussian state heads, then the frozen Stage 1 encoder::

      z_i = G_phi(V_hat_i, E_hat_i),  R_hat_i = H_psi(z_i),  Phi_i = z_i - z_base(s_i)

* masked-sensor pretraining with ``|M_mask| = floor(0.4 N_i)``.

Optional text metadata enters as precomputed embeddings from a frozen
language-model encoder, declared through ``text_embed_dim``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import Stage2Config


class FourierFeatures(nn.Module):
    """``gamma(x, y)``: sinusoidal features of the cell coordinates.

    ``bands`` geometric frequencies over the normalised coordinate, followed
    by a linear projection so the features can be added to the
    ``d_model``-wide modality tokens.
    """

    def __init__(self, bands: int = 8, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(bands, dtype=torch.float32)
        )
        self.raw_dim = 2 * 2 * bands
        self.proj = nn.Linear(self.raw_dim, out_dim) if out_dim is not None else None

    @property
    def out_dim(self) -> int:
        return self.proj.out_features if self.proj is not None else self.raw_dim

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        scaled = xy.unsqueeze(-1) * self.frequencies  # B x 2 x bands
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(start_dim=1)
        return self.proj(features) if self.proj is not None else features


class StalenessEmbedding(nn.Module):
    """``tau(t - t_ref)``: sinusoidal embedding of the observation age.

    The time scale is a configurable half-life in days.
    """

    def __init__(self, dim: int, half_life_days: float = 5.0) -> None:
        super().__init__()
        self.dim = dim
        self.half_life_days = float(half_life_days)
        self.proj = nn.Linear(1, dim)

    def forward(self, age_days: torch.Tensor) -> torch.Tensor:
        decayed = torch.exp(-age_days / self.half_life_days).unsqueeze(-1)
        return torch.sin(self.proj(decayed))


class ModalityProjection(nn.Module):
    """``W_m`` plus the learnable modality embedding ``e_m`` (per modality)."""

    def __init__(self, modality_dims: Dict[str, int], d_model: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, d_model) for name, dim in modality_dims.items()}
        )
        self.embeddings = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(d_model)) for name in modality_dims}
        )
        self.names: List[str] = list(modality_dims)

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self.projections:
            raise KeyError(f"unknown modality {name!r}; expected one of {self.names}")
        return self.projections[name](x) + self.embeddings[name]


class Stage2(nn.Module):
    """Masked sensor transformer over heterogeneous, incomplete observations."""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        d_v: int,
        d_e: int,
        text_embed_dim: Optional[int] = None,
        config: Stage2Config = Stage2Config(),
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_v = d_v
        self.d_e = d_e

        self.modalities = ModalityProjection(modality_dims, config.d_model)
        self.location = FourierFeatures(config.fourier_bands, out_dim=config.d_model)
        self.staleness = StalenessEmbedding(config.d_model, config.staleness_half_life_days)

        self.token_norm = nn.LayerNorm(config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)

        # Learnable state queries, plus the per-token decode head.
        self.queries = nn.Parameter(torch.randn(config.query_tokens, config.d_model) * 0.02)
        self.token_head = nn.Linear(config.d_model, config.d_model)

        self.v_mu = nn.Linear(config.d_model, d_v)
        self.v_log_var = nn.Linear(config.d_model, d_v)
        self.e_mu = nn.Linear(config.d_model, d_e)
        self.e_log_var = nn.Linear(config.d_model, d_e)
        self.r_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())

        self.text_embed_dim = text_embed_dim
        if text_embed_dim is not None:
            # Accepts precomputed embeddings from an external frozen encoder.
            self.text_proj = nn.Linear(text_embed_dim, config.d_model)

    # -- token assembly ----------------------------------------------------
    def build_tokens(
        self,
        observations: Sequence[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
        batch_size: int,
    ) -> torch.Tensor:
        """Assemble ``T_i`` from per-modality observations.

        ``observations`` is a sequence of ``(modality_name, x_n, xy_n, age_days)``
        with shapes ``(B, dim)``, ``(B, 2)`` and ``(B,)``. Sequence length is
        whatever the caller supplies, so an episode with fewer available
        modalities simply produces fewer tokens; no padding is introduced.
        """
        tokens = []
        for name, x_n, xy_n, age_days in observations:
            projected = self.modalities(name, x_n)
            tokens.append(
                projected
                + self.location(xy_n)
                + self.staleness(age_days)
            )
        if not tokens:
            return torch.zeros(batch_size, 0, self.d_model)
        stacked = torch.stack(tokens, dim=1)  # B x N x d_model
        return self.token_norm(stacked)

    def forward(
        self,
        observations: Sequence[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
        text_embedding: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = observations[0][1].shape[0] if observations else 1
        tokens = self.build_tokens(observations, batch_size)
        if text_embedding is not None:
            if self.text_embed_dim is None:
                raise ValueError(
                    "text_embedding was supplied but the model was built without "
                    "text_embed_dim; declare the embedding dimension explicitly"
                )
            tokens = torch.cat([tokens, self.text_proj(text_embedding).unsqueeze(1)], dim=1)

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat([queries, tokens], dim=1)
        encoded = self.transformer(sequence)

        n_queries = self.config.query_tokens
        q_v, q_e, q_r = encoded[:, 0], encoded[:, 1], encoded[:, 2]
        token_out = self.token_head(encoded[:, n_queries:])

        return {
            "v_mu": self.v_mu(q_v),
            "v_log_var": self.v_log_var(q_v),
            "e_mu": self.e_mu(q_e),
            "e_log_var": self.e_log_var(q_e),
            "r_hat": self.r_head(q_r),
            "token_out": token_out,
            "tokens": tokens,
        }

    def sample_state(self, out: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw ``V_hat`` and ``E_hat`` from the Gaussian heads.

        Uses the mean when gradients are disabled, so evaluation is
        deterministic.
        """
        if not torch.is_grad_enabled():
            return out["v_mu"], out["e_mu"]
        v = out["v_mu"] + torch.exp(0.5 * out["v_log_var"]) * torch.randn_like(out["v_mu"])
        e = out["e_mu"] + torch.exp(0.5 * out["e_log_var"]) * torch.randn_like(out["e_mu"])
        return v, e

    def sigma(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Scalar predictive uncertainty ``sigma_i``.

        The mean standard deviation of the vegetation head.
        """
        return torch.exp(0.5 * out["v_log_var"]).mean(dim=-1)


def make_mask(batch: int, n_tokens: int, ratio: float = 0.4,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Boolean mask with ``floor(ratio * n_tokens)`` positions set per row."""
    n_mask = int(ratio * n_tokens)
    mask = torch.zeros(batch, n_tokens, dtype=torch.bool)
    if n_mask == 0:
        return mask
    for b in range(batch):
        idx = torch.randperm(n_tokens, generator=generator)[:n_mask]
        mask[b, idx] = True
    return mask
