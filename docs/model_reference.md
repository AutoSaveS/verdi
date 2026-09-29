# Model

`verdi/model/` implements the three stages of Section 3.3 and Appendix G.
Trained weights and the fine-tuned Stage 3 language model are not included;
Stage 3 returns the diagnostic prompts.

## Stage 1 - Spatial World Model

| Element | Setting |
|---|---|
| Vegetation encoder | `d_V -> 128 -> 128`, LayerNorm, ReLU |
| Environment encoder | `d_E -> 128 -> 128` |
| Cross-attention | bidirectional, 2 layers, 4 heads, `d_model = 128` |
| Latent | `G_phi: 256 -> 2K`, `K = 6` |
| Factors | thermal, water, wind, soil, competition, residual |
| Decoder | `K -> 128 -> 256 -> (d_V + d_E)` |
| Resilience head | `H_psi: K -> 64 -> 32 -> 1`, sigmoid |

Loss: dual reconstruction + disentanglement (marginal KL, total correlation,
weak supervision) + resilience prediction, with `beta` annealed 1 to 4 and
`lambda_TC = 1`, `lambda_a = 0.5`.

`lambda_r` of the resilience term defaults to 1.0. The weak-supervision term
uses pairs that differ in one factor; `verdi.model.losses.weak_supervision_align`
takes them as explicit index pairs.

## Stage 2 - Masked Sensor Transformer

| Element | Setting |
|---|---|
| Token | `W_m(x_n) + e_m + gamma(location) + tau(age)` |
| Projection | modality-specific linear to `d = 256` |
| Position | Fourier features of the cell coordinates |
| Staleness | sinusoidal embedding of the observation age |
| Sequence | variable length, concatenated without padding |
| Transformer | 6 layers, 8 heads, FFN 1024, dropout 0.1 |
| Queries | three learnable tokens `q_V`, `q_E`, `q_R` |
| State heads | Gaussian `V`, `E`; scalar `R` |
| Masking | `floor(0.4 * N_i)` tokens per episode for MSM pretraining |

Fourier band count and staleness half-life are configured in `Stage2Config`.
Text metadata from a frozen language-model encoder enters as precomputed
embeddings through `text_embed_dim`.

`use_gaussian_nll` (default on) trains `sigma` with the negative
log-likelihood of `V_hat ~ N(mu_V, sigma_V)`.

## Stage 3 - diagnostic reasoning

* `Phi_i = z_i - z_base(s_i)`, with `z_base` the species mean of units above a
  healthy-baseline threshold.
* A Gaussian mixture over `Phi_i` for units below the vulnerability threshold,
  fitted with scikit-learn; the number of components defaults to a BIC search.
* A three-level prompt cascade: evidence (cluster mean, extent, per-unit
  attributions, uncertainties), prior reasoning, instruction.

Both thresholds, the covariance type and the instruction wording are arguments
with stated defaults. The fine-tuning configuration (LLaMA-3-8B, LoRA rank 16,
alpha 32, 4-bit GPTQ) is recorded in `Stage3Config`.
