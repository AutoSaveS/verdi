# Reporting status of the analyses

Several analyses are specified in the manuscript but not reported, because the
retained records do not allow them to be verified. This file mirrors the
statistical-status table of the response letter, so that nothing here is
mistaken for a completed result.

## Reported

| Analysis | Where |
|---|---|
| Held-out R^2 by method and city | mean +/- SD over five seeds |
| Predictive uncertainty by city | per-city standard deviation |
| Experiment 2 paired comparisons | Bonferroni-corrected significance and Cohen's d |
| Retention relative to the full-modality configuration | C2 acceptance |

## Specified but not reported

| Analysis | Status |
|---|---|
| Expected calibration error (ECE) | protocol only; no numerical values |
| Latent-indicator Spearman alignment | protocol only |
| Off-target leakage diagnostics | protocol only |
| BCa bootstrap intervals | not reported |
| Residual Moran's I | planned diagnostic, not reported |
| Wilcoxon and cross-experiment significance | not reported |
| Cross-city transfer estimates | design description only |
| Stage 3 paired-bootstrap significance and inter-rater reliability | records not verifiable |
| Hyperparameter sensitivity sweeps | design only |
| Delta resolution ablation | design only |
| Label-combination strategies other than `R*_C` | design only |

The metrics are implemented in `verdi/eval/metrics.py` so a project can compute
them and record the choices, not because the manuscript reports values.

## Interpretation limits carried into the code

* The labels are proxies; they do not measure recovery, persistence, functional
  stability or regeneration.
* Latent factor readouts and Stage 3 narratives are associative, not causal.
* The case studies are retrospective illustrations, not deployment evaluations.
* Near-real-time refers to the five-day update cadence only.
* `R*_C` shares NDVI with several inputs, so label-input overlap is possible.
