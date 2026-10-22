# WP4 analysis plan (fixed before the WP4 grid was run)

Grid: `python -m fedpeer.experiments` (625 unique configurations, see `fedpeer/experiments.py`).
Common settings: 10 cells (20 in the scale check), 96 epochs of 15 min, T_max = 10 unless varied, mock extractor
(noise 0.05), λ = 0.7, μ = 10, sync every 4 epochs, accuracy target for time-to-target: weight MAE ≤ 0.10.
Seeds 0–9 for the primary experiments (E1, E2, E4 late join, E5); 0–4 for E3, E4 scale and E6.

## Hypotheses and primary metrics

| Family | Hypothesis | Experiment | Comparisons | Metrics (direction) |
|---|---|---|---|---|
| F1 | H1 transfer / cold start | E1 per heterogeneity | P vs B3, P vs B4 | `w_mae_mean` (↓), `w_mae_cold` (↓), `rank_mean` (↑), `brier_mean` (↓) |
| F1b | H1 newly joining cells | E4 late join, per heterogeneity | P vs B3, P vs B4 | `w_mae_cold_late` (↓), `time_to_target` (↓) |
| F2 | H2 robustness | E2 per (regime, fraction) | P vs B3, P vs B4 | `w_mae_honest_cp` (↓) |
| F3 | H3 downstream | E1 per heterogeneity | P vs B3 | `welfare` (↑), `sla_satisfaction` (↑), `agreement_rate` (↑), `jain` (↑) |

Negative transfer is a possible outcome, not a failure of the analysis; it is reported with the same tests.

## Tests

- Unit of analysis: one simulation run. Comparisons are paired by seed (identical traces and preferences).
- Paired difference oriented so that > 0 means P is better.
- Two-sided exact sign-flip permutation test on paired differences (no normality assumption).
- Effect size: mean paired difference with a 95% percentile bootstrap CI (10 000 resamples) and Cohen's d_z.
- Multiplicity: Holm's step-down procedure within each family at α = 0.05.
- E3, E5, E6 and the leakage/cost metrics are descriptive (mean and 95% CI), not tested.

## Known limitations fixed in advance

- Synthetic preferences, messages and KPI traces; mock extractor in the grid (real LLM F1 reported separately).
- Transfer gain compares runs on the same seed, whose negotiation trajectories differ after the first decision.
- Leakage Λ is measured against a uniform prior for two observers: the opponent's peer model and the same-tenant
  aggregator. No external observer model.
