# WP4 amendment 1: tenant-level adversaries (declared before these runs)

Motivation, from the first WP4 analysis: with adversaries sampled per (tenant, cell) agent, pooled cross-cell
reputation (eq. 18) let a counterpart that lies in some cells inherit a good reputation from cells where it is
honest. Whether pooling helps when deception is tenant-wide was not tested. This amendment adds that test; the
original plan (`wp4_preregistration.md`) and its results are unchanged.

## Design (E7)

- `adversary_unit = "tenant"`: one of the three tenants, drawn per seed, uses the evidence regime in every cell
  (one third of all agents; `adversary_frac` is ignored). iid preferences, all other settings as in E2.
- Regimes: noise, contradictory, strategic_lie, rep_then_lie. Seeds 0–9.
- Methods: B3, B4, P, and P without pooled reputation (per-cell reputation) for every regime.
- Ablations of P under strategic_lie: median, FedAvg weights, no personalization, no language, no verification.

## Hypotheses and metrics

With tenant-wide deception the models that can be corrupted are the models **of the deceiving tenant**, so the
metric is `w_mae_adversarial_cp` (weight MAE of models whose counterpart is adversarial, ↓); `gate_false` (mean gate
on false preference claims, ↓) is the mechanism check.

| Family | Hypothesis | Comparisons per regime | Metrics |
|---|---|---|---|
| F4 | A1: pooled reputation helps when deception is tenant-wide | P vs P without pooled reputation | `w_mae_adversarial_cp`, `gate_false` |
| F4 | A2: H2 retested at tenant level | P vs B4, P vs B3 | `w_mae_adversarial_cp` |

16 tests in one family; same procedures as the original plan (paired by seed, exact sign-flip test, bootstrap CI,
d_z, Holm at α = 0.05). E7 ablations are descriptive.
