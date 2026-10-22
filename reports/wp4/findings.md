# WP4 findings (written after reading `report.md`)

Numbers are from `report.md` (625 runs, 10 seeds unless stated). "Significant" means Holm-adjusted p < 0.05
within the pre-registered family.

## Process notes

- A code review after the first grid run found four implementation bugs (strategic lies that were partly true,
  reservation-level consistency compared across roles, federated theta not pushed to models that had not negotiated
  yet, weight-order claims counted twice across roles). They were fixed and the whole grid was re-run before any
  analysis; the pre-registered plan was not changed. The pre-fix runs are kept outside git
  (`results/wp4/runs_before_review.jsonl`).
- Power: with 10 seeds the exact sign-flip test cannot go below p = 2/1024 ≈ 0.002, and Holm over the 24 tests of
  F1 then caps the smallest adjusted p at ≈ 0.047. Several large effects (d_z ≈ 1.3–1.8, p = 0.004–0.008) are
  therefore not significant after correction. More seeds would be a declared amendment, not a silent change.
- `w_mae_cold_late` has n = 9: one seed drew no late-joining cell.

## H1 transfer and cold start

- **Shared priorities (iid): supported against local learning.** P beats B3 on mean weight error (+0.011),
  first-negotiation error (+0.039) and rank agreement (+0.072), all significant. Newly joining cells start much
  better with P (+0.063 first-negotiation error, significant).
- **P vs FedAvg (B4) under iid: no difference** in error; rank agreement slightly better (not significant).
- **Time to the 0.10 accuracy target: no difference** in any comparison.
- **Heterogeneous priorities: negative transfer.** Under severe heterogeneity P is worse than local B3 on mean error
  (−0.016) and held-out Brier loss (−0.016), both significant. Clustered: same direction, not significant.
- **Personalization limits negative transfer:** under severe heterogeneity P beats B4 on mean error (+0.007) and rank
  agreement (+0.052), both significant; removing personalization makes P as bad as B4 (E5, E6).

## H2 robustness

- Against local B3, P has lower error on models of honest counterparts in all 8 attack scenarios (direction 8/8),
  significant for contradictory 10% and reputation-build-up-then-lie 10%.
- **Against the unweighted federation B4 (the comparison H2 names), P shows no reliable difference.** H2 is not
  supported as stated.
- **Pooled reputation launders cell-local deception (E5).** With adversaries sampled per (tenant, cell) agent, a
  counterpart that lies in some cells inherits a good pooled reputation from cells where it is honest: the mean gate
  on false claims rises from 0.17 (per-cell reputation) to 0.47. P without pooled reputation has the lowest
  contamination (0.085) and the best rank agreement (0.788) of all variants. This is the proposal's "transfer
  assumption" failing in a specific, explainable way; pooling is only safe if deception is tenant-wide.
- Language is the component that matters most for honest-counterpart models: removing it raises contamination from
  0.088 to 0.106.

## H3 downstream outcomes

- **Not supported.** No significant welfare, SLA or agreement-rate gain from P over B3 in any heterogeneity level;
  fairness (Jain on surplus) is slightly lower under clustered preferences (significant, −0.008). Better peer models
  did not translate into better agreements in this simulator (consistent with lesson L5).

## Leakage and cost (iid)

- Opponent-observer leakage Λ: B3 0.21, B4 0.29, P 0.28; the same-tenant aggregator reaches Λ ≈ 0.33–0.39.
  Federation makes counterpart preferences more inferable, as the proposal anticipates.
- Federation traffic ≈ 112 KB per simulated day for 10 cells; about 2 100 extracted messages per run.

## Suggested next steps (for discussion, not done)

1. Repeat E2/E5 with **tenant-level** adversaries (the counterpart lies in every cell) to test when pooled reputation
   helps; declare it as an amendment first.
2. Clustered federation (named as optional in the proposal) for the clustered and severe settings.
3. More seeds for F1 as a declared power amendment.
