# WP4 amendment 2: real LLM in the loop and a chip-aware offer rule (declared before these runs)

Motivation, from the WP4 results and the manuscript draft:
1. All 835 runs used a mock extractor; the real LLM was only benchmarked offline (held-out micro-F1 0.897,
   `reports/llm/heldout_gpt-4.1-nano_seed1_n200.json`). Whether the main learning and robustness results hold with a
   real LLM inside the negotiation loop was not tested.
2. H3 was not supported: better peer models did not give better agreements. The offer rule (eq. 6) gives the peer
   model weight 1 − λ = 0.3, and only among contracts in a narrow band above the agent's own target, so model quality
   can hardly change which contract is offered. Classic negotiation work proposes trade-off (logrolling) moves:
   keep one's own utility and give the counterpart what it values most, i.e. concede on issues that are cheap for
   oneself and valuable for the counterpart ("bargaining chips" in the trade-off sense; not the procurement items of
   the Bargaining Chips framework of Baarslag et al. 2021).

Part 2 is **post hoc** (motivated by the H3 result) and therefore **exploratory**. The original plan
(`wp4_preregistration.md`), amendment 1 and their results are unchanged. This file is committed before any E8 or E9
run exists.

## Design

**E8 — real LLM extractor in the loop.** `extractor = "llm"`, model **gpt-5.4-nano** (snapshot 2026-03-17) through
the OpenAI API, with the JSON schema and the prompt in `fedpeer/llm.py` frozen at this commit (the prompt of the
held-out check). gpt-4.1-nano, used in the held-out check, has been retired by the provider (the API returns 404), so
the held-out check is repeated with gpt-5.4-nano on the same 200 messages (seed 1, prompt unchanged) and reported next
to the earlier result. Responses
are cached by message text, so an identical message always gets the identical extraction (no sampling variance).
Settings: E1 iid with honest senders, and E2 iid with 30% strategic liars (agent level). Methods B3, B4, P.
Seeds 0–9. All other settings as in the original plan.

**E9 — chip-aware offer rule (exploratory).** `tradeoff = True` for both parties: among feasible contracts whose own
utility is at least the time-dependent concession target, the agent offers the one with the highest estimated
counterpart utility under its peer model (ties to its own utility). Acceptance (AC_next, reservation) and everything
else are unchanged. Settings: E1 (iid, clustered, severe; honest senders). Methods B2, B3, B4, P. Seeds 0–9. Plain-rule
runs come from E1.

## Hypotheses and tests

Same procedures as the original plan: unit = run, paired by seed, two-sided exact sign-flip test, 95% bootstrap CI,
Cohen's d_z, Holm at α = 0.05 within each family. Differences oriented so that > 0 favours the first-named method.

| Family | Hypothesis | Comparisons | Metrics |
|---|---|---|---|
| F5 (10 tests) | R1: the F1 (iid) and F2 (30% strategic lie) results hold with a real LLM extractor | P vs B3, P vs B4, both with the LLM extractor | iid: `w_mae_mean`, `w_mae_cold`, `rank_mean`, `brier_mean`; strategic lie 30%: `w_mae_honest_cp` |
| F6 (12 tests, exploratory) | C1: the chip-aware rule raises welfare; C2: with the chip-aware rule, better peer models give better agreements (H3 retested) | per heterogeneity: P-chip vs P-plain (`welfare`, `nash_gap`); P-chip vs B3-chip (`welfare`); B3-chip vs B2-chip (`welfare`, the language-improved model vs offers-only) | `welfare`, `nash_gap` |

Descriptive (not tested): mock vs LLM side by side for B3, B4, P (learning metrics, `extraction_f1`, gates, welfare);
chip vs plain for B2–P (welfare, Nash gap, Pareto distance, agreement rate, Jain, SLA, rank agreement).

## Known limitations fixed in advance

- E8 messages still come from the sentence templates; the LLM is tested on extraction, not on free-form language.
- The response cache makes E8 deterministic given the message text; LLM run-to-run variability is not measured.
- F6 is exploratory and post hoc; a positive result would motivate a confirmatory study, not establish C1/C2.
- 10 seeds limit power as in the original plan (smallest attainable adjusted p in F6 ≈ 12 × 2/1024 ≈ 0.023).
