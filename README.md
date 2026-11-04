# fedpeer — Paper 0 simulator

Reliability-aware personalized federated peer modeling for LLM-based multi-cell O-RAN negotiation.
This repository implements the proposal step by step; **WP1 (simulator, B0/B1/B2), WP2 (gated language, B3) and
WP3 (federation, B4 and P) are done**.

| Module | Proposal |
|---|---|
| `fedpeer/contracts.py` | contract x = (q, d, κ) (eq. 3), private additive utilities and reservation values (eqs. 5–6), preference heterogeneity |
| `fedpeer/network.py` | PRB budget and SLA need model (eqs. 1–2), traffic regimes, credit ledger with conservation (eq. 4) |
| `fedpeer/peer_model.py` | local peer model θ and logistic acceptance (eqs. 7–8), offers-only bid loss |
| `fedpeer/negotiation.py` | alternating offers, deadline T_max, eq. 21 offer choice (λ = own-utility weight), AC_next acceptance |
| `fedpeer/bargaining.py` | Nash product, Nash gap, Pareto set and distance (eqs. 13–16, Proposition 1); evaluator only |
| `fedpeer/evidence.py` | typed claims and evidence records, synthetic message profiles, mock extractor, consistency, product gate (eq. 11), outbound allowlist |
| `fedpeer/llm.py` | LLM extractor over any OpenAI-compatible endpoint, JSON-schema output, on-disk cache |
| `fedpeer/reputation.py` | KPI-claim verifier, decayed Beta reputation with event-id dedupe (eq. 12), cross-cell pooled reputation (eq. 18) |
| `fedpeer/federation.py` | tenant-owned pair-specific aggregation: FedAvg, reliability weights, median (eqs. 15–16), personalization (eq. 17) |
| `fedpeer/simulator.py` | epoch loop, B0 static, B1 full-information Nash mediator, B2 offers-only, B3 gated language, B4 FedAvg, P proposed, emergency fallback |
| `fedpeer/run.py` | CLI, logs, transfer gain Δ_transfer against B3 of the same seed (eq. 19) |
| `fedpeer/experiments.py` | WP4 grid E1–E6 (heterogeneity, corruption, deadlines, late-joining cells, scale, ablations, sensitivity) |
| `fedpeer/analysis.py` | pre-registered paired sign-flip tests, Holm correction, bootstrap CIs, tables and figures |

Agents only see their own utility, the validator's feasible set and their own peer model; ground truth is read
only by the evaluator. Fallback allocations are logged separately and never counted as agreements.

## Run

```bash
python -m pip install numpy pytest
python -m pytest -q
python -m fedpeer.run --methods B0 B1 B2 --cells 10 --epochs 96 --seeds 0 1
```

Language runs (B3) use the mock extractor by default:

```bash
python -m fedpeer.run --methods B2 B3 --evidence strategic_lie --adversary-frac 0.3
python -m fedpeer.run --methods B3 --evidence strategic_lie --no-verification   # ablation
```

Evidence profiles: `honest`, `noise`, `contradictory`, `strategic_lie`, `rep_then_lie`.

Federation (B4 = B3 + FedAvg; P = reliability weights + personalization + pooled reputation by default):

```bash
python -m fedpeer.run --methods B3 B4 P --heterogeneity severe --seeds 0 1
python -m fedpeer.run --methods B3 P --aggregation median --no-personalize --no-pooled-reputation   # ablations
```

`transfer_gain` > 0 means lower held-out acceptance Brier loss than local-only B3 on the same seed.

Real LLM extraction (`pip install openai`, key read from `OPENAI_API_KEY`, responses cached in `.llm_cache/`):

```bash
python -m fedpeer.bench_llm --model gpt-4.1-nano --n 20 --price-in 0.10 --price-out 0.40
# held-out check used in the paper (prompt frozen; 144 API calls, about $0.008 on 23 Oct 2026):
python -m fedpeer.bench_llm --seed 1 --n 200 --tuning-seed 0 --price-in 0.10 --price-out 0.40 --out reports/llm/heldout_gpt-4.1-nano_seed1_n200.json
# gpt-4.1-nano has since been retired; the same check repeated with its replacement (amendment 2):
python -m fedpeer.bench_llm --model gpt-5.4-nano --seed 1 --n 200 --tuning-seed 0 --out reports/llm/heldout_gpt-5.4-nano_seed1_n200.json
python -m fedpeer.bench_llm --model qwen2.5:3b --base-url http://localhost:11434/v1   # Ollama
```

Logs go to the `--out` folder (`*_negotiations.csv`, `*_sla.csv`, `summary.json`).

## WP4 evaluation

The analysis plan was committed before any result existed: [`docs/wp4_preregistration.md`](docs/wp4_preregistration.md).
Two amendments were each committed before their own runs: [`docs/wp4_amendment1.md`](docs/wp4_amendment1.md)
(tenant-level adversaries) and [`docs/wp4_amendment2.md`](docs/wp4_amendment2.md) (real LLM extractor in the loop;
exploratory chip-aware offer rule).

```bash
python -m fedpeer.experiments --workers 4     # 1015 runs, resumable; results/wp4/runs.jsonl
python -m fedpeer.experiments --skip-llm      # all but the 60 E8 runs, which call the OpenAI API
python -m fedpeer.analysis                    # reports/wp4/report.md, figures, copy of runs.jsonl
python -m fedpeer.analysis --paper-figures paper/figures   # also the title-free figures for the manuscript
```

Results: [`reports/wp4/report.md`](reports/wp4/report.md).

## Next

WP5: manuscript (kept outside this repository).
