# fedpeer — Paper 0 simulator

Reliability-aware personalized federated peer modeling for LLM-based multi-cell O-RAN negotiation.
This repository implements the proposal step by step; **WP1 (simulator, B0/B1/B2) and WP2 (gated language, B3)
are done**.

| Module | Proposal |
|---|---|
| `fedpeer/contracts.py` | contract x = (q, d, κ) (eq. 3), private additive utilities and reservation values (eqs. 5–6), preference heterogeneity |
| `fedpeer/network.py` | PRB budget and SLA need model (eqs. 1–2), traffic regimes, credit ledger with conservation (eq. 4) |
| `fedpeer/peer_model.py` | local peer model θ and logistic acceptance (eqs. 7–8), offers-only bid loss |
| `fedpeer/negotiation.py` | alternating offers, deadline T_max, eq. 21 offer choice (λ = own-utility weight), AC_next acceptance |
| `fedpeer/bargaining.py` | Nash product, Nash gap, Pareto set and distance (eqs. 13–16, Proposition 1); evaluator only |
| `fedpeer/evidence.py` | typed claims and evidence records, synthetic message profiles, mock extractor, consistency, product gate (eq. 11), outbound allowlist |
| `fedpeer/llm.py` | LLM extractor over any OpenAI-compatible endpoint, JSON-schema output, on-disk cache |
| `fedpeer/reputation.py` | KPI-claim verifier and decayed Beta reputation with event-id dedupe (eq. 12) |
| `fedpeer/simulator.py` | epoch loop, B0 static, B1 full-information Nash mediator, B2 offers-only learning, B3 gated language, emergency fallback |

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

Real LLM extraction (`pip install openai`, key read from `OPENAI_API_KEY`, responses cached in `.llm_cache/`):

```bash
python -m fedpeer.bench_llm --model gpt-4.1-nano --n 20 --price-in 0.10 --price-out 0.40
python -m fedpeer.bench_llm --model qwen2.5:3b --base-url http://localhost:11434/v1   # Ollama
```

Logs go to the `--out` folder (`*_negotiations.csv`, `*_sla.csv`, `summary.json`).

## Next

WP3: per-tenant pair-specific federation (B4 FedAvg, reliability weighting, median, personalization → P).
