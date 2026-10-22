# fedpeer — Paper 0 simulator

Reliability-aware personalized federated peer modeling for LLM-based multi-cell O-RAN negotiation.
This repository implements the proposal step by step; **WP1 (simulator and baselines B0/B1/B2) is done**.

| Module | Proposal |
|---|---|
| `fedpeer/contracts.py` | contract x = (q, d, κ) (eq. 3), private additive utilities and reservation values (eqs. 5–6), preference heterogeneity |
| `fedpeer/network.py` | PRB budget and SLA need model (eqs. 1–2), traffic regimes, credit ledger with conservation (eq. 4) |
| `fedpeer/peer_model.py` | local peer model θ and logistic acceptance (eqs. 7–8), offers-only bid loss |
| `fedpeer/negotiation.py` | alternating offers, deadline T_max, eq. 21 offer choice (λ = own-utility weight), AC_next acceptance |
| `fedpeer/bargaining.py` | Nash product, Nash gap, Pareto set and distance (eqs. 13–16, Proposition 1); evaluator only |
| `fedpeer/simulator.py` | epoch loop, B0 static, B1 full-information Nash mediator, B2 offers-only learning, emergency fallback |

Agents only see their own utility, the validator's feasible set and their own peer model; ground truth is read
only by the evaluator. Fallback allocations are logged separately and never counted as agreements.

## Run

```bash
python -m pip install numpy pytest
python -m pytest -q
python -m fedpeer.run --methods B0 B1 B2 --cells 10 --epochs 96 --seeds 0 1
```

Logs go to `results/wp1/` (`*_negotiations.csv`, `*_sla.csv`, `summary.json`).

## Next

WP2: typed language evidence, LLM extractor (OpenAI-compatible client), product gate, verifier and Beta reputation (B3).
WP3: per-tenant pair-specific federation (B4 FedAvg, reliability weighting, median, personalization → P).
