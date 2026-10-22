"""Run methods on matched scenarios; write per-negotiation and SLA logs plus summary.json."""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from .contracts import HETEROGENEITY
from .evidence import PROFILES
from .federation import RULES
from .simulator import METHODS, Config, run

COLUMNS = ("method", "seed", "heterogeneity", "evidence", "agreement_rate", "welfare", "sla_satisfaction",
           "w_mae_cold", "w_mae_mean", "rank_mean", "brier_mean", "transfer_gain", "w_mae_honest_cp", "seconds")


def _add_transfer_gain(summaries):
    """Eq. 19 against the local-only B3 of the same seed: held-out Brier loss and weight error (> 0 = transfer helps)."""
    local = {s["seed"]: s for s in summaries if s["method"] == "B3"}
    for s in summaries:
        if s["seed"] in local and "brier_mean" in s:
            s["transfer_gain"] = local[s["seed"]]["brier_mean"] - s["brier_mean"]
            s["transfer_gain_w_mae"] = local[s["seed"]]["w_mae_mean"] - s["w_mae_mean"]
            s["transfer_gain_cold_w_mae"] = local[s["seed"]]["w_mae_cold"] - s["w_mae_cold"]


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--cells", type=int, default=10)
    p.add_argument("--epochs", type=int, default=96)
    p.add_argument("--t-max", type=int, default=10)
    p.add_argument("--heterogeneity", default="clustered", choices=HETEROGENEITY)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--evidence", default="honest", choices=PROFILES)
    p.add_argument("--adversary-frac", type=float, default=0.3)
    p.add_argument("--no-language", action="store_true")
    p.add_argument("--no-verification", action="store_true")
    p.add_argument("--extractor", default="mock", choices=("mock", "llm"))
    p.add_argument("--llm-model", default="gpt-4.1-nano")
    p.add_argument("--base-url", default=None)
    p.add_argument("--aggregation", default="reliability", choices=RULES)
    p.add_argument("--no-personalize", action="store_true")
    p.add_argument("--no-pooled-reputation", action="store_true")
    p.add_argument("--mu", type=float, default=10.0)
    p.add_argument("--sync-epochs", type=int, default=4)
    p.add_argument("--out", default="results/run")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in a.seeds:
        for m in a.methods:
            cfg = Config(method=m, n_cells=a.cells, n_epochs=a.epochs, t_max=a.t_max,
                         heterogeneity=a.heterogeneity, seed=seed, evidence=a.evidence,
                         adversary_frac=a.adversary_frac, language=not a.no_language,
                         verification=not a.no_verification, extractor=a.extractor,
                         llm_model=a.llm_model, base_url=a.base_url, aggregation=a.aggregation,
                         personalize=not a.no_personalize, pooled_reputation=not a.no_pooled_reputation,
                         mu=a.mu, sync_epochs=a.sync_epochs)
            t0 = time.perf_counter()
            res = run(cfg)
            _write_csv(out / f"{m}_seed{seed}_negotiations.csv", res["negotiations"])
            _write_csv(out / f"{m}_seed{seed}_sla.csv", res["sla"])
            summaries.append({**asdict(cfg), **res["summary"], "seconds": time.perf_counter() - t0})
    _add_transfer_gain(summaries)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))

    print(" ".join(f"{c[:12]:>12}" for c in COLUMNS))
    for s in summaries:
        print(" ".join(f"{s[c]:>12.3f}" if isinstance(s.get(c), float) else f"{s.get(c, '-')!s:>12}"
                       for c in COLUMNS))


if __name__ == "__main__":
    main()
