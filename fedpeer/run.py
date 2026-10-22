"""Run methods on matched scenarios; write per-negotiation and SLA logs plus summary.json."""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from .contracts import HETEROGENEITY
from .simulator import METHODS, Config, run

COLUMNS = ("method", "seed", "triggers", "negotiations", "agreement_rate", "timeout_rate", "fallback_rate",
           "rounds_agreed", "welfare", "nash_gap", "pareto_dist", "sla_satisfaction", "utilization", "seconds")


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
    p.add_argument("--out", default="results/wp1")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in a.seeds:
        for m in a.methods:
            cfg = Config(method=m, n_cells=a.cells, n_epochs=a.epochs, t_max=a.t_max,
                         heterogeneity=a.heterogeneity, seed=seed)
            t0 = time.perf_counter()
            res = run(cfg)
            _write_csv(out / f"{m}_seed{seed}_negotiations.csv", res["negotiations"])
            _write_csv(out / f"{m}_seed{seed}_sla.csv", res["sla"])
            summaries.append({**asdict(cfg), **res["summary"], "seconds": time.perf_counter() - t0})
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))

    print(" ".join(f"{c[:12]:>12}" for c in COLUMNS))
    for s in summaries:
        print(" ".join(f"{s[c]:>12.3f}" if isinstance(s[c], float) else f"{s[c]!s:>12}" for c in COLUMNS))


if __name__ == "__main__":
    main()
