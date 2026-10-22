"""WP4 experiment grid (proposal Sec. 9). Each unique configuration runs once, in parallel; results are appended
to runs.jsonl as they finish, so an interrupted grid resumes where it stopped."""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

from .contracts import HETEROGENEITY
from .simulator import Config, run

SEEDS = range(10)
LEARNERS = ("B3", "B4", "P")


def grid():
    g = []

    def add(methods, seeds=SEEDS, **kw):
        g.extend(Config(method=m, seed=s, **kw) for m in methods for s in seeds)

    # E1 heterogeneity (H1, H3): every method, honest senders
    for h in HETEROGENEITY:
        add(("B0", "B1", "B2", *LEARNERS), heterogeneity=h)
    # E2 corruption (H2): shared priorities, so attacks are not confounded with negative transfer
    for ev in ("noise", "contradictory", "strategic_lie", "rep_then_lie"):
        for frac in (0.1, 0.3):
            add(LEARNERS, heterogeneity="iid", evidence=ev, adversary_frac=frac)
    # E3 deadlines (T = 10 is in E1)
    for t_max in (5, 20):
        add(("B2", *LEARNERS), seeds=range(5), heterogeneity="iid", t_max=t_max)
    # E4 newly joining cells (cold start) and scale
    for h in ("iid", "clustered"):
        add(LEARNERS, heterogeneity=h, late_join_frac=0.3)
    add(LEARNERS, seeds=range(5), heterogeneity="iid", n_cells=20)
    # E5 ablations of P under 30% strategic liars (full P and B3/B4 come from E2)
    lie = dict(heterogeneity="iid", evidence="strategic_lie", adversary_frac=0.3)
    for kw in ({"aggregation": "median"}, {"aggregation": "fedavg"}, {"personalize": False},
               {"pooled_reputation": False}, {"language": False}, {"verification": False}):
        add(("P",), **lie, **kw)
    # E6 sensitivity of P (base P comes from E1 iid)
    for kw in ({"mu": 1.0}, {"mu": 100.0}, {"sync_epochs": 2}, {"sync_epochs": 8}, {"lam": 0.5}, {"lam": 0.9}):
        add(("P",), seeds=range(5), heterogeneity="iid", **kw)
    return list({key(c): c for c in g}.values())


def key(cfg):
    return hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:16]


def _work(cfg):
    t0 = time.perf_counter()
    summary = run(cfg)["summary"]
    return {"key": key(cfg), **asdict(cfg), **summary, "seconds": time.perf_counter() - t0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/wp4/runs.jsonl")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="only count the runs still to do")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["key"] for line in out.read_text().splitlines()} if out.exists() else set()
    # heaviest first, so the tail of the grid is short
    todo = sorted((c for c in grid() if key(c) not in done), key=lambda c: (c.method in LEARNERS) * c.n_cells,
                  reverse=True)
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    if a.dry_run or not todo:
        return
    t0 = time.perf_counter()
    with Pool(a.workers) as pool, open(out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(_work, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {rec['method']} seed={rec['seed']} {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
