"""WP4 analysis exactly as pre-registered in docs/wp4_preregistration.md: paired exact sign-flip tests,
Holm correction per family, bootstrap CIs, descriptive tables and figures."""
import argparse
import itertools
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import HETEROGENEITY
from .simulator import Config

DEFAULTS = asdict(Config())  # runs logged before a field existed are read with that field's default

NAN = float("nan")
# validated categorical slots in fixed order (dataviz validator: CVD and normal-vision PASS; contrast WARN, so every
# chart carries direct labels and a table)
COLORS = {"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "B3": "#eda100", "B4": "#e87ba4", "P": "#008300",
          "P*": "#9ec5a0"}  # P* = a P variant: same hue family as P, always direct-labelled
LABELS = {"P*": "P, per-cell reputation"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e0"
LOWER_BETTER = {"w_mae_mean", "w_mae_cold", "brier_mean", "w_mae_cold_late", "time_to_target", "w_mae_honest_cp",
                "w_mae_adversarial_cp", "gate_false", "false_admission_rate", "timeout_rate", "nash_gap"}
BASE = dict(n_cells=10, n_epochs=96, t_max=10, evidence="honest", adversary_frac=0.3, adversary_unit="agent",
            language=True, verification=True, aggregation="reliability", personalize=True, pooled_reputation=True,
            mu=10.0, sync_epochs=4, late_join_frac=0.0, lam=0.7)
REGIMES = ("noise", "contradictory", "strategic_lie", "rep_then_lie")
LIE = dict(heterogeneity="iid", evidence="strategic_lie", adversary_frac=0.3)


def select(runs, **kw):
    """Runs of the base configuration overridden by kw, keyed by seed."""
    want = {**BASE, **kw}
    return {r["seed"]: r for r in runs if all(r.get(k, DEFAULTS[k]) == v for k, v in want.items())}


def tenant_level(**kw):
    """Selector kwargs for amendment 1 (E7); method "P*" means P with per-cell reputation."""
    def sel(runs, method):
        extra = {"method": "P", "pooled_reputation": False} if method == "P*" else {"method": method}
        return select(runs, heterogeneity="iid", adversary_unit="tenant", **kw, **extra)
    return sel


def _vals(sel, metric):
    v = np.array([r.get(metric, NAN) for r in sel.values()], float)
    return v[~np.isnan(v)]


def signflip_p(d):
    """Exact two-sided sign-flip permutation p-value for the mean paired difference."""
    signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float((np.abs(signs @ d) >= abs(d.sum()) - 1e-12).mean())


def boot_ci(x, rng):
    if len(x) < 2:
        return NAN, NAN
    m = rng.choice(x, (10_000, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(p_runs, base_runs, metric):
    """P vs a baseline, paired by seed; diff > 0 means P is better."""
    seeds = sorted(s for s in set(p_runs) & set(base_runs)
                   if not np.isnan(p_runs[s].get(metric, NAN)) and not np.isnan(base_runs[s].get(metric, NAN)))
    x = np.array([p_runs[s][metric] for s in seeds])
    y = np.array([base_runs[s][metric] for s in seeds])
    d = y - x if metric in LOWER_BETTER else x - y
    if len(d) < 2:
        return {"n": len(d), "diff": NAN, "lo": NAN, "hi": NAN, "d_z": NAN, "p": NAN}
    lo, hi = boot_ci(d, np.random.default_rng(0))
    sd = d.std(ddof=1)
    return {"n": len(d), "diff": float(d.mean()), "lo": lo, "hi": hi,
            "d_z": float(d.mean() / sd) if sd > 0 else NAN, "p": signflip_p(d)}


def holm(ps):
    ps = np.asarray(ps, float)
    adj, running = np.full(len(ps), NAN), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def families(runs):
    fam = {"F1 transfer, E1": [], "F1b newly joining cells, E4": [], "F2 robustness, E2": [], "F3 downstream, E1": []}
    for h in HETEROGENEITY:
        P = select(runs, method="P", heterogeneity=h)
        for base in ("B3", "B4"):
            for m in ("w_mae_mean", "w_mae_cold", "rank_mean", "brier_mean"):
                fam["F1 transfer, E1"].append((h, m, base, compare(P, select(runs, method=base, heterogeneity=h), m)))
        for m in ("welfare", "sla_satisfaction", "agreement_rate", "jain"):
            fam["F3 downstream, E1"].append((h, m, "B3", compare(P, select(runs, method="B3", heterogeneity=h), m)))
    for h in ("iid", "clustered"):
        P = select(runs, method="P", heterogeneity=h, late_join_frac=0.3)
        for base in ("B3", "B4"):
            for m in ("w_mae_cold_late", "time_to_target"):
                fam["F1b newly joining cells, E4"].append(
                    (h, m, base, compare(P, select(runs, method=base, heterogeneity=h, late_join_frac=0.3), m)))
    for ev in REGIMES:
        for frac in (0.1, 0.3):
            kw = dict(heterogeneity="iid", evidence=ev, adversary_frac=frac)
            P = select(runs, method="P", **kw)
            for base in ("B3", "B4"):
                fam["F2 robustness, E2"].append(
                    (f"{ev} {frac:.0%}", "w_mae_honest_cp", base, compare(P, select(runs, method=base, **kw),
                                                                          "w_mae_honest_cp")))
    if any(r.get("adversary_unit") == "tenant" for r in runs):
        rows = fam["F4 tenant-level adversaries, amendment 1"] = []
        for ev in REGIMES:
            sel = tenant_level(evidence=ev)
            P = sel(runs, "P")
            for m in ("w_mae_adversarial_cp", "gate_false"):
                rows.append((ev, m, "P per-cell rep.", compare(P, sel(runs, "P*"), m)))
            for base in ("B4", "B3"):
                rows.append((ev, "w_mae_adversarial_cp", base, compare(P, sel(runs, base), "w_mae_adversarial_cp")))
    for rows in fam.values():
        for row, adj in zip(rows, holm([r[3]["p"] for r in rows])):
            row[3]["p_holm"] = float(adj)
    return fam


def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def describe(sel, metric, rng):
    v = _vals(sel, metric)
    if not len(v):
        return "–"
    lo, hi = boot_ci(v, rng)
    return f"{v.mean():.3f}" + (f" ±{(hi - lo) / 2:.3f}" if not np.isnan(lo) else "")


def table(rows, metrics, rng):
    """rows: [(label, selection)] -> markdown table of mean ±CI half-width over seeds."""
    out = ["| | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for label, sel in rows:
        out.append(f"| {label} | {len(sel)} | " + " | ".join(describe(sel, m, rng) for m in metrics) + " |")
    return "\n".join(out)


# ---- figures (static PNG for the manuscript)
def _style(ax, ylabel):
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _bars(ax, groups, methods, value, ylabel):
    """Grouped bars with 95% CI whiskers and a direct value label on each bar."""
    rng = np.random.default_rng(0)
    width = 0.8 / len(methods)
    for j, m in enumerate(methods):
        for i, g in enumerate(groups):
            v = _vals(value(g, m), "x")
            if not len(v):
                continue
            lo, hi = boot_ci(v, rng)
            x = i + (j - (len(methods) - 1) / 2) * width
            ax.bar(x, v.mean(), width * 0.92, color=COLORS[m], label=LABELS.get(m, m) if i == 0 else None,
                   edgecolor="white", linewidth=1)
            if not np.isnan(lo):
                ax.errorbar(x, v.mean(), [[v.mean() - lo], [hi - v.mean()]], color=INK, linewidth=1, capsize=2)
            ax.text(x, (hi if not np.isnan(hi) else v.mean()) * 1.01, f"{v.mean():.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=MUTED, rotation=90)
    ax.set_xticks(range(len(groups)), groups)
    ax.margins(y=0.15)  # headroom so value labels clear the panel title
    _style(ax, ylabel)


def _as_x(sel, metric):
    return {s: {"x": r.get(metric, NAN)} for s, r in sel.items()}


def figures(runs, out, titles=True):
    """titles=False drops the in-figure titles, for the manuscript where captions carry them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "savefig.dpi": 200})
    learners = ("B2", "B3", "B4", "P")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (metric, label) in zip(axes, (("w_mae_mean", "weight MAE, all negotiations (lower is better)"),
                                          ("w_mae_cold", "weight MAE, first negotiation (lower is better)"),
                                          ("rank_mean", "weight-rank agreement (higher is better)"))):
        _bars(ax, HETEROGENEITY, learners, lambda g, m: _as_x(select(runs, method=m, heterogeneity=g), metric), label)
    axes[0].legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    titles and fig.suptitle("E1 · peer-model quality by preference heterogeneity (10 seeds, 95% CI)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig1_e1_learning.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, h in zip(axes, HETEROGENEITY):
        ends = []
        for m in learners:
            curves = np.array([r["curve_w_mae"] for r in select(runs, method=m, heterogeneity=h).values()], float)
            if not len(curves):
                continue
            y = np.nanmean(curves, 0)
            ax.plot(range(len(y)), y, color=COLORS[m], linewidth=2, label=m)
            ends.append([y[-1], m])
        placed = -np.inf
        for yv, m in sorted(ends):  # bottom-up, each end label at least 0.006 above the previous one
            placed = max(yv, placed + 0.006)
            ax.text(len(y) - 0.6, placed, m, color=INK, fontsize=7, va="center")
        ax.set_xticks(range(0, 16, 3))
        ax.set_title(h, color=INK, loc="left")
        ax.set_xlabel("negotiations already seen with this counterpart", color=MUTED)
        _style(ax, "weight MAE" if h == "iid" else "")
    axes[0].legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.1))
    titles and fig.suptitle("E1 · learning curves (mean over 10 seeds)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_e1_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, frac in zip(axes, (0.1, 0.3)):
        _bars(ax, REGIMES, ("B3", "B4", "P"), lambda g, m: _as_x(
            select(runs, method=m, heterogeneity="iid", evidence=g, adversary_frac=frac), "w_mae_honest_cp"),
              "weight MAE, honest counterparts" if frac == 0.1 else "")
        ax.set_title(f"{frac:.0%} adversarial agents", color=INK, loc="left")
    axes[0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.1))
    titles and fig.suptitle("E2 · contamination of models of honest counterparts (lower is better)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig3_e2_contamination.png")
    plt.close(fig)

    variants = [("P (full)", {}), ("median", {"aggregation": "median"}), ("FedAvg weights", {"aggregation": "fedavg"}),
                ("no personalization", {"personalize": False}), ("per-cell reputation", {"pooled_reputation": False}),
                ("no language", {"language": False}), ("no verification", {"verification": False})]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (label, kw) in enumerate(variants):
        v = _vals(select(runs, method="P", **LIE, **kw), "w_mae_honest_cp")
        if not len(v):
            continue
        lo, hi = boot_ci(v, rng)
        ax.barh(i, v.mean(), 0.6, color=COLORS["P"] if not kw else "#9ec5a0", edgecolor="white")
        ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], color=INK, linewidth=1, capsize=2)
        ax.text(hi * 1.01, i, f"{v.mean():.3f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(variants)), [v[0] for v in variants])
    ax.invert_yaxis()
    _style(ax, "")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("weight MAE on honest counterparts (lower is better)", color=MUTED)
    titles and ax.set_title("E5 · ablations of P under 30% strategic liars", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig4_e5_ablation.png")
    plt.close(fig)

    if any(r.get("adversary_unit") == "tenant" for r in runs):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        for ax, (metric, label) in zip(axes, (("w_mae_adversarial_cp", "weight MAE, models of the deceiving tenant"),
                                              ("gate_false", "mean gate on false claims"))):
            _bars(ax, REGIMES, ("B3", "B4", "P", "P*"),
                  lambda g, m: _as_x(tenant_level(evidence=g)(runs, m), metric), label)
        axes[0].legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
        titles and fig.suptitle("E7 · one tenant deceives in every cell (amendment 1; lower is better)", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "fig5_e7_tenant_level.png")
        plt.close(fig)


def report(runs, out):
    rng = np.random.default_rng(0)
    learners, all_methods = ("B2", "B3", "B4", "P"), ("B0", "B1", "B2", "B3", "B4", "P")
    lines = ["# WP4 results", "",
             f"{len(runs)} runs. Analysis as pre-registered in `docs/wp4_preregistration.md`. Values are means over "
             "seeds ± half-width of the 95% bootstrap CI. Paired differences are oriented so that > 0 means P is "
             "better; p is the exact sign-flip test, p_holm is Holm-adjusted within the family.", ""]

    lines += ["## Pre-registered tests", ""]
    for name, rows in families(runs).items():
        lines += [f"### {name}", "", "| scenario | metric | P vs | n | diff | 95% CI | d_z | p | p_holm | |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for scen, metric, base, r in rows:
            sig = "✔" if r.get("p_holm", 1) < 0.05 else ""
            lines.append(f"| {scen} | {metric} | {base} | {r['n']} | {_fmt(r['diff'], 4)} | "
                         f"[{_fmt(r['lo'], 4)}, {_fmt(r['hi'], 4)}] | {_fmt(r['d_z'], 2)} | {_fmt(r['p'])} | "
                         f"{_fmt(r.get('p_holm'))} | {sig} |")
        lines.append("")

    lines += ["## E1 · heterogeneity (honest senders)", "", "![E1 learning](fig1_e1_learning.png)", "",
              "![E1 curves](fig2_e1_curves.png)", ""]
    for h in HETEROGENEITY:
        lines += [f"**{h}**", "", table([(m, select(runs, method=m, heterogeneity=h)) for m in all_methods],
                                        ["w_mae_mean", "w_mae_cold", "rank_mean", "brier_mean", "time_to_target",
                                         "welfare", "nash_gap", "jain", "sla_satisfaction", "agreement_rate"], rng), ""]
    lines += ["Leakage and cost (iid):", "",
              table([(m, select(runs, method=m, heterogeneity="iid")) for m in learners],
                    ["leak_mean", "leak_aggregator", "messages", "fl_bytes", "seconds"], rng), ""]

    lines += ["## E2 · corruption (iid preferences)", "", "![E2 contamination](fig3_e2_contamination.png)", ""]
    for ev in REGIMES:
        for frac in (0.1, 0.3):
            kw = dict(heterogeneity="iid", evidence=ev, adversary_frac=frac)
            lines += [f"**{ev}, {frac:.0%} adversarial**", "",
                      table([(m, select(runs, method=m, **kw)) for m in ("B3", "B4", "P")],
                            ["w_mae_honest_cp", "w_mae_adversarial_cp", "gate_true", "gate_false",
                             "false_admission_rate", "extraction_f1", "welfare"], rng), ""]

    lines += ["## E3 · deadlines (iid)", ""]
    for t_max in (5, 10, 20):
        seeds = range(5)
        lines += [f"**T_max = {t_max}**", "",
                  table([(m, {s: r for s, r in select(runs, method=m, heterogeneity="iid", t_max=t_max).items()
                              if s in seeds}) for m in learners],
                        ["agreement_rate", "timeout_rate", "no_agreement_rate", "rounds_agreed", "welfare",
                         "w_mae_mean"], rng), ""]

    lines += ["## E4 · newly joining cells and scale", ""]
    for h in ("iid", "clustered"):
        lines += [f"**30% of cells join at mid-horizon, {h}**", "",
                  table([(m, select(runs, method=m, heterogeneity=h, late_join_frac=0.3)) for m in ("B3", "B4", "P")],
                        ["w_mae_cold_late", "w_mae_cold", "time_to_target", "target_reached", "w_mae_mean"], rng), ""]
    lines += ["**20 cells, iid**", "", table([(m, select(runs, method=m, heterogeneity="iid", n_cells=20))
                                              for m in ("B3", "B4", "P")],
                                             ["w_mae_mean", "w_mae_cold", "rank_mean", "fl_bytes", "seconds"], rng), ""]

    lines += ["## E5 · ablations of P (30% strategic liars, iid)", "", "![E5 ablation](fig4_e5_ablation.png)", "",
              table([("B3", select(runs, method="B3", **LIE)), ("B4", select(runs, method="B4", **LIE)),
                     ("P", select(runs, method="P", **LIE))] +
                    [(f"P, {k}={v}", select(runs, method="P", **LIE, **{k: v}))
                     for k, v in (("aggregation", "median"), ("aggregation", "fedavg"), ("personalize", False),
                                  ("pooled_reputation", False), ("language", False), ("verification", False))],
                    ["w_mae_honest_cp", "w_mae_mean", "rank_mean", "gate_false", "welfare"], rng), ""]

    if any(r.get("adversary_unit") == "tenant" for r in runs):
        lines += ["## E7 · tenant-level adversaries (amendment 1, iid)", "",
                  "One tenant uses the regime in every cell. `P*` = P with per-cell instead of pooled reputation.", "",
                  "![E7 tenant level](fig5_e7_tenant_level.png)", ""]
        for ev in REGIMES:
            sel = tenant_level(evidence=ev)
            lines += [f"**{ev}**", "", table([(m, sel(runs, m)) for m in ("B3", "B4", "P", "P*")],
                                             ["w_mae_adversarial_cp", "w_mae_honest_cp", "gate_true", "gate_false",
                                              "rank_mean", "welfare"], rng), ""]
        sel = tenant_level(evidence="strategic_lie")
        lines += ["**Ablations of P, strategic_lie**", "",
                  table([("P", sel(runs, "P")), ("P*", sel(runs, "P*"))] +
                        [(f"P, {k}={v}", select(runs, method="P", heterogeneity="iid", adversary_unit="tenant",
                                                evidence="strategic_lie", **{k: v}))
                         for k, v in (("aggregation", "median"), ("aggregation", "fedavg"), ("personalize", False),
                                      ("language", False), ("verification", False))],
                        ["w_mae_adversarial_cp", "w_mae_honest_cp", "gate_false", "rank_mean"], rng), ""]

    lines += ["## E6 · sensitivity of P (iid, honest)", "",
              table([("P (μ=10, sync 4, λ=0.7)", {s: r for s, r in select(runs, method="P", heterogeneity="iid").items()
                                                  if s < 5})] +
                    [(f"P, {k}={v}", select(runs, method="P", heterogeneity="iid", **{k: v}))
                     for k, v in (("mu", 1.0), ("mu", 100.0), ("sync_epochs", 2), ("sync_epochs", 8), ("lam", 0.5),
                                  ("lam", 0.9))],
                    ["w_mae_mean", "w_mae_cold", "rank_mean", "welfare", "fl_bytes"], rng), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/wp4/runs.jsonl")
    p.add_argument("--out", default="reports/wp4")
    p.add_argument("--paper-figures", help="also write the figures without titles to this folder")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(line) for line in Path(a.runs).read_text().splitlines()]
    shutil.copyfile(a.runs, out / "runs.jsonl")  # raw per-run summaries travel with the report
    report(runs, out)
    figures(runs, out)
    if a.paper_figures:
        Path(a.paper_figures).mkdir(parents=True, exist_ok=True)
        figures(runs, Path(a.paper_figures), titles=False)
    print(f"wrote {out / 'report.md'} and {len(list(out.glob('fig*.png')))} figures from {len(runs)} runs")


if __name__ == "__main__":
    main()
