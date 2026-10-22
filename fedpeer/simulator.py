"""Multi-cell epoch loop (proposal Sec. 8, Steps 1-6) for B0 static slicing, B1 full-information
Nash mediator and B2 offers-only local peer models."""
from dataclasses import dataclass

import numpy as np

from .bargaining import benchmarks
from .contracts import Q_GRID, X, sample_preferences
from .negotiation import Negotiator, negotiate
from .network import STATIC_SHARE, TENANTS, Ledger, demand_traces
from .peer_model import PeerModel

METHODS = ("B0", "B1", "B2")
BILATERAL = ("agreement", "timeout", "no_agreement")


@dataclass
class Config:
    method: str = "B2"
    n_cells: int = 10
    n_epochs: int = 96
    t_max: int = 10
    heterogeneity: str = "clustered"
    seed: int = 0
    lam: float = 0.7             # own-utility weight in eq. 21
    risk_threshold: float = 0.9  # trigger when need > threshold * allocation
    fallback_cap: float = 10.0   # PRBs of pre-authorized emergency allocation
    credit0: float = 100.0
    budget: float = 100.0        # PRBs per cell


def _alloc(static, active, n):
    """Current allocation b and committed outgoing PRBs of each tenant in cell n."""
    b, out = static.copy(), np.zeros_like(static)
    for cell, R, D, q, _ in active:
        if cell == n:
            b[R] += q
            b[D] -= q
            out[D] += q
    return b, out


def _evaluate(pR, pD, feasible, idx):
    """Evaluator-only ground truth for one bilateral outcome; never visible to the agents."""
    uR, uD = pR.utility("R"), pD.utility("D")
    bm = benchmarks(uR, uD, pR.r["R"], pD.r["D"], feasible, idx)
    rec = {"fir_empty": bm["fir_empty"]}
    if idx is not None:
        q, d, k = X[idx]
        rec.update(q=q, d=d, kappa=k, u_r=uR[idx], u_d=uD[idx], surplus_r=uR[idx] - pR.r["R"],
                   surplus_d=uD[idx] - pD.r["D"], welfare=uR[idx] + uD[idx],
                   nash_gap=bm["nash_gap"], pareto_dist=bm["pareto_dist"])
    return rec


def run(cfg):
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method {cfg.method!r}")
    rng = np.random.default_rng(cfg.seed)
    n_t = len(TENANTS)
    # drawn first and only here, so every method sees identical traces and preferences per seed
    need, regimes = demand_traces(rng, cfg.n_cells, cfg.n_epochs, cfg.budget)
    prefs = sample_preferences(rng, n_t, cfg.n_cells, cfg.heterogeneity)
    static = STATIC_SHARE * cfg.budget
    ledger = Ledger(n_t, cfg.credit0)
    active, peers, negs, sla = [], {}, [], []

    for t in range(cfg.n_epochs):
        active = [c for c in active if c[4] > t]
        for n in range(cfg.n_cells):
            nd = need[n, :, t]
            b, _ = _alloc(static, active, n)
            for R in np.argsort(-nd / np.maximum(b, 1e-9)):  # a donor may have lent its whole share
                b, out = _alloc(static, active, n)
                if nd[R] <= cfg.risk_threshold * b[R]:
                    continue
                slack = np.minimum(b - nd, static - out)  # no re-donating borrowed PRBs; keeps b >= 0
                donors = [D for D in range(n_t) if D != R and slack[D] >= Q_GRID[0]]
                base = {"t": t, "cell": n, "regime": regimes[n], "receiver": TENANTS[R]}
                if cfg.method == "B0":
                    negs.append({**base, "outcome": "static"})
                    continue
                feas = {D: (X[:, 0] <= slack[D]) & (X[:, 2] <= ledger.balance[R]) for D in donors}
                agreed = False

                def settle(D, idx):
                    q, d, kappa = X[idx]
                    ledger.transfer(R, D, kappa)
                    active.append([n, R, D, q, t + int(d)])

                if cfg.method == "B1":
                    best = None
                    for D in donors:
                        bm = benchmarks(prefs[R, n].utility("R"), prefs[D, n].utility("D"),
                                        prefs[R, n].r["R"], prefs[D, n].r["D"], feas[D])
                        if not bm["fir_empty"] and (best is None or bm["nash_max"] > best[0]):
                            best = (bm["nash_max"], D, bm["nash_idx"])
                    if best:
                        _, D, idx = best
                        negs.append({**base, "donor": TENANTS[D], "donor_slack": slack[D], "outcome": "agreement",
                                     "rounds": 0, **_evaluate(prefs[R, n], prefs[D, n], feas[D], idx)})
                        settle(D, idx)
                        agreed = True
                    elif donors:
                        negs.append({**base, "outcome": "no_agreement", "rounds": 0, "fir_empty": True})
                else:
                    uR, rR = prefs[R, n].utility("R"), prefs[R, n].r["R"]
                    models = {D: (peers.setdefault((n, R, D), PeerModel()), peers.setdefault((n, D, R), PeerModel()))
                              for D in donors}
                    # receiver's own view only: estimated acceptance times its own surplus
                    score = {D: np.where(feas[D], models[D][0].p_accept("D") * np.maximum(uR - rR, 0), 0).max()
                             for D in donors}
                    for D in sorted(donors, key=score.get, reverse=True):
                        m_rd, m_dr = models[D]
                        err = {"w_mae": m_rd.weight_mae(prefs[D, n].w), "u_mae": m_rd.utility_mae(prefs[D, n], "D"),
                               "n_seen": m_rd.n_negotiations}
                        outcome, idx, rounds, hist = negotiate(
                            Negotiator(uR, rR, "R", m_rd.utility("D"), cfg.lam),
                            Negotiator(prefs[D, n].utility("D"), prefs[D, n].r["D"], "D", m_dr.utility("R"), cfg.lam),
                            feas[D], cfg.t_max)
                        m_rd.observe(hist, "D", outcome == "agreement")
                        m_dr.observe(hist, "R", outcome == "agreement")
                        negs.append({**base, "donor": TENANTS[D], "donor_slack": slack[D], "outcome": outcome,
                                     "rounds": rounds, **err, **_evaluate(prefs[R, n], prefs[D, n], feas[D], idx)})
                        if outcome == "agreement":
                            settle(D, idx)
                            agreed = True
                            break

                if not agreed:
                    others = [D for D in range(n_t) if D != R]
                    D = max(others, key=lambda j: slack[j])
                    g = min(cfg.fallback_cap, nd[R] - b[R], slack[D])
                    if g > 0:  # pre-authorized capped emergency allocation, one epoch, no credit
                        active.append([n, R, D, g, t + 1])
                        negs.append({**base, "donor": TENANTS[D], "outcome": "fallback", "q_fallback": g})
                    elif not donors:
                        negs.append({**base, "outcome": "no_donor"})

            b, _ = _alloc(static, active, n)
            assert (b >= -1e-9).all() and b.sum() <= cfg.budget + 1e-9, "PRB budget or non-negativity violated"
            sla += [{"t": t, "cell": n, "tenant": TENANTS[s], "need": nd[s], "alloc": b[s]} for s in range(n_t)]

    return {"negotiations": negs, "sla": sla, "credit_total": float(ledger.balance.sum()),
            "summary": summarize(negs, sla, cfg.budget)}


def summarize(negs, sla, budget):
    nan = float("nan")
    bil = [r for r in negs if r["outcome"] in BILATERAL]
    agr = [r for r in bil if r["outcome"] == "agreement"]
    triggers = len({(r["t"], r["cell"], r["receiver"]) for r in negs})

    def rate(rows, outcome):
        return sum(r["outcome"] == outcome for r in rows) / len(rows) if rows else nan

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else nan

    need = np.array([r["need"] for r in sla])
    alloc = np.array([r["alloc"] for r in sla])
    ok = alloc >= need
    s = {"triggers": triggers, "negotiations": len(bil),
         "agreement_rate": rate(bil, "agreement"), "timeout_rate": rate(bil, "timeout"),
         "no_agreement_rate": rate(bil, "no_agreement"),
         "fallback_rate": sum(r["outcome"] == "fallback" for r in negs) / triggers if triggers else nan,
         "rounds_agreed": mean(agr, "rounds"), "welfare": mean(agr, "welfare"),
         "nash_gap": mean(agr, "nash_gap"), "pareto_dist": mean(agr, "pareto_dist"),
         "sla_satisfaction": float(ok.mean()),
         "utilization": float(np.minimum(alloc, need).sum() / (budget * len(sla) / len(TENANTS)))}
    for name in TENANTS:
        s[f"sla_{name}"] = float(ok[[r["tenant"] == name for r in sla]].mean())
    learned = [r for r in bil if "w_mae" in r]
    if learned:  # cold = model has seen no negotiation with this counterpart, warm = at least 10
        for key in ("w_mae", "u_mae"):
            s[f"{key}_cold"] = mean([r for r in learned if r["n_seen"] == 0], key)
            s[f"{key}_warm"] = mean([r for r in learned if r["n_seen"] >= 10], key)
    return s
