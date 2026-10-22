"""Multi-cell epoch loop (proposal Sec. 8, Steps 1-8): B0 static slicing, B1 full-information Nash mediator,
B2 offers-only local peer models, B3 = B2 + gated language with verified reputation, B4 = B3 + FedAvg and
P = B3 + pair-specific reliability-aware federation + personalization + pooled verified reputation."""
from dataclasses import dataclass

import numpy as np

from .bargaining import benchmarks
from .contracts import Q_GRID, X, sample_preferences
from .evidence import ADMIT, MockExtractor, compose, consistency, f1, gate, is_true, match_counts
from .federation import aggregate, personalize
from .negotiation import Negotiator, negotiate
from .network import STATIC_SHARE, TENANTS, Ledger, demand_traces
from .peer_model import PeerModel
from .reputation import BetaReputation, verify

METHODS = ("B0", "B1", "B2", "B3", "B4", "P")
LANGUAGE = ("B3", "B4", "P")
FEDERATED = ("B4", "P")
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
    # B3 language evidence
    evidence: str = "honest"     # profile of adversarial agents, see evidence.PROFILES
    adversary_frac: float = 0.3  # share of (tenant, cell) agents using that profile; the rest are honest
    language: bool = True        # ablation: False keeps messages but never uses them
    verification: bool = True    # ablation: False never updates reputation
    extractor: str = "mock"      # "mock" or "llm"
    extract_noise: float = 0.05
    llm_model: str = "gpt-4.1-nano"
    base_url: str | None = None  # e.g. http://localhost:11434/v1 for Ollama
    # P federation (B4 always uses FedAvg, no personalization, per-cell reputation)
    aggregation: str = "reliability"  # "reliability" | "fedavg" | "median"
    personalize: bool = True
    pooled_reputation: bool = True
    mu: float = 10.0                  # eq. 17 cold-start hyperparameter
    sync_epochs: int = 4              # 60 min at 15-min epochs


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

    lang_rng = np.random.default_rng([cfg.seed, 1])  # own stream, so traces and preferences stay matched
    profiles = {(s, n): cfg.evidence if lang_rng.random() < cfg.adversary_frac else "honest"
                for s in range(n_t) for n in range(cfg.n_cells)}
    if cfg.method in LANGUAGE and cfg.extractor == "llm":
        from .llm import LLMExtractor
        extractor = LLMExtractor(cfg.llm_model, cfg.base_url)
    else:
        extractor = MockExtractor(np.random.default_rng([cfg.seed, 2]), cfg.extract_noise)
    rep, pending, said = BetaReputation(), [], {}
    stats = {"tp": 0, "fp": 0, "fn": 0, "g_true": [], "g_false": [], "fl_bytes": 0, "syncs": 0}

    rule = "fedavg" if cfg.method == "B4" else cfg.aggregation
    personal = cfg.method == "P" and cfg.personalize
    pooled = cfg.method == "P" and cfg.pooled_reputation
    fed = {}  # (owner tenant s, counterpart s') -> latest global theta of that tenant's aggregator

    def peer(n, s, sp):
        """Persistent model of s' held by tenant s in cell n; a new one starts from s's federated model if any."""
        if (n, s, sp) not in peers:
            peers[n, s, sp] = PeerModel()
            if cfg.method in FEDERATED and (s, sp) in fed:
                peers[n, s, sp].load(fed[s, sp])  # D_n = 0, so beta = 1: pure transfer at cold start
        return peers[n, s, sp]

    def sync():
        """Method step 8: each tenant aggregates its own agents' models of one counterpart across cells."""
        for s, sp in {(s, sp) for _, s, sp in peers}:
            members = [peers[n, s, sp] for n in range(cfg.n_cells)
                       if (n, s, sp) in peers and peers[n, s, sp].n_negotiations > 0]
            if not members:
                continue
            fed[s, sp] = aggregate([m.theta for m in members], [m.n_negotiations for m in members],
                                   [m.reliability_mass for m in members], rule)
            for m in members:
                m.load(personalize(fed[s, sp], m.theta, m.n_negotiations, cfg.mu) if personal else fed[s, sp])
            stats["fl_bytes"] += 2 * len(members) * fed[s, sp].nbytes  # upload + download
        stats["syncs"] += 1

    def converse(hist, R, D, n, t):
        """A message rides with every offer (Method step 4); returns the gated preference claims each side learns."""
        learn = {R: [], D: []}
        for k, (role, _) in enumerate(hist):
            snd, obs = (R, D) if role == "R" else (D, R)
            prof = profiles[snd, n]
            lying = prof == "strategic_lie" or (prof == "rep_then_lie" and t >= cfg.n_epochs // 2)
            msg = compose(lang_rng, prof, prefs[snd, n], role, need[n, snd, min(t + 1, cfg.n_epochs - 1)],
                          snd, n, t, f"{t}:{n}:{R}:{D}:{k}", lying)
            evs = extractor(msg)
            for key, x in zip(("tp", "fp", "fn"), match_counts(msg.claims, evs)):
                stats[key] += x
            for ev in evs:
                if not ev.v:
                    continue
                if ev.claim.kind == "kpi_need":  # verifiable later, feeds reputation only
                    if t + 1 < cfg.n_epochs:
                        pending.append((t + 1, n, obs, snd, ev.claim.value, ev.event_id))
                    continue
                history = said.setdefault((n, obs, snd), [])
                # a = 1: preference claims are not verifiable now (neutral, not "true")
                # shortcut: pooled reputation is read live, not only at sync epochs; upgrade if transport delay matters
                r = rep.pooled(obs, snd) if pooled else rep.score((n, obs, snd))
                g = gate(ev.c, ev.v, consistency(ev.claim, history), 1.0, r)
                history.append(ev.claim)
                stats["g_true" if is_true(ev.claim, prefs[snd, n], role) else "g_false"].append(g)
                if cfg.language:
                    learn[obs].append((ev.claim, role, g))
        return learn

    for t in range(cfg.n_epochs):
        active = [c for c in active if c[4] > t]
        if cfg.verification:  # Method step 7: verify after the measurement window
            for due, n, obs, snd, claimed, eid in [p for p in pending if p[0] <= t]:
                rep.update((n, obs, snd), verify(claimed, need[n, snd, due]), eid)
        pending[:] = [p for p in pending if p[0] > t]
        if cfg.method in FEDERATED and t > 0 and t % cfg.sync_epochs == 0:
            sync()
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
                    models = {D: (peer(n, R, D), peer(n, D, R)) for D in donors}
                    # receiver's own view only: estimated acceptance times its own surplus
                    score = {D: np.where(feas[D], models[D][0].p_accept("D") * np.maximum(uR - rR, 0), 0).max()
                             for D in donors}
                    for D in sorted(donors, key=score.get, reverse=True):
                        m_rd, m_dr = models[D]
                        err = {"w_mae": m_rd.weight_mae(prefs[D, n].w), "u_mae": m_rd.utility_mae(prefs[D, n], "D"),
                               "rank": m_rd.rank_agreement(prefs[D, n].w), "n_seen": m_rd.n_negotiations,
                               "counterpart_profile": profiles[D, n]}
                        outcome, idx, rounds, hist = negotiate(
                            Negotiator(uR, rR, "R", m_rd.utility("D"), cfg.lam),
                            Negotiator(prefs[D, n].utility("D"), prefs[D, n].r["D"], "D", m_dr.utility("R"), cfg.lam),
                            feas[D], cfg.t_max)
                        if hist:  # held-out: scored before the model sees this negotiation
                            err["brier"] = m_rd.brier(hist, "D", outcome == "agreement")
                        learn = converse(hist, R, D, n, t) if cfg.method in LANGUAGE else {R: (), D: ()}
                        m_rd.observe(hist, "D", outcome == "agreement", learn[R])
                        m_dr.observe(hist, "R", outcome == "agreement", learn[D])
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
            "summary": summarize(negs, sla, cfg.budget, stats if cfg.method in LANGUAGE else None)}


def summarize(negs, sla, budget, stats=None):
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
        for key in ("w_mae", "u_mae", "rank", "brier"):
            rows = [r for r in learned if key in r]
            s[f"{key}_mean"] = mean(rows, key)
            s[f"{key}_cold"] = mean([r for r in rows if r["n_seen"] == 0], key)
            s[f"{key}_warm"] = mean([r for r in rows if r["n_seen"] >= 10], key)
        # H2: error on models of honest counterparts shows contamination spreading from adversarial ones
        s["w_mae_honest_cp"] = mean([r for r in learned if r["counterpart_profile"] == "honest"], "w_mae")
        s["w_mae_adversarial_cp"] = mean([r for r in learned if r["counterpart_profile"] != "honest"], "w_mae")
    if stats:
        g_true, g_false = np.array(stats["g_true"]), np.array(stats["g_false"])
        s["extraction_f1"] = f1(stats["tp"], stats["fp"], stats["fn"])
        s["false_admission_rate"] = float((g_false > ADMIT).mean()) if len(g_false) else nan
        s["true_admission_rate"] = float((g_true > ADMIT).mean()) if len(g_true) else nan
        s["gate_true"] = float(g_true.mean()) if len(g_true) else nan
        s["gate_false"] = float(g_false.mean()) if len(g_false) else nan
        s["fl_bytes"], s["syncs"] = stats["fl_bytes"], stats["syncs"]
    return s
