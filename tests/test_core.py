import numpy as np
import pytest

from fedpeer.bargaining import benchmarks
from fedpeer.contracts import Q_GRID, X, sample_preferences
from fedpeer.negotiation import Negotiator, negotiate
from fedpeer.network import Ledger
from fedpeer.peer_model import PeerModel
from fedpeer.simulator import METHODS, Config, run


def test_ledger_conserves_and_rejects_invalid_transfers():
    ledger = Ledger(3, 10)
    ledger.transfer(0, 1, 4)
    ledger.transfer(1, 2, 7)
    assert ledger.balance.sum() == 30
    with pytest.raises(ValueError):
        ledger.transfer(0, 1, 7)  # overdraft
    with pytest.raises(ValueError):
        ledger.transfer(0, 1, -1)


def test_nash_optimum_is_pareto_efficient():  # Proposition 1
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(20):
        u_r, u_d, feas = rng.random(200), rng.random(200), rng.random(200) < 0.8
        bm = benchmarks(u_r, u_d, 0.3, 0.3, feas)
        if bm["fir_empty"]:
            continue
        at_opt = benchmarks(u_r, u_d, 0.3, 0.3, feas, bm["nash_idx"])
        assert at_opt["pareto_dist"] == 0 and at_opt["nash_gap"] == 0
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("method", METHODS)
def test_simulation_invariants(method):
    res = run(Config(method=method, n_cells=3, n_epochs=48, seed=2))
    assert res["credit_total"] == pytest.approx(3 * 100)
    assert res["summary"]["triggers"] > 0
    for r in res["negotiations"]:
        if r["outcome"] == "agreement":
            assert not r["fir_empty"] and r["surplus_r"] >= 0 and r["surplus_d"] >= 0
            assert r["q"] <= r["donor_slack"]
        if r["outcome"] == "fallback":
            assert "u_r" not in r and "welfare" not in r


@pytest.mark.parametrize("seed", range(4))
def test_offers_only_peer_model_reduces_weight_error(seed):
    # weights, not absolute utility: accept/reject data only identifies U - r, so r and gamma trade off
    rng = np.random.default_rng(seed)
    prefs = sample_preferences(rng, 2, 1, "severe")
    me, opp = prefs[0, 0], prefs[1, 0]
    model = PeerModel()
    err0 = model.weight_mae(opp.w)
    for _ in range(20):
        feas = X[:, 0] <= rng.choice(Q_GRID[2:])
        outcome, _, _, hist = negotiate(Negotiator(me.utility("R"), me.r["R"], "R", model.utility("D")),
                                        Negotiator(opp.utility("D"), opp.r["D"], "D"), feas, 10)
        model.observe(hist, "D", outcome == "agreement")
    assert model.weight_mae(opp.w) < 0.75 * err0
