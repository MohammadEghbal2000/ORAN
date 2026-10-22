import numpy as np
import pytest

from fedpeer.federation import aggregate, personalize
from fedpeer.peer_model import PeerModel
from fedpeer.reputation import BetaReputation
from fedpeer.simulator import Config, run


def _theta(w0):
    return np.array([w0, (1 - w0) / 2, (1 - w0) / 2, 1, 1, 1, 0.3, 0.4, 0.1])


def test_aggregation_rules():
    thetas = [_theta(0.2), _theta(0.6)]
    assert aggregate(thetas, [1, 3], [0, 0], "fedavg")[0] == pytest.approx(0.5)
    assert aggregate(thetas, [1, 3], [3, 1], "reliability")[0] == pytest.approx(0.3)
    assert aggregate(thetas, [1, 3], [0, 0], "reliability")[0] == pytest.approx(0.5)  # sum G = 0: data weights
    assert aggregate([_theta(0.2), _theta(0.3), _theta(0.9)], [1, 1, 1], [1, 1, 1], "median")[0] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        aggregate([np.ones(7), np.ones(7)], [1, 1], [1, 1], "fedavg")  # non-aligned schema
    with pytest.raises(ValueError):
        aggregate(thetas, [1, 1], [1, 1], "mean_of_everything")


def test_personalization_schedule():
    g, loc = _theta(0.6), _theta(0.2)
    assert personalize(g, loc, 0, 10) == pytest.approx(g)  # cold start: pure transfer
    assert personalize(g, loc, 10, 10)[0] == pytest.approx(0.4)
    assert personalize(g, loc, 1e9, 10)[0] == pytest.approx(0.2, abs=1e-6)


def test_pooled_reputation_counts_prior_once():
    rep = BetaReputation(delta=1.0)
    rep.update((0, 1, 2), 1, "a")
    rep.update((5, 1, 2), 1, "b")
    rep.update((5, 2, 1), 0, "c")  # other direction, not pooled in
    assert rep.pooled(1, 2) == pytest.approx(3 / 4)


def test_theta_load_roundtrip_and_simplex_repair():
    m = PeerModel()
    m.z = m.z + np.linspace(-0.5, 0.5, 9)
    th = m.theta
    m2 = PeerModel()
    m2.load(th)
    assert m2.theta == pytest.approx(th)
    m2.load(np.array([0.5, 0.5, 0.5, 1, 1, 1, 0.3, 0.3, 0.1]))  # e.g. a median off the simplex
    assert m2.theta[:3].sum() == pytest.approx(1)


def test_federation_transfers_to_cold_start_cells():
    """H1 core: with shared tenant priorities, a fresh cell's first model is already informed."""
    res = {m: run(Config(method=m, n_cells=6, n_epochs=48, seed=3, heterogeneity="iid"))["summary"]
           for m in ("B3", "P")}
    assert res["P"]["w_mae_cold"] < res["B3"]["w_mae_cold"]
    assert res["P"]["fl_bytes"] > 0 and res["B3"]["fl_bytes"] == 0
