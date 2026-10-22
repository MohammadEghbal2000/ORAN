import json
from types import SimpleNamespace

import numpy as np
import pytest

from fedpeer.contracts import Q_GRID, X, sample_preferences
from fedpeer.evidence import Claim, MockExtractor, compose, consistency, gate, match_counts, screen
from fedpeer.llm import LLMExtractor
from fedpeer.negotiation import Negotiator, negotiate
from fedpeer.peer_model import PeerModel
from fedpeer.reputation import BetaReputation, verify


def test_gate_bounds_and_neutral_verification():
    assert gate(0.9, 1, 0.8, 1.0, 0.5) == pytest.approx(0.9 * 0.8 * 0.5)  # a = 1 leaves the product unchanged
    assert gate(1, 0, 1, 1, 1) == 0  # invalid schema blocks everything
    for bad in [(1.1, 1, 1, 1, 1), (1, 1, -0.1, 1, 1)]:
        with pytest.raises(ValueError):
            gate(*bad)


def test_reputation_formula_decay_and_dedupe():
    rep = BetaReputation(delta=1.0)
    assert rep.score("k") == 0.5
    rep.update("k", 1, "e1")
    assert rep.score("k") == pytest.approx(2 / 3)
    assert not rep.update("k", 1, "e1")  # duplicate event is ignored
    assert rep.score("k") == pytest.approx(2 / 3)

    rep = BetaReputation(delta=0.5)
    rep.update("k", 0, "a")
    rep.update("k", 0, "b")
    rep.update("k", 1, "c")  # a = 1, b = 0.5 * (0.5 * 1 + 1) = 0.75
    assert rep.score("k") == pytest.approx(2 / 3.75)
    assert verify(100, 96) == 1 and verify(100, 90) == 0


def test_allowlist_blocks_forbidden_content():
    ok = {"type": "offer", "claims": [{"kind": "weight_order", "a": "q", "b": "d"}]}
    assert screen(ok) is ok
    for bad in [{"type": "offer", "weights": [0.2, 0.3, 0.5]},
                {"type": "offer", "claims": [{"kind": "reservation_level", "level": 0.31}]},
                {"type": "offer", "claims": [{"kind": "weight_order", "a": "q", "b": "d", "value": 0.4}]},
                {"type": "offer", "claims": [{"kind": "exact_weight", "a": "q"}]},
                {"type": "free_text"}]:
        with pytest.raises(ValueError):
            screen(bad)


def test_consistency_drops_on_contradiction():
    c = Claim("weight_order", "q", "d")
    assert consistency(c, []) == 1
    assert consistency(c, [Claim("weight_order", "d", "q")] * 3) == pytest.approx(1 / 4)


def test_mock_extractor_is_exact_without_noise():
    rng = np.random.default_rng(0)
    prefs = sample_preferences(rng, 1, 1, "iid")[0, 0]
    msg = compose(rng, "honest", prefs, "R", 40.0, 0, 0, 0, "m", False)
    assert match_counts(msg.claims, MockExtractor(rng, noise=0)(msg)) == (len(msg.claims), 0, 0)


class FakeClient:
    """Stands in for openai.OpenAI; counts calls so the cache can be checked."""

    def __init__(self, content):
        self.calls, self.content = 0, content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        assert kw["response_format"]["type"] == "json_schema"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def test_llm_extractor_parses_and_caches(tmp_path):
    claims = {"claims": [{"kind": "weight_order", "a": "d", "b": "kappa", "level": "", "value": 0, "confidence": 0.9},
                         {"kind": "kpi_need", "a": "", "b": "", "level": "", "value": 42, "confidence": 1.5}]}
    client = FakeClient(json.dumps(claims))
    ext = LLMExtractor("fake", client=client, cache_dir=tmp_path)
    msg = SimpleNamespace(text="Duration matters more than credits. Need above 42 PRBs.", sender=1, cell=0, t=3,
                          msg_id="m")
    evs = ext(msg)
    assert [e.claim for e in evs] == [Claim("weight_order", "d", "kappa"), Claim("kpi_need", value=42.0)]
    assert evs[1].c == 1.0 and all(e.v == 1 for e in evs)
    ext(msg)
    assert client.calls == 1 and ext.usage["cache_hits"] == 1

    assert LLMExtractor("fake", client=FakeClient("not json"), cache_dir=tmp_path / "x")(msg) == []


def _train(seed, profile, lying, rep_score, n_neg=12):
    """Same offers for both models; one also learns from gated language.
    Returns [B2 weight MAE, B3 weight MAE, B2 rank agreement, B3 rank agreement]."""
    rng = np.random.default_rng(seed)
    prefs = sample_preferences(rng, 2, 1, "severe")
    me, opp = prefs[0, 0], prefs[1, 0]
    b2, b3 = PeerModel(), PeerModel()
    ext, said = MockExtractor(np.random.default_rng(seed)), []
    for k in range(n_neg):
        feas = X[:, 0] <= rng.choice(Q_GRID[2:])
        outcome, _, _, hist = negotiate(Negotiator(me.utility("R"), me.r["R"], "R", b2.utility("D")),
                                        Negotiator(opp.utility("D"), opp.r["D"], "D"), feas, 10)
        lang = []
        for j, (role, _) in enumerate(hist):
            if role != "D":
                continue
            msg = compose(rng, profile, opp, "D", 30.0, 1, 0, k, f"{k}:{j}", lying)
            for ev in ext(msg):
                if ev.claim.kind != "kpi_need":
                    lang.append((ev.claim, "D", gate(ev.c, ev.v, consistency(ev.claim, said), 1.0, rep_score)))
                    said.append(ev.claim)
        b2.observe(hist, "D", outcome == "agreement")
        b3.observe(hist, "D", outcome == "agreement", lang)
    return [b2.weight_mae(opp.w), b3.weight_mae(opp.w), b2.rank_agreement(opp.w), b3.rank_agreement(opp.w)]


def test_honest_language_lowers_model_error():
    # ordinal claims mainly fix the weight ranking; magnitudes still come from offers
    e = np.array([_train(s, "honest", False, 0.5) for s in range(6)]).mean(0)
    assert e[1] < e[0] and e[3] > e[2] + 0.1


def test_lies_from_low_reputation_sender_do_little_harm():
    low = np.array([_train(s, "strategic_lie", True, 1 / 12) for s in range(6)]).mean(0)  # ~10 failed checks
    high = np.array([_train(s, "strategic_lie", True, 0.9) for s in range(6)]).mean(0)
    assert low[1] < low[0] + 0.03 and low[3] > low[2] - 0.15
    assert high[1] > low[1] + 0.03  # the gate, not luck, is what limits the damage
