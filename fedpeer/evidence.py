"""Typed language evidence, synthetic counterpart messages, mock extractor, fixed product gate and
outbound allowlist (proposal Secs. 6.2-6.3, 7.2; eqs. 9, 11)."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .contracts import ISSUES

KINDS = ("weight_order", "reservation_level", "kpi_need")
LEVELS = {"low": (0.0, 0.32), "mid": (0.32, 0.38), "high": (0.38, 1.0)}  # coarse bins; exact values never sent
PROFILES = ("honest", "noise", "contradictory", "strategic_lie", "rep_then_lie")
# g above this counts as "admitted" for the false-admission metric. Note: with reputation at its 0.5 prior
# (no verification yet) g <= 0.5, so nothing is admitted; compare mean gate on true vs false claims too.
ADMIT = 0.5


@dataclass(frozen=True)
class Claim:
    """One statement as said in a message; what is said need not be true."""
    kind: str
    a: str = ""          # weight_order: issue said to matter more
    b: str = ""          # weight_order: issue said to matter less
    level: str = ""      # reservation_level: low | mid | high
    value: float = 0.0   # kpi_need: PRB need claimed to be exceeded next epoch

    def key(self):
        return self.kind, self.a, self.b, self.level, round(self.value)


@dataclass(frozen=True)
class Evidence:
    """Extracted claim with extraction confidence c, schema validity v and provenance."""
    claim: Claim
    c: float
    v: float
    sender: int
    cell: int
    t: int
    event_id: str


@dataclass
class Message:
    sender: int
    cell: int
    t: int
    role: str
    msg_id: str
    claims: tuple        # as stated in the text, so observable; the sender's true preferences are not
    text: str
    clarity: float = 1.0  # < 1 for ambiguous phrasing (extraction-noise profile)


def valid(claim):
    """Schema validity v."""
    if claim.kind == "weight_order":
        return claim.a in ISSUES and claim.b in ISSUES and claim.a != claim.b
    if claim.kind == "reservation_level":
        return claim.level in LEVELS
    return claim.kind == "kpi_need" and claim.value > 0


def level_of(r):
    return next(k for k, (lo, hi) in LEVELS.items() if lo <= r < hi)


# ---- outbound allowlist (Sec. 7.2): permitted message types only, never exact weights or reservation values
ALLOWED_TYPES = ("offer", "accept", "reject")
CLAIM_FIELDS = {"weight_order": {"kind", "a", "b"}, "reservation_level": {"kind", "level"},
                "kpi_need": {"kind", "value"}}


def screen(payload):
    """Pass a drafted outbound message through unchanged, or raise ValueError on any non-allowlisted content."""
    if set(payload) - {"type", "offer", "claims"} or payload.get("type") not in ALLOWED_TYPES:
        raise ValueError(f"message not allowlisted: {sorted(payload)} type={payload.get('type')!r}")
    for c in payload.get("claims", []):
        if c.get("kind") not in CLAIM_FIELDS or set(c) - CLAIM_FIELDS[c["kind"]]:
            raise ValueError(f"claim not allowlisted: {c}")
        if c["kind"] == "reservation_level" and c.get("level") not in LEVELS:
            raise ValueError(f"reservation must be a coarse level, got {c.get('level')!r}")
    return payload


# ---- synthetic counterpart messages (ground truth stays with the evaluator)
NAMES = {"q": "the number of PRBs", "d": "the lease duration", "kappa": "the credit price"}
TEMPLATES = {
    "weight_order": ["{A} matters more to us than {B}.", "For us, {B} is less important than {A}.",
                     "We care about {A} more than about {B}."],
    "weight_order_vague": ["{A}... well, maybe more than {B}, hard to say right now.",
                           "Honestly {B} and {A} both matter, {A} a bit more perhaps."],
    "low": ["We are quite flexible and can accept modest terms."],
    "mid": ["We need a reasonable deal, nothing extreme."],
    "high": ["Our walk-away point is high; we need a strong deal."],
    "kpi_need": ["We expect our PRB need next epoch to exceed {X} PRBs.",
                 "Our forecast says we will need more than {X} PRBs next epoch."],
}


def compose(rng, profile, prefs, role, next_need, sender, cell, t, msg_id, lying):
    """Draft a message for a sender with the given profile; `lying` is a consistent strategic lie
    (reversed weight order, 'high' reservation, inflated KPI forecast)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    w = -prefs.w if lying else prefs.w  # negation flips every pairwise order; reversing the array would not
    i, j = rng.choice(3, 2, replace=False)
    hi, lo = (i, j) if w[i] >= w[j] else (j, i)
    if profile == "contradictory" and rng.random() < 0.5:
        hi, lo = lo, hi
    claims = [{"kind": "weight_order", "a": ISSUES[hi], "b": ISSUES[lo]}]
    if rng.random() < 0.5:
        level = "high" if lying else level_of(prefs.r[role])
        if profile == "contradictory" and rng.random() < 0.5:
            level = str(rng.choice(list(LEVELS)))
        claims.append({"kind": "reservation_level", "level": level})
    if role == "R":
        claims.append({"kind": "kpi_need", "value": float(round(next_need * (1.4 if lying else rng.uniform(0.9, 0.98))))})
    claims = screen({"type": "offer", "claims": claims})["claims"]

    vague = profile == "noise"
    parts = []
    for c in claims:
        if c["kind"] == "weight_order":
            tpl = rng.choice(TEMPLATES["weight_order_vague" if vague else "weight_order"])
            parts.append(str(tpl).format(A=NAMES[c["a"]], B=NAMES[c["b"]]))
        elif c["kind"] == "reservation_level":
            parts.append(str(rng.choice(TEMPLATES[c["level"]])))
        else:
            parts.append(str(rng.choice(TEMPLATES["kpi_need"])).format(X=int(c["value"])))
    text = " ".join(parts)
    return Message(sender, cell, t, role, msg_id, tuple(Claim(**c) for c in claims), text[0].upper() + text[1:],
                   0.3 if vague else 1.0)


class MockExtractor:
    """Deterministic LLM stand-in: misses or corrupts stated claims more often as clarity falls,
    and reports lower confidence on corrupted ones."""

    def __init__(self, rng, noise=0.05):
        self.rng, self.noise = rng, noise

    def __call__(self, msg):
        p_err = min(0.9, self.noise * (1 + 10 * (1 - msg.clarity)))
        out = []
        for k, cl in enumerate(msg.claims):
            u = self.rng.random()
            if u < p_err / 2:
                continue  # missed
            if u < p_err:
                cl, c = _corrupt(cl, self.rng), self.rng.uniform(0.3, 0.7)
            else:
                c = self.rng.uniform(0.75, 1.0)
            out.append(Evidence(cl, c, float(valid(cl)), msg.sender, msg.cell, msg.t, f"{msg.msg_id}:{k}"))
        return out


def _corrupt(cl, rng):
    if cl.kind == "weight_order":
        return Claim("weight_order", cl.b, cl.a)
    if cl.kind == "reservation_level":
        return Claim("reservation_level", level=str(rng.choice([lv for lv in LEVELS if lv != cl.level])))
    return Claim("kpi_need", value=float(round(cl.value * rng.uniform(0.7, 1.3))))


# ---- reliability
def consistency(claim, history):
    """k: Laplace-smoothed agreement with the same sender's earlier claims (1 with no history)."""
    if claim.kind == "weight_order":
        same = sum(h.kind == claim.kind and (h.a, h.b) == (claim.a, claim.b) for h in history)
        opp = sum(h.kind == claim.kind and (h.a, h.b) == (claim.b, claim.a) for h in history)
    elif claim.kind == "reservation_level":
        same = sum(h.kind == claim.kind and h.level == claim.level for h in history)
        opp = sum(h.kind == claim.kind and h.level != claim.level for h in history)
    else:
        return 1.0
    return (1 + same) / (1 + same + opp)


def gate(c, v, k, a, r):
    """Fixed product gate g = c v k a r (eq. 11); a = 1 is neutral for claims not verifiable now, not 'true'."""
    if not all(0 <= x <= 1 for x in (c, v, k, a, r)):
        raise ValueError(f"gate factors must lie in [0, 1]: {(c, v, k, a, r)}")
    return c * v * k * a * r


def is_true(claim, prefs, role):
    """Evaluator only: does a preference claim match the sender's private ground truth?"""
    if claim.kind == "weight_order":
        return prefs.w[ISSUES.index(claim.a)] > prefs.w[ISSUES.index(claim.b)]
    return level_of(prefs.r[role]) == claim.level


def match_counts(stated, extracted):
    """(tp, fp, fn) of extracted claims against stated claims, for extraction micro-F1."""
    s, e = Counter(c.key() for c in stated), Counter(ev.claim.key() for ev in extracted)
    tp = sum((s & e).values())
    return tp, sum(e.values()) - tp, sum(s.values()) - tp


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
