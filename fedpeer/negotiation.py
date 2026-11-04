"""Bilateral alternating offers: time-dependent concession, eq. 21 offer choice, AC_next acceptance, deadline T_max."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Negotiator:
    """Decision agent: sees only its own private utility, the validator's feasible set and its own peer model."""
    utility: np.ndarray                  # own U over all contracts (private)
    reservation: float
    role: str
    opp_utility: np.ndarray | None = None  # peer-model estimate of the counterpart's U; None = no model
    lam: float = 0.7                     # weight on OWN utility in eq. 21
    concession: float = 0.5              # e < 1: Boulware
    band: float = 0.05                   # iso-utility band around the concession target
    tradeoff: bool = False               # chip-aware offers (amendment 2): see offer()

    def target(self, k, T, feasible):
        top = self.utility[feasible].max()
        # min() guards float rounding at k = 0, which could otherwise exclude the top contract itself
        return min(top, self.reservation + (top - self.reservation) * (1 - (k / max(T - 1, 1)) ** (1 / self.concession)))

    def offer(self, k, T, feasible):
        """Best eq. 21 score among contracts near the target; None if no feasible contract beats the BATNA."""
        if not feasible.any() or self.utility[feasible].max() < self.reservation:
            return None
        tgt = self.target(k, T, feasible)
        above = feasible & (self.utility >= tgt)
        if self.tradeoff and self.opp_utility is not None:
            # trade-off move (logrolling): keep own utility at least at the target and give the counterpart the contract
            # it values most, i.e. concede on issues cheap for us and valuable for it; ties go to our own utility
            return int(np.where(above, self.opp_utility + 1e-9 * self.utility, -np.inf).argmax())
        cand = above & (self.utility <= tgt + self.band)
        if not cand.any():
            cand = above & (self.utility == self.utility[above].min())
        opp = 0 if self.opp_utility is None else self.opp_utility
        score = np.where(cand, self.lam * self.utility + (1 - self.lam) * opp, -np.inf)
        return int(score.argmax())

    def accepts(self, idx, k, T, feasible):
        """Deterministic check: feasible, at least the BATNA, and no worse than our own next offer (AC_next)."""
        if not feasible[idx] or self.utility[idx] < self.reservation:
            return False
        return k + 1 >= T or self.utility[idx] >= self.target(k + 1, T, feasible)


def negotiate(receiver, donor, feasible, t_max):
    """Receiver opens. Returns (outcome, agreed index or None, offers made, [(proposer role, index)])."""
    parties, history = (receiver, donor), []
    for k in range(t_max):
        proposer, responder = parties[k % 2], parties[1 - k % 2]
        idx = proposer.offer(k, t_max, feasible)
        if idx is None:
            return "no_agreement", None, k, history
        history.append((proposer.role, idx))
        if responder.accepts(idx, k, t_max, feasible):
            return "agreement", idx, k + 1, history
    return "timeout", None, t_max, history
