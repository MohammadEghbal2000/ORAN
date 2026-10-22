"""KPI-claim verifier and decayed Beta reputation (proposal Sec. 6.4, eq. 12)."""


def verify(claimed, actual, tol=0.05):
    """z = 1 if 'our need next epoch will exceed X' held within a relative tolerance."""
    return float(actual >= claimed * (1 - tol))


class BetaReputation:
    """Claim correctness under the verifier, per key (cell, observer, counterpart); not intrinsic trust.
    Each event id counts once, so duplicate verifier outputs cannot inflate the evidence."""

    def __init__(self, delta=0.95):
        if not 0 < delta <= 1:
            raise ValueError("delta must lie in (0, 1]")
        self.delta, self.counts, self.seen = delta, {}, set()

    def update(self, key, z, event_id):
        if event_id in self.seen:
            return False
        self.seen.add(event_id)
        a, b = self.counts.get(key, (0.0, 0.0))
        self.counts[key] = (self.delta * a + z, self.delta * b + 1 - z)
        return True

    def score(self, key):
        a, b = self.counts.get(key, (0.0, 0.0))
        return (1 + a) / (2 + a + b)

    def pooled(self, observer, counterpart):
        """Cross-cell aggregated reputation (eq. 18): counts summed over cells, prior counted once."""
        ab = [v for (_, o, c), v in self.counts.items() if (o, c) == (observer, counterpart)]
        a, b = sum(x for x, _ in ab), sum(y for _, y in ab)
        return (1 + a) / (2 + a + b)
