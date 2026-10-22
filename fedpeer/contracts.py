"""Contract space x = (q, d, kappa) and private additive utilities (proposal eqs. 3, 5, 6)."""
import itertools
from dataclasses import dataclass

import numpy as np

Q_GRID = np.arange(5, 55, 5)      # reassigned PRBs
D_GRID = np.array([1, 2, 4, 8])   # duration in epochs
K_GRID = np.arange(0, 22, 2)      # internal credits paid by receiver R to donor D

X = np.array(list(itertools.product(Q_GRID, D_GRID, K_GRID)), dtype=float)  # (M, 3) raw contracts
XN = (X - X.min(0)) / (X.max(0) - X.min(0))                                 # per-issue normalized to [0, 1]
ROLES = ("R", "D")
HETEROGENEITY = ("iid", "clustered", "severe")


def oriented(xn, role):
    """Issue values oriented so 1 is best for the role: R wants more q, longer d, fewer credits; D the opposite."""
    return np.abs((np.array([0, 0, 1]) if role == "R" else np.array([1, 1, 0])) - xn)


@dataclass
class Preferences:
    """Private ground truth of one tenant in one cell; only its owner and the evaluator may read it."""
    w: np.ndarray       # issue weights on the simplex
    gamma: np.ndarray   # issue-evaluation shapes e_h(x) = x^gamma_h
    r: dict             # reservation (BATNA) utility per role

    def utility(self, role):
        return (self.w * oriented(XN, role) ** self.gamma).sum(1)


def sample_preferences(rng, n_tenants, n_cells, heterogeneity):
    """Synthetic preferences per (tenant, cell); heterogeneity sets preference non-IID only, not traffic."""
    if heterogeneity not in HETEROGENEITY:
        raise ValueError(f"unknown heterogeneity {heterogeneity!r}")
    proto_w = rng.dirichlet(np.full(3, 2.0), size=(n_tenants, 2))  # two clusters per tenant
    proto_g = np.exp(rng.normal(0, 0.4, size=(n_tenants, 2, 3)))
    prefs = {}
    for s in range(n_tenants):
        for n in range(n_cells):
            if heterogeneity == "severe":
                w, g = rng.dirichlet(np.ones(3)), np.exp(rng.normal(0, 0.4, 3))
            else:
                c = n % 2 if heterogeneity == "clustered" else 0
                w = rng.dirichlet(proto_w[s, c] * 50)
                g = proto_g[s, c] * np.exp(rng.normal(0, 0.05, 3))
            prefs[s, n] = Preferences(w, g, {"R": rng.uniform(0.25, 0.45), "D": rng.uniform(0.25, 0.45)})
    return prefs
