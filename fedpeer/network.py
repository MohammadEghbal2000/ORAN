"""Cells, tenant demand traces, static slicing and the credit ledger (proposal eqs. 1, 2, 4).

shortcut: SLA is a PRB-need model (satisfied iff allocation >= need), no scheduler or channel model;
upgrade to a latency/throughput predictor if reviewers ask for radio realism.
"""
import numpy as np

TENANTS = ("URLLC", "eMBB", "mMTC")
STATIC_SHARE = np.array([0.3, 0.5, 0.2])   # B0 static slicing
TENANT_SHIFT_H = np.array([-3.0, 0.0, 4.0])  # tenants peak at different hours, so donors exist
REGIMES = ("residential", "office", "stadium", "surge")
EPOCH_MIN = 15


def _profile(regime, h):
    if regime == "residential":
        return 0.45 + 0.6 * np.exp(-((h - 20) / 3) ** 2)
    if regime == "office":
        return 0.45 + 0.6 * np.exp(-((h - 13) / 3) ** 2)
    if regime == "stadium":
        return 0.4 + 0.75 * ((h >= 18) & (h < 21))
    return 0.6 + 0.15 * np.sin(2 * np.pi * h / 24)  # surge: flat, random surges added below


def demand_traces(rng, n_cells, n_epochs, budget):
    """PRB need[n, s, t] per cell, tenant and epoch, plus the traffic regime of each cell."""
    regimes = rng.choice(REGIMES, size=n_cells)
    h = (np.arange(n_epochs) * EPOCH_MIN / 60) % 24
    need = np.empty((n_cells, len(TENANTS), n_epochs))
    for n, reg in enumerate(regimes):
        for s in range(len(TENANTS)):
            base = STATIC_SHARE[s] * budget * _profile(reg, (h + TENANT_SHIFT_H[s]) % 24)
            need[n, s] = base * rng.lognormal(0, 0.1, n_epochs)
        for t in np.flatnonzero(rng.random(n_epochs) < (0.03 if reg == "surge" else 0.01)):
            need[n, rng.integers(len(TENANTS)), t:t + 4] *= 1.5
    return need, regimes


class Ledger:
    """Tenant-level credit ledger; transfers are atomic and conserve the total (eq. 4)."""

    def __init__(self, n_tenants, initial):
        self.balance = np.full(n_tenants, float(initial))

    def transfer(self, receiver, donor, kappa):
        if not 0 <= kappa <= self.balance[receiver]:
            raise ValueError(f"invalid credit transfer {kappa} with receiver balance {self.balance[receiver]}")
        self.balance[receiver] -= kappa
        self.balance[donor] += kappa
