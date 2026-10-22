"""Tenant-owned, pair-specific aggregation of peer models across cells (proposal Sec. 7, eqs. 15-17).

Each tenant s runs one aggregator per ordered pair s -> s'; only its own agents' schema-aligned theta records
and reliability summaries are uploaded. Raw offers, messages and KPIs never leave the cell.
"""
import numpy as np

THETA_DIM = 9
RULES = ("fedavg", "reliability", "median")


def aggregate(thetas, data_mass, rel_mass, rule):
    """Global theta: FedAvg (weights D_n), reliability (weights G_n, eq. 15) or coordinate-wise median.
    With sum G_n = 0 the reliability rule falls back to data weights instead of dividing by zero."""
    thetas = np.asarray(thetas, float)
    if thetas.ndim != 2 or thetas.shape[1] != THETA_DIM:
        raise ValueError(f"non-aligned peer-model schemas: shape {thetas.shape}")
    if rule not in RULES:
        raise ValueError(f"unknown aggregation rule {rule!r}")
    if rule == "median":
        return np.median(thetas, 0)
    w = np.asarray(rel_mass if rule == "reliability" and sum(rel_mass) > 0 else data_mass, float)
    return w @ thetas / w.sum()


def personalize(theta_global, theta_local, d_n, mu):
    """Eq. 17: beta = mu / (mu + D_n), so cold-start cells lean on the federation."""
    beta = mu / (mu + d_n)
    return beta * theta_global + (1 - beta) * theta_local
