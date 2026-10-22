"""Ground-truth bargaining benchmarks, for evaluation only (proposal eqs. 13-16, Proposition 1)."""
import numpy as np


def pareto_mask(U):
    """Rows of U (k, 2) not weakly dominated with at least one strict improvement."""
    ge = (U[None, :, :] >= U[:, None, :]).all(2)  # ge[i, j]: j weakly dominates i
    gt = (U[None, :, :] > U[:, None, :]).any(2)
    return ~(ge & gt).any(1)


def benchmarks(u_r, u_d, r_r, r_d, feasible, agreed=None):
    """Nash-product optimum over F_IR, plus Nash gap and Pareto distance of the agreed contract index."""
    ir = feasible & (u_r >= r_r) & (u_d >= r_d)
    if not ir.any():
        return {"fir_empty": True}
    U = np.column_stack([u_r[ir], u_d[ir]])
    nash = (U[:, 0] - r_r) * (U[:, 1] - r_d)
    out = {"fir_empty": False, "nash_max": nash.max(), "nash_idx": int(np.flatnonzero(ir)[nash.argmax()]),
           "welfare_max": U.sum(1).max()}
    if agreed is not None:
        x = np.array([u_r[agreed], u_d[agreed]])
        out["nash_gap"] = nash.max() - (x[0] - r_r) * (x[1] - r_d)
        out["pareto_dist"] = np.linalg.norm(U[pareto_mask(U)] - x, axis=1).min()
    return out
