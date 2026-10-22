import numpy as np
import pytest

from fedpeer.analysis import compare, holm, signflip_p


def test_signflip_exact_extremes():
    assert signflip_p(np.ones(10)) == pytest.approx(2 / 1024)  # all same sign: only the two extreme sign vectors
    assert signflip_p(np.array([1.0, -1.0])) == 1.0


def test_holm_matches_hand_computation():
    # sorted p 0.01, 0.02, 0.04 -> 0.03, 0.04, 0.04 (monotone)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_compare_orients_differences():
    p = {s: {"w_mae_mean": 0.1, "welfare": 1.2} for s in range(5)}
    b = {s: {"w_mae_mean": 0.2, "welfare": 1.0} for s in range(5)}
    assert compare(p, b, "w_mae_mean")["diff"] == pytest.approx(0.1)  # lower error -> positive
    assert compare(p, b, "welfare")["diff"] == pytest.approx(0.2)
