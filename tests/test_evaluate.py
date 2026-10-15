"""Day 5: scoring one trial. Run:  pytest tests/test_evaluate.py -q"""
import numpy as np

from tracker.camera import attitude_from_pointing
from tracker.evaluate import CORRECT, CORRECT_ARCSEC, NO_SOLUTION, OUTCOMES, WRONG, score
from tracker.identify import Solution


def sol_at(A, n=7):
    return Solution(A=A, det=np.arange(n), cat=np.arange(n))


def test_outcome_codes():
    assert (CORRECT, WRONG, NO_SOLUTION) == (0, 1, 2)
    assert OUTCOMES == ("correct", "wrong", "no solution")


def test_no_solution():
    outcome, total, pointing, roll, n_matched = score(None, np.eye(3))
    assert outcome == NO_SOLUTION and n_matched == 0
    assert np.isnan(total) and np.isnan(pointing) and np.isnan(roll)


def test_correct_and_wrong_split_at_the_threshold():
    A = attitude_from_pointing(83.8, -5.4, 30.0)
    just_in = attitude_from_pointing(83.8, -5.4, 30.0 + (CORRECT_ARCSEC - 1) / 3600)   # pure roll
    just_out = attitude_from_pointing(83.8, -5.4, 30.0 + (CORRECT_ARCSEC + 1) / 3600)
    out = score(sol_at(just_in), A)
    assert out[0] == CORRECT and abs(out[1] - (CORRECT_ARCSEC - 1)) < 1e-6 and out[4] == 7
    assert score(sol_at(just_out), A)[0] == WRONG
    assert score(sol_at(attitude_from_pointing(200.0, 10.0, 0.0)), A)[0] == WRONG
