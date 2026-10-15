"""Scoring one trial: did the tracker get it right, get it wrong, or say it doesn't know?"""
import numpy as np

from tracker.solve import attitude_error

CORRECT, WRONG, NO_SOLUTION = 0, 1, 2
OUTCOMES = ("correct", "wrong", "no solution")
# Far above honest error (Day 3 worst correct frame 41.6", Day 2 roll p99 28") and far below
# a misidentification, which is off by degrees.
CORRECT_ARCSEC = 300.0


def score(sol, A_true):
    """identify() result + true attitude -> (outcome, total, pointing, roll, n_matched).
    Errors in arcsec (NaN when there is no solution); outcome is CORRECT, WRONG or NO_SOLUTION."""
    if sol is None:
        return NO_SOLUTION, np.nan, np.nan, np.nan, 0
    total, pointing, roll = attitude_error(sol.A, A_true)
    outcome = CORRECT if total < CORRECT_ARCSEC else WRONG
    return outcome, total, pointing, roll, len(sol.det)
