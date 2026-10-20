"""Day 12: the optical-navigation campaign engine (CPU backend; GPU rows are checked in test_gpu.py).
Run:  pytest tests/test_opnav_campaign.py -q"""
import numpy as np
import pytest

from tracker.camera import C_PIX, F_PIX, N_PIX
from tracker.catalog import load_catalog
from tracker.evaluate import CORRECT, NO_SOLUTION, WRONG
from tracker.identify import build_pair_table
from tracker.opnav_campaign import (ATT_TRACKER, ATT_TRUE, COLUMNS, CORRECT_FRACTION, EDGE_MARGIN_PX, FEW_POINTS,
                                    HEAD_MOUNT, NO_TRACKER, OK, run_batch, run_condition, score_position, solve_trial,
                                    trial_geometry)
from tracker.world import R_EQ, earth_in_view

col = {name: k for k, name in enumerate(COLUMNS)}
NEAR = 300_000.0          # a cheap distance for the end-to-end tests (small disc: fast CPU render)


@pytest.fixture(scope="module")
def cat():
    return load_catalog(5.5)


@pytest.fixture(scope="module")
def table(cat):
    return build_pair_table(cat.r)


@pytest.mark.parametrize("distance", [42_164.0, 100_000.0, 1_000_000.0])
@pytest.mark.parametrize("phase", [0.0, 90.0, 135.0])
def test_geometry_puts_the_whole_disc_in_the_frame_at_the_requested_phase(distance, phase):
    for trial in range(8):
        p, s, A, r_px, off_px = trial_geometry(seed=1, trial=trial, distance=distance, phase_deg=phase)
        assert np.isclose(np.linalg.norm(p), distance) and np.isclose(np.linalg.norm(s), 1.0)
        assert np.isclose(np.degrees(np.arccos(np.clip(p @ s / distance, -1, 1))), phase, atol=1e-4)   # phase angle at Earth (arccos is ill-conditioned near 0)
        visible, (u, v), rho, pr = earth_in_view(p, A)
        assert visible and np.isclose(pr, r_px)
        assert np.hypot(u - C_PIX, v - C_PIX) <= off_px + 1e-6 and off_px <= C_PIX - EDGE_MARGIN_PX - r_px + 1e-9
        assert np.linalg.det(A) > 0.999 and np.allclose(A @ A.T, np.eye(3))


def test_geometry_is_paired_across_cells():
    """Same seed and trial -> same direction to Earth, same Sun azimuth draw, same roll, at every distance and
    phase: the sweeps are paired comparisons (common random numbers), as on Day 5."""
    p1, s1, A1, *_ = trial_geometry(3, 7, 42_164.0, 90.0)
    p2, s2, A2, *_ = trial_geometry(3, 7, 400_000.0, 90.0)
    p3, s3, A3, *_ = trial_geometry(3, 7, 42_164.0, 45.0)
    assert np.allclose(p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2))
    assert np.allclose(A1, A3) and np.allclose(p1, p3)                 # the phase changes the Sun only
    assert not np.allclose(s1, s3)
    assert np.allclose(np.linalg.det(HEAD_MOUNT), 1.0) and np.allclose(HEAD_MOUNT @ HEAD_MOUNT.T, np.eye(3))
    assert np.isclose(HEAD_MOUNT[2] @ np.array([0, 0, 1.0]), 0.0)     # the head looks 90 deg away from Earth


def test_score_thresholds():
    p = np.array([100_000.0, 0.0, 0.0])
    assert score_position(None, p)[0] == NO_SOLUTION
    assert score_position(p * (1 + 0.5 * CORRECT_FRACTION), p)[0] == CORRECT
    assert score_position(p * (1 + 2 * CORRECT_FRACTION), p)[0] == WRONG
    outcome, total, rng, pct, cross = score_position(np.array([100_500.0, 0.0, 0.0]), p)
    assert outcome == CORRECT and np.isclose(rng, 500.0) and np.isclose(pct, 0.5) and cross < 1e-6


def test_solve_trial_without_an_attitude_is_an_honest_no_solution():
    res = solve_trial(np.zeros((N_PIX, N_PIX), np.float32), np.array([NEAR, 0, 0.0]), np.array([0, 1, 0.0]), None, 5.0)
    assert res[0] == NO_TRACKER and res[2] == NO_SOLUTION and np.isnan(res[3])


def test_rows_describe_each_trial(cat, table):
    trials = np.arange(3)
    rows = run_batch(trials, seed=1, distance=NEAR, phase=90.0, read_noise=5.0, catalog=cat, table=table)
    assert rows.shape == (6, len(COLUMNS))                            # tracker rows + true-attitude rows
    trk, tru = rows[rows[:, col["att_src"]] == ATT_TRACKER], rows[rows[:, col["att_src"]] == ATT_TRUE]
    assert trk[:, col["trial"]].tolist() == list(range(3)) == tru[:, col["trial"]].tolist()
    for k, t in enumerate(trials):
        p, s, A, r_px, off = trial_geometry(1, t, NEAR, 90.0)
        assert np.allclose(trk[k, [col["px"], col["py"], col["pz"]]], p)
        assert np.allclose(trk[k, [col["sx"], col["sy"], col["sz"]]], s)
        assert np.isclose(trk[k, col["pixel_radius"]], r_px)
    assert np.all(rows[:, col["reason"]] == OK) and np.all(rows[:, col["outcome"]] == CORRECT)
    assert np.all(trk[:, col["att_pointing"]] < 30) and np.all(trk[:, col["att_roll"]] < 120)   # the second head's tracker
    assert np.all(tru[:, col["att_total"]] == 0)
    assert np.all(rows[:, col["n_limb"]] >= 100) and np.all(np.abs(rows[:, col["range_pct"]]) < 0.5)
    assert np.all(np.isfinite(rows[:, col["range_sphere_km"]]))


def test_a_trial_does_not_depend_on_its_batch(cat, table):
    kw = dict(seed=2, distance=NEAR, phase=45.0, read_noise=5.0, catalog=cat, table=table, true_attitude=False)
    together = run_batch(np.array([1, 2]), **kw)
    alone = run_batch(np.array([2]), **kw)
    assert np.array_equal(together[1], alone[0], equal_nan=True)


def test_no_true_attitude_rows_when_not_asked(cat, table):
    rows = run_batch(np.array([0]), seed=2, distance=NEAR, phase=45.0, read_noise=10.0, catalog=cat, table=table,
                     true_attitude=False)
    assert rows.shape == (1, len(COLUMNS)) and rows[0, col["att_src"]] == ATT_TRACKER and rows[0, col["read_noise"]] == 10.0


def test_run_condition_resumes_without_repeating(cat, table, tmp_path):
    out = tmp_path / "cell.csv"
    kw = dict(seed=5, distance=NEAR, phase=90.0, read_noise=5.0, catalog=cat, table=table, batch=2, true_attitude=False)
    run_condition(out, n_trials=3, **kw)                  # "crash" after 3 trials
    run_condition(out, n_trials=4, **kw)                  # resume up to 4
    rows = np.loadtxt(out, delimiter=",", skiprows=1)
    assert sorted(rows[:, col["trial"]].tolist()) == list(range(4))
    direct = run_batch(np.arange(4), seed=5, distance=NEAR, phase=90.0, read_noise=5.0, catalog=cat, table=table,
                       true_attitude=False)
    order = np.argsort(rows[:, col["trial"]])
    assert np.allclose(rows[order], direct, equal_nan=True, rtol=1e-6)


def test_workers_give_the_same_rows(cat, table, tmp_path):
    """The Mac run uses several processes; each trial is a pure function of (seed, trial), so the rows are the same."""
    out1, out2 = tmp_path / "one.csv", tmp_path / "two.csv"
    kw = dict(n_trials=4, seed=6, distance=NEAR, phase=90.0, read_noise=5.0, catalog=cat, table=table, batch=4,
              true_attitude=False)
    run_condition(out1, workers=1, **kw)
    run_condition(out2, workers=2, **kw)
    a, b = (np.loadtxt(f, delimiter=",", skiprows=1) for f in (out1, out2))
    a, b = a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])]
    assert np.allclose(a, b, equal_nan=True, rtol=1e-6)
