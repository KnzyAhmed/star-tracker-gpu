"""Day 5: the campaign engine (CPU backend; the GPU backend is checked in test_gpu.py).
Run:  pytest tests/test_campaign.py -q"""
import numpy as np
import pytest

from tracker.campaign import COLUMNS, run_batch, run_condition, trial_attitude
from tracker.camera import pointing_from_attitude, project
from tracker.catalog import load_catalog
from tracker.evaluate import CORRECT
from tracker.identify import build_pair_table

col = {name: k for k, name in enumerate(COLUMNS)}


@pytest.fixture(scope="module")
def cat():
    return load_catalog(5.5)


@pytest.fixture(scope="module")
def table(cat):
    return build_pair_table(cat.r)


def test_rows_describe_each_trial(cat, table):
    trials = np.arange(6)
    rows = run_batch(trials, seed=1, read_noise=5.0, hot_pixels=0, catalog=cat, table=table)
    assert rows.shape == (6, len(COLUMNS))
    assert rows[:, col["trial"]].tolist() == list(range(6))
    for k, t in enumerate(trials):
        A = trial_attitude(1, t)
        ra, dec, roll = pointing_from_attitude(A)
        assert np.allclose(rows[k, [col["ra"], col["dec"], col["roll"]]], [ra, dec, roll])
        assert rows[k, col["n_in_view"]] == project(A, cat.r)[2].sum()
    assert np.all(rows[:, col["outcome"]] == CORRECT)               # baseline: all solved
    assert np.all(rows[:, col["pointing"]] < 20)


def test_a_trial_does_not_depend_on_its_batch(cat, table):
    """Trial 4 alone == trial 4 inside a batch: needed for resuming and for CPU == GPU."""
    kw = dict(seed=2, read_noise=5.0, hot_pixels=3, catalog=cat, table=table)
    together = run_batch(np.array([3, 4]), **kw)
    alone = run_batch(np.array([4]), **kw)
    assert np.array_equal(together[1], alone[0], equal_nan=True)


def test_conditions_share_the_same_skies(cat, table):
    """Same seed + trial -> same attitude for every noise level (paired comparisons)."""
    a = run_batch(np.arange(3), seed=3, read_noise=5.0, hot_pixels=0, catalog=cat, table=table)
    b = run_batch(np.arange(3), seed=3, read_noise=10.0, hot_pixels=0, catalog=cat, table=table)
    for name in ("ra", "dec", "roll", "n_in_view"):
        assert np.array_equal(a[:, col[name]], b[:, col[name]])
    assert np.all(b[:, col["n_detected"]] <= a[:, col["n_detected"]])   # more noise, fewer stars


def test_hot_pixels_add_detections(cat, table):
    kw = dict(seed=4, read_noise=5.0, catalog=cat, table=table)
    clean = run_batch(np.arange(4), hot_pixels=0, **kw)
    hot = run_batch(np.arange(4), hot_pixels=10, **kw)
    extra = hot[:, col["n_detected"]] - clean[:, col["n_detected"]]
    assert np.all(extra >= 8) and np.all(extra <= 10)    # a hot pixel can land on a star


def test_run_condition_resumes_without_repeating(cat, table, tmp_path):
    out = tmp_path / "c.csv"
    kw = dict(seed=5, read_noise=5.0, hot_pixels=0, catalog=cat, table=table, batch=3)
    run_condition(out, n_trials=4, **kw)                  # "crash" after 4 trials
    run_condition(out, n_trials=7, **kw)                  # resume up to 7
    rows = np.loadtxt(out, delimiter=",", skiprows=1)
    assert sorted(rows[:, col["trial"]].tolist()) == list(range(7))
    direct = run_batch(np.arange(7), seed=5, read_noise=5.0, hot_pixels=0, catalog=cat, table=table)
    order = np.argsort(rows[:, col["trial"]])
    assert np.allclose(rows[order], direct, equal_nan=True, rtol=1e-6)    # CSV keeps 7 digits
