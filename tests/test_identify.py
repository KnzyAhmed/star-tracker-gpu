"""Day 3: star identification (pair table, voting, verify). Run:  pytest tests/test_identify.py -q"""
import numpy as np
import pytest

from tracker.catalog import load_catalog
from tracker.identify import PAIR_MAX_DEG, build_pair_table


def random_unit(rng, n):
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def brute_force_pairs(r, max_deg):
    """Every pair (i < j) closer than max_deg, by checking all N^2 of them."""
    ang = np.degrees(np.arccos(np.clip(r @ r.T, -1, 1)))
    i, j = np.nonzero(np.triu(ang < max_deg, k=1))
    return set(zip(i.tolist(), j.tolist()))


@pytest.fixture(scope="module")
def cat():
    return load_catalog(5.5)


# ---------------------------------------------------------------- build_pair_table

def test_table_is_sorted_by_angle(cat):
    t = build_pair_table(cat.r)
    assert np.all(np.diff(t.angle) >= 0)


def test_table_angles_are_correct_and_below_max(cat):
    t = build_pair_table(cat.r)
    cos = (cat.r[t.i] * cat.r[t.j]).sum(axis=1)
    assert np.allclose(np.cos(t.angle), cos, atol=1e-12)
    assert t.angle.max() < np.radians(PAIR_MAX_DEG)


def test_each_pair_once_with_i_less_than_j(cat):
    t = build_pair_table(cat.r)
    assert np.all(t.i < t.j)
    assert len(set(zip(t.i.tolist(), t.j.tolist()))) == len(t.angle)


def test_table_is_complete():
    """Same set of pairs as checking every pair by brute force."""
    rng = np.random.default_rng(0)
    r = random_unit(rng, 400)
    t = build_pair_table(r, max_sep_deg=28.3)
    assert set(zip(t.i.tolist(), t.j.tolist())) == brute_force_pairs(r, 28.3)


def test_fov_diagonal_covers_every_pair_on_the_sensor():
    """Corners of a 20 deg square sensor are 28.0 deg apart; the table must reach that far."""
    t = np.tan(np.radians(10))
    corner_to_corner = np.degrees(np.arccos((1 - 2 * t * t) / (1 + 2 * t * t)))
    assert 28.0 < corner_to_corner < PAIR_MAX_DEG


def test_real_catalog_size(cat):
    """2,847 stars; each has ~6 % of the sky within 28.3 deg -> ~2847 * 170 / 2 ~ 240k pairs."""
    assert 200_000 < len(build_pair_table(cat.r).angle) < 300_000
