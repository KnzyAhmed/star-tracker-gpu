"""Day 3: star identification (pair table, voting, verify). Run:  pytest tests/test_identify.py -q"""
import numpy as np
import pytest

from tracker.camera import project, random_attitude
from tracker.catalog import load_catalog
from tracker.identify import ARCSEC, PAIR_MAX_DEG, angle_between, build_pair_table, vote


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


@pytest.fixture(scope="module")
def table(cat):
    return build_pair_table(cat.r)


def perturb(rng, b, sigma_arcsec):
    b = b + rng.normal(scale=np.radians(sigma_arcsec / 3600), size=b.shape)
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def visible_stars(cat, rng):
    """A random attitude and the exact camera-frame vectors of the catalog stars in view."""
    A = random_attitude(rng)
    idx = np.flatnonzero(project(A, cat.r)[2])
    return A, idx, cat.r[idx] @ A.T


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


# ---------------------------------------------------------------- vote

def same_or_unresolvable(cat, best, idx, eps_arcsec=60):
    """Right star, or its partner in a double closer than eps (e.g. Beta Mon A/B, 9.6 arcsec
    apart): pair angles cannot tell those apart, and either choice moves the attitude < eps."""
    return angle_between(cat.r[best], cat.r[idx]) <= eps_arcsec * ARCSEC


def test_vote_noise_free_names_every_star(cat, table):
    rng = np.random.default_rng(1)
    for _ in range(10):
        A, idx, b = visible_stars(cat, rng)
        best, votes = vote(b, table, len(cat), eps_arcsec=60)
        assert best.shape == votes.shape == (len(b),)
        assert same_or_unresolvable(cat, best, idx).all()


def test_vote_true_star_gets_a_vote_from_every_pair(cat, table):
    """Noise-free: star k pairs with the other N-1 stars, and each of those pairs is in the table."""
    rng = np.random.default_rng(2)
    A, idx, b = visible_stars(cat, rng)
    best, votes = vote(b, table, len(cat), eps_arcsec=60)
    assert np.all(votes >= len(b) - 1)


def test_vote_with_realistic_noise(cat, table):
    """Day 2 measured ~7 arcsec per star; nearly every star must still be named."""
    rng = np.random.default_rng(3)
    right = total = 0
    for _ in range(20):
        A, idx, b = visible_stars(cat, rng)
        best, _ = vote(perturb(rng, b, 7), table, len(cat), eps_arcsec=60)
        right += same_or_unresolvable(cat, best, idx).sum()
        total += len(idx)
    assert right / total > 0.97


def test_vote_survives_a_fake_star(cat, table):
    """A hot pixel is a dot with no catalog star. The real stars must still be named, and the
    fake's best guess must score below every real star (measured: true in 98.5 % of frames;
    the verify step in identify() is what finally rejects it)."""
    rng = np.random.default_rng(4)
    A, idx, b = visible_stars(cat, rng)
    fake = np.array([[0.05, -0.03, 1.0]]) / np.linalg.norm([0.05, -0.03, 1.0])
    best, votes = vote(np.vstack([b, fake]), table, len(cat), eps_arcsec=60)
    assert same_or_unresolvable(cat, best[:-1], idx).all()
    assert votes[-1] < votes[:-1].min()


def test_vote_fewer_than_two_stars(cat, table):
    best, votes = vote(np.array([[0.0, 0.0, 1.0]]), table, len(cat), eps_arcsec=60)
    assert best.tolist() == [-1] and votes.tolist() == [0]
    best, votes = vote(np.zeros((0, 3)), table, len(cat), eps_arcsec=60)
    assert len(best) == len(votes) == 0
