"""Day 2: attitude solve (Wahba / SVD) and attitude error. Run:  pytest tests/test_solve.py -q"""
import numpy as np
import pytest

from tracker.camera import random_attitude
from tracker.solve import solve_attitude


def is_rotation(A):
    return np.allclose(A @ A.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(A), 1.0)


def random_unit(rng, n):
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def perturb(rng, b, sigma_rad):
    """Nudge each unit vector by a small random angle (about sigma_rad per axis), renormalise."""
    b = b + rng.normal(scale=sigma_rad, size=b.shape)
    return b / np.linalg.norm(b, axis=1, keepdims=True)


# ---------------------------------------------------------------- solve_attitude

def test_noise_free_recovers_attitude_exactly():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = random_attitude(rng)
        r = random_unit(rng, 10)
        b = r @ A.T
        A_est = solve_attitude(b, r)
        assert A_est.shape == (3, 3)
        assert np.allclose(A_est, A, atol=1e-12)


def test_two_stars_are_enough():
    """Two non-parallel vectors fix a rotation uniquely, even though B has rank 2."""
    rng = np.random.default_rng(1)
    A = random_attitude(rng)
    r = random_unit(rng, 2)
    A_est = solve_attitude(r @ A.T, r)
    assert np.allclose(A_est, A, atol=1e-10)


def test_result_is_a_rotation_even_with_noise():
    rng = np.random.default_rng(2)
    A = random_attitude(rng)
    r = random_unit(rng, 5)
    b = perturb(rng, r @ A.T, 1e-3)
    assert is_rotation(solve_attitude(b, r))


def test_never_returns_a_reflection():
    """If b is a MIRROR image of r, the best orthogonal matrix is a reflection (det -1).
    The diag(1, 1, det U det V) fix must still return a proper rotation."""
    rng = np.random.default_rng(3)
    r = random_unit(rng, 6)
    b = r * np.array([1.0, 1.0, -1.0])           # mirror through the xy-plane
    A_est = solve_attitude(b, r)
    assert is_rotation(A_est)


def test_zero_weight_ignores_an_outlier():
    rng = np.random.default_rng(4)
    A = random_attitude(rng)
    r = random_unit(rng, 6)
    b = r @ A.T
    b[0] = random_unit(rng, 1)[0]                # one star is garbage
    w = np.array([0.0, 1, 1, 1, 1, 1])
    assert np.allclose(solve_attitude(b, r, w), A, atol=1e-12)
    assert not np.allclose(solve_attitude(b, r), A, atol=1e-3)   # unweighted: pulled off


def test_uniform_weights_scale_does_not_matter():
    rng = np.random.default_rng(5)
    A = random_attitude(rng)
    r = random_unit(rng, 8)
    b = perturb(rng, r @ A.T, 1e-4)
    assert np.allclose(solve_attitude(b, r), solve_attitude(b, r, np.full(8, 7.3)), atol=1e-12)


def test_noise_averages_down():
    """30 stars with 10 arcsec noise each: the solution is far better than any one star."""
    rng = np.random.default_rng(6)
    sigma = np.radians(10 / 3600)
    errs = []
    for _ in range(50):
        A = random_attitude(rng)
        r = random_unit(rng, 30)
        A_est = solve_attitude(perturb(rng, r @ A.T, sigma), r)
        dA = A_est @ A.T
        errs.append(np.degrees(np.arccos(np.clip((np.trace(dA) - 1) / 2, -1, 1))) * 3600)
    assert np.median(errs) < 6.0
