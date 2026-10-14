"""Day 2: attitude solve (Wahba / SVD) and attitude error. Run:  pytest tests/test_solve.py -q"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracker.camera import attitude_from_pointing, random_attitude
from tracker.solve import attitude_error, solve_attitude


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


# ---------------------------------------------------------------- attitude_error

ARCSEC = np.radians(1 / 3600)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_identical_attitudes_give_zero_not_nan():
    """arccos((trace - 1) / 2) of a value rounded to 1.0000000000000002 is NaN. Must be 0."""
    rng = np.random.default_rng(10)
    for _ in range(200):
        A = random_attitude(rng)
        errs = attitude_error(A, A)
        assert len(errs) == 3
        assert np.all(np.isfinite(errs)) and np.all(np.asarray(errs) < 1e-6)


def test_pure_roll_error():
    A_true = attitude_from_pointing(83.8, -5.4, 30.0)
    A_est = attitude_from_pointing(83.8, -5.4, 30.0 + 20 / 3600)     # 20 arcsec more roll
    total, pointing, roll = attitude_error(A_est, A_true)
    assert abs(total - 20) < 1e-3 and abs(roll - 20) < 1e-3 and pointing < 1e-3


def test_pure_pointing_error():
    """Tilt about the camera x-axis: the boresight moves, no twist about it."""
    A_true = attitude_from_pointing(200.0, 30.0, 0.0)
    A_est = rot_x(15 * ARCSEC) @ A_true
    total, pointing, roll = attitude_error(A_est, A_true)
    assert abs(total - 15) < 1e-3 and abs(pointing - 15) < 1e-3 and roll < 1e-3


def test_errors_are_non_negative_and_symmetric():
    rng = np.random.default_rng(11)
    A = random_attitude(rng)
    A_est = rot_x(-7 * ARCSEC) @ A
    e1 = np.asarray(attitude_error(A_est, A))
    e2 = np.asarray(attitude_error(A, A_est))
    assert np.all(e1 >= 0) and np.allclose(e1, e2, atol=1e-6)


def test_milliarcsecond_precision():
    """1 mas = 4.8e-9 rad. cos of that is 1 - 1.2e-17, which float64 rounds to exactly 1,
    so an arccos-of-trace formula reports 0. The atan2 form must still see it."""
    A = attitude_from_pointing(10.0, 20.0, 0.0)
    total, pointing, roll = attitude_error(rot_x(0.001 * ARCSEC) @ A, A)
    assert abs(total - 0.001) < 1e-5 and abs(pointing - 0.001) < 1e-5


def test_large_errors():
    A = attitude_from_pointing(10.0, 20.0, 0.0)
    total, pointing, roll = attitude_error(rot_x(np.pi) @ A, A)        # flipped right over
    assert abs(total - 648000) < 1e-3 and abs(pointing - 648000) < 1e-3
    total, pointing, roll = attitude_error(attitude_from_pointing(10.0, 20.0, 180.0), A)
    assert abs(total - 648000) < 1e-3 and abs(roll - 648000) < 1e-3 and pointing < 1e-3


def test_small_errors_add_in_quadrature():
    """For small errors, total^2 = pointing^2 + roll^2."""
    rng = np.random.default_rng(12)
    A = random_attitude(rng)
    for _ in range(20):
        rv = rng.normal(scale=30 * ARCSEC, size=3)
        A_est = Rotation.from_rotvec(rv).as_matrix() @ A
        total, pointing, roll = attitude_error(A_est, A)
        assert abs(total ** 2 - (pointing ** 2 + roll ** 2)) < 1e-3 * total ** 2 + 1e-9


# ---------------------------------------------------------------- Day 2 gate, one field

def test_orion_true_identities_gives_a_few_arcsec():
    """render -> detect -> match each detection to its true star -> solve. No identification yet."""
    from scipy.spatial import cKDTree
    from tracker.camera import unproject
    from tracker.catalog import load_catalog
    from tracker.detect_cpu import detect
    from tracker.render_cpu import BACKGROUND, READ_NOISE, render_sky

    cat = load_catalog(5.5)
    A = attitude_from_pointing(83.8, -1.0, 0.0)
    img, u_true, v_true, idx = render_sky(A, cat, rng=np.random.default_rng(1))
    u, v, f = detect(img, BACKGROUND, READ_NOISE)
    dist, j = cKDTree(np.column_stack([u_true, v_true])).query(np.column_stack([u, v]))
    m = dist < 1.5
    A_est = solve_attitude(unproject(u[m], v[m]), cat.r[idx[j[m]]])
    total, pointing, roll = attitude_error(A_est, A)
    assert pointing < 5 and roll < 60
