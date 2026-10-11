"""Day 1 gate, part 1: the camera model.  Run:  pytest tests/test_camera.py -v"""
import numpy as np
import pytest

from tracker.camera import (
    C_PIX, F_PIX, FOV_DEG, N_PIX,
    attitude_from_pointing, project, radec_to_unit, random_attitude, unproject,
)


def is_rotation(A):
    return np.allclose(A @ A.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(A), 1.0)


def test_constants():
    assert N_PIX == 1024
    assert FOV_DEG == 20.0
    assert np.isclose(C_PIX, 511.5)
    assert np.isclose(F_PIX * np.tan(np.radians(FOV_DEG / 2)), N_PIX / 2)


@pytest.mark.parametrize("ra, dec, expected", [
    (0.0, 0.0, [1, 0, 0]),
    (90.0, 0.0, [0, 1, 0]),
    (180.0, 0.0, [-1, 0, 0]),
    (123.4, 90.0, [0, 0, 1]),
    (0.0, -90.0, [0, 0, -1]),
])
def test_radec_to_unit_known_points(ra, dec, expected):
    r = radec_to_unit(np.array([ra]), np.array([dec]))
    assert r.shape == (1, 3)
    assert np.allclose(r[0], expected, atol=1e-12)


def test_radec_to_unit_is_unit_length():
    rng = np.random.default_rng(0)
    r = radec_to_unit(rng.uniform(0, 360, 1000), rng.uniform(-90, 90, 1000))
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0)


def test_random_attitude_is_rotation():
    rng = np.random.default_rng(1)
    for _ in range(100):
        assert is_rotation(random_attitude(rng))


def test_random_attitude_is_uniform_on_the_sky():
    """Uniform on a sphere -> half of boresights within 30 deg of the equator.
    Uniform RA/Dec gives about a third, and fails here."""
    rng = np.random.default_rng(2)
    z = np.array([random_attitude(rng)[2, 2] for _ in range(20000)])
    assert abs(np.mean(np.abs(z) < 0.5) - 0.5) < 0.015


def test_pointing_puts_target_on_boresight():
    A = attitude_from_pointing(83.8, -5.4, 0.0)
    assert is_rotation(A)
    target = radec_to_unit(np.array([83.8]), np.array([-5.4]))[0]
    assert np.allclose(A[2], target)
    u, v, vis = project(A, target[None, :])
    assert vis[0] and np.isclose(u[0], C_PIX) and np.isclose(v[0], C_PIX)


def test_north_up_east_left_at_zero_roll():
    A = attitude_from_pointing(83.8, -5.4, 0.0)
    north = radec_to_unit(np.array([83.8]), np.array([-5.4 + 2.0]))
    east = radec_to_unit(np.array([83.8 + 2.0]), np.array([-5.4]))
    u_n, v_n, _ = project(A, north)
    u_e, v_e, _ = project(A, east)
    assert v_n[0] < C_PIX - 50 and abs(u_n[0] - C_PIX) < 5
    assert u_e[0] < C_PIX - 50 and abs(v_e[0] - C_PIX) < 5


def test_roll_rotates_about_boresight():
    A0 = attitude_from_pointing(200.0, 30.0, 0.0)
    A90 = attitude_from_pointing(200.0, 30.0, 90.0)
    assert np.allclose(A0[2], A90[2])
    assert np.isclose(abs(A0[0] @ A90[0]), 0.0, atol=1e-12)


def test_project_unproject_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        A = random_attitude(rng)
        off = rng.uniform(-1, 1, size=(200, 2)) * np.tan(np.radians(8))
        b = np.column_stack([off[:, 0], off[:, 1], np.ones(200)])
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        r = b @ A
        u, v, vis = project(A, r)
        assert vis.all()
        b_back = unproject(u, v)
        assert np.allclose(b_back @ A, r, atol=1e-12)


def test_visibility_edges():
    A = np.eye(3)
    def dir_at(angle_deg):
        a = np.radians(angle_deg)
        return np.array([[np.sin(a), 0.0, np.cos(a)]])
    assert project(A, dir_at(9.9))[2][0]
    assert not project(A, dir_at(10.5))[2][0]
    assert not project(A, np.array([[0.0, 0.0, -1.0]]))[2][0]
