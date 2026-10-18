"""Day 8: world geometry for optical navigation. Run:  pytest tests/test_world.py -q"""
import numpy as np
import pytest

from tracker.camera import C_PIX, F_PIX, attitude_from_pointing, random_attitude
from tracker.world import (EARTH_LEVEL, R_EQ, R_POL, earth_in_view, sample_position, sample_sun)


def test_constants_are_wgs84_and_sane():
    assert R_EQ == 6378.137 and R_POL == 6356.752
    assert 0 < R_EQ - R_POL < 22                         # ~21.4 km of flattening
    assert 1000 < EARTH_LEVEL < 60000                    # far above the 5-sigma threshold (45), below a mag-0 star peak


def test_sample_position_bounds_and_log_uniform_distance():
    rng = np.random.default_rng(0)
    p = np.array([sample_position(rng, 42_164.0, 400_000.0) for _ in range(20000)])
    assert p.shape == (20000, 3)
    r = np.linalg.norm(p, axis=1)
    assert r.min() >= 42_164.0 and r.max() <= 400_000.0
    # log-uniform: log r is uniform, so its median is the log-midpoint and its quartiles sit at 1/4, 3/4
    lo, hi = np.log(42_164.0), np.log(400_000.0)
    q = (np.log(r) - lo) / (hi - lo)
    assert abs(np.median(q) - 0.5) < 0.015
    assert abs(np.percentile(q, 25) - 0.25) < 0.015 and abs(np.percentile(q, 75) - 0.75) < 0.015


def test_sample_position_direction_is_uniform_on_the_sphere():
    """Half of uniformly spread directions have |z| < 0.5 (uniform in z is the signature of uniform on a sphere)."""
    rng = np.random.default_rng(1)
    p = np.array([sample_position(rng, 1.0, 1.0) for _ in range(20000)])      # r_min == r_max: fixed distance
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)
    assert abs(np.mean(np.abs(p[:, 2]) < 0.5) - 0.5) < 0.015
    assert abs(p.mean(axis=0)).max() < 0.02


def test_sample_sun_is_a_uniform_unit_vector():
    rng = np.random.default_rng(2)
    s = np.array([sample_sun(rng) for _ in range(20000)])
    assert np.allclose(np.linalg.norm(s, axis=1), 1.0)
    assert abs(np.mean(np.abs(s[:, 2]) < 0.5) - 0.5) < 0.015


@pytest.mark.parametrize("dist, rho_deg, px", [(42_164.0, 8.70, 444), (384_400.0, 0.95, 48)])
def test_earth_in_view_sanity_values(dist, rho_deg, px):
    """The two reference values from the plan: GEO and lunar distance, camera pointing straight at Earth."""
    p = np.array([dist, 0.0, 0.0])
    A = attitude_from_pointing(180.0, 0.0, 0.0)          # boresight toward -x, i.e. at Earth's centre
    visible, uv, rho, pixel_radius = earth_in_view(p, A)
    assert visible
    assert np.allclose(uv, [C_PIX, C_PIX], atol=1e-6)
    assert abs(np.degrees(rho) - rho_deg) < 0.01
    assert abs(pixel_radius - px) < 1.0
    assert np.isclose(pixel_radius, F_PIX * np.tan(rho))


def test_earth_in_view_off_axis_and_out_of_view():
    p = np.array([100_000.0, 0.0, 0.0])
    A = attitude_from_pointing(180.0, 5.0, 0.0)          # 5 deg above Earth's centre: Earth below the image centre
    visible, (u, v), rho, pr = earth_in_view(p, A)
    assert visible and abs(u - C_PIX) < 1e-6 and v > C_PIX + 100
    A = attitude_from_pointing(0.0, 0.0, 0.0)            # looking away from Earth
    assert earth_in_view(p, A)[0] is False
    A = attitude_from_pointing(180.0, 25.0, 0.0)         # 25 deg off: Earth (3.7 deg radius) entirely outside the 20 deg field
    assert earth_in_view(p, A)[0] is False


def test_earth_in_view_partially_in_frame_counts_as_visible():
    p = np.array([100_000.0, 0.0, 0.0])
    A = attitude_from_pointing(180.0, 11.0, 0.0)         # centre 1 deg outside the edge, disc radius 3.7 deg: limb is in frame
    visible, uv, rho, pr = earth_in_view(p, A)
    assert visible


def test_earth_in_view_rejects_positions_inside_earth():
    with pytest.raises(ValueError):
        earth_in_view(np.array([1000.0, 0.0, 0.0]), np.eye(3))


def test_earth_in_view_random_attitudes_never_crash():
    rng = np.random.default_rng(3)
    for _ in range(200):
        out = earth_in_view(sample_position(rng, 42_164.0, 1e6), random_attitude(rng))
        assert out[0] in (True, False) and np.isfinite(out[2]) and np.isfinite(out[3])
