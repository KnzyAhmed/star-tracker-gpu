"""Day 10: horizon fit -> position. Run:  pytest tests/test_opnav.py -q"""
import numpy as np
import pytest

from tracker.camera import attitude_from_pointing, random_attitude, unproject
from tracker.catalog import load_catalog
from tracker.earth_render import render_earth, render_scene
from tracker.identify import build_pair_table
from tracker.limb import find_limb
from tracker.opnav import (attitude_second_head, fit_horizon_ellipsoid, fit_horizon_sphere, limb_directions,
                           position_error, position_from_image)
from tracker.render_cpu import BACKGROUND, READ_NOISE
from tracker.world import R_EQ, R_POL, sample_position

GEO, FAR, LUNAR = 42_164.0, 100_000.0, 384_400.0


def exact_limb_directions(p, a, c, n=180, arc_deg=360.0, centre_deg=0.0):
    """Exact tangent directions from p to the ellipsoid, on an arc of the limb. Built in scaled space, where the
    ellipsoid is the unit sphere and tangent rays make the angle rho = arcsin(1/|p'|) with the centre direction."""
    S = np.array([1 / a, 1 / a, 1 / c])
    ps = p * S
    dist = np.linalg.norm(ps)
    e = -ps / dist
    rho = np.arcsin(1.0 / dist)
    u1 = np.cross(e, [0, 0, 1.0])
    u1 = u1 / np.linalg.norm(u1) if np.linalg.norm(u1) > 1e-9 else np.array([1.0, 0, 0])
    u2 = np.cross(e, u1)
    psi = np.radians(centre_deg) + np.radians(arc_deg) * (np.arange(n) + 0.5) / n - np.radians(arc_deg) / 2
    ds = np.cos(rho) * e + np.sin(rho) * (np.cos(psi)[:, None] * u1 + np.sin(psi)[:, None] * u2)
    d = ds / S                                    # back to world space: the ray direction (unnormalised)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.mark.parametrize("dist", [GEO, FAR, LUNAR])
def test_sphere_fit_recovers_the_position_exactly(dist):
    rng = np.random.default_rng(int(dist))
    for _ in range(5):
        p = sample_position(rng, dist, dist)
        d = exact_limb_directions(p, R_EQ, R_EQ)
        p_est = fit_horizon_sphere(d, R_EQ)
        assert np.allclose(p_est, p, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("dist", [GEO, LUNAR])
def test_ellipsoid_fit_recovers_the_position_exactly(dist):
    rng = np.random.default_rng(int(dist) + 1)
    for _ in range(5):
        p = sample_position(rng, dist, dist)
        d = exact_limb_directions(p, R_EQ, R_POL)
        assert np.allclose(fit_horizon_ellipsoid(d, R_EQ, R_POL), p, rtol=1e-9, atol=1e-6)
        assert np.allclose(fit_horizon_ellipsoid(d, R_EQ, R_EQ), fit_horizon_sphere(d, R_EQ))   # c = a is the sphere


def test_partial_arc_still_recovers_exactly_without_noise():
    p = np.array([0.0, LUNAR, 0.0])
    d = exact_limb_directions(p, R_EQ, R_EQ, n=60, arc_deg=60.0, centre_deg=30.0)
    assert np.allclose(fit_horizon_sphere(d, R_EQ), p, rtol=1e-8, atol=1e-6)


def test_sphere_model_on_an_oblate_earth_is_biased_and_the_ellipsoid_model_is_not():
    """Seen from the equatorial plane the silhouette is an ellipse with semi-axes R_EQ (wide) and ~R_POL (tall).
    A sphere of radius R_EQ fitted to the full limb sees a mean radius ~(R_EQ + R_POL)/2 and so places the camera
    R_EQ / R_mean - 1 = +0.17 % too far: +71 km at GEO, +645 km at lunar distance. From the pole the bias
    is small but NOT zero at finite distance: the tangent circle sits at z = c^2/D = 958 km (radius 6305 km),
    so tan(rho) = 6305 / 41206 and the sphere model gives 42,167 km: +3.4 km by hand."""
    for dist, expect_km in [(GEO, 71.0), (LUNAR, 645.0)]:
        p = np.array([dist, 0.0, 0.0])
        d = exact_limb_directions(p, R_EQ, R_POL)
        total, rng_err, cross = position_error(fit_horizon_sphere(d, R_EQ), p)
        assert abs(rng_err - expect_km) < 0.08 * expect_km and cross < 1e-3 * dist
        assert position_error(fit_horizon_ellipsoid(d, R_EQ, R_POL), p)[0] < 1e-6
    p = np.array([0.0, 0.0, GEO])
    pole_bias = position_error(fit_horizon_sphere(exact_limb_directions(p, R_EQ, R_POL), R_EQ), p)[1]
    assert 2.5 < pole_bias < 4.0


def test_error_shrinks_with_more_limb_points():
    """Noise of 0.3 px on each limb point: the position error should fall roughly like 1/sqrt(points)."""
    p = np.array([FAR, 0.0, 0.0])
    rng = np.random.default_rng(7)
    sigma = 0.3 / 2903.7                            # 0.3 px in radians
    errs = {}
    for n in (20, 180):
        trials = []
        for _ in range(40):
            d = exact_limb_directions(p, R_EQ, R_EQ, n=n, arc_deg=150.0)
            d += rng.normal(scale=sigma, size=d.shape)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            trials.append(position_error(fit_horizon_sphere(d, R_EQ), p)[0])
        errs[n] = np.median(trials)
    assert 2.0 < errs[20] / errs[180] < 4.5


def test_robust_fit_ignores_a_few_bad_points():
    p = np.array([GEO, 0.0, 0.0])
    d = exact_limb_directions(p, R_EQ, R_EQ, n=180, arc_deg=150.0)
    bad = d.copy()
    bad[:6] = unproject(np.array([200.0, 300, 400, 500, 600, 700]), np.array([100.0, 100, 100, 100, 100, 100])) @ attitude_from_pointing(180.0, 0.0, 0.0)
    assert position_error(fit_horizon_sphere(bad, R_EQ), p)[0] < 5.0                 # 6 outliers of 180: rejected
    assert position_error(fit_horizon_sphere(bad, R_EQ, robust=False), p)[0] > 50.0  # without rejection they hurt


def test_position_error_splits_range_and_cross_range():
    p = np.array([100_000.0, 0.0, 0.0])
    total, rng_err, cross = position_error(np.array([101_000.0, 0.0, 0.0]), p)
    assert abs(rng_err - 1000.0) < 1e-6 and cross < 1e-6 and abs(total - 1000.0) < 1e-6
    total, rng_err, cross = position_error(np.array([100_000.0, 500.0, 0.0]), p)
    assert abs(cross - 500.0) < 1.5 and abs(rng_err) < 1.5 and abs(total - 500.0) < 1e-6
    total, rng_err, cross = position_error(np.array([99_000.0, 0.0, 0.0]), p)
    assert abs(rng_err + 1000.0) < 1e-6                                                 # signed: too close is negative


def test_limb_directions_are_unit_world_vectors():
    A = attitude_from_pointing(180.0, 5.0, 30.0)
    d = limb_directions(np.array([[511.5, 511.5], [600.0, 400.0]]), A)
    assert d.shape == (2, 3) and np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert np.allclose(d[0], A[2])                                                     # the image centre is the boresight


def test_end_to_end_with_the_true_attitude_phase_90():
    """Rendered frame (sphere, stars, noise) -> find_limb -> fit: range within 0.1 % at GEO (0.03 px limb error)."""
    cat = load_catalog(5.5)
    p = np.array([GEO, 0.0, 0.0])
    A = attitude_from_pointing(180.0, 0.0, 0.0)
    s = np.array([0.0, 1.0, 0.0])
    img, mask, u, v, idx = render_scene(p, A, s, cat, rng=np.random.default_rng(3))
    p_est = position_from_image(img, A, s, BACKGROUND, READ_NOISE, a=R_EQ, c=R_EQ)
    total, rng_err, cross = position_error(p_est, p)
    assert abs(rng_err) / GEO < 1e-3 and cross / GEO < 1e-3


def test_no_solution_when_earth_is_not_usable():
    cat = load_catalog(5.5)
    p = np.array([GEO, 0.0, 0.0])
    A = attitude_from_pointing(180.0, 0.0, 0.0)
    img, *_ = render_scene(p, A, -p / GEO, cat, rng=np.random.default_rng(4))        # night side
    assert position_from_image(img, A, -p / GEO, BACKGROUND, READ_NOISE) is None
    A2 = attitude_from_pointing(0.0, 0.0, 0.0)                                         # Earth not in view
    img, *_ = render_scene(p, A2, p / GEO, cat, rng=np.random.default_rng(5))
    assert position_from_image(img, A2, p / GEO, BACKGROUND, READ_NOISE) is None


def test_second_head_attitude():
    """A second star-tracker head with a known mounting rotation M (camera2 = M camera1) looks away from Earth;
    its own star frame is solved and mapped back: A_est = M.T @ A_head_est, within a few arcsec."""
    from tracker.solve import attitude_error
    cat = load_catalog(5.5)
    table = build_pair_table(cat.r)
    rng = np.random.default_rng(6)
    A = attitude_from_pointing(180.0, 0.0, 0.0)
    M = attitude_from_pointing(90.0, 0.0, 0.0) @ A.T                                   # head boresight 90 deg away
    A_est = attitude_second_head(A, M, cat, table, rng)
    assert A_est is not None
    total, pointing, roll = attitude_error(A_est, A)
    assert pointing < 10 and roll < 60
