"""Day 9: finding Earth's limb in the image. Run:  pytest tests/test_limb.py -q"""
import numpy as np
import pytest
from scipy.optimize import brentq

from tracker.camera import C_PIX, F_PIX, attitude_from_pointing, unproject
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.earth_render import render_earth, render_scene
from tracker.limb import exclusion_mask, find_limb
from tracker.render_cpu import BACKGROUND, READ_NOISE
from tracker.world import R_EQ, earth_in_view

GEO, FAR, LUNAR = 42_164.0, 100_000.0, 384_400.0
TOWARD_EARTH = attitude_from_pointing(180.0, 0.0, 0.0)
SUN_90 = np.array([0.0, 1.0, 0.0])            # phase 90 for a camera on +x


def sun_at_phase(deg):
    a = np.radians(deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


def true_limb_error(points, p, A, R=R_EQ):
    """Radial distance (px) of each limb point from the TRUE limb curve, signed (+ = outside).
    A camera ray b is on the limb iff b . e = cos(rho), e = Earth's centre direction in the camera frame.
    For each point we solve for the limb radius along the ray from the projected centre through the point."""
    _, (uc, vc), rho, _ = earth_in_view(p, A, R=R)
    e = A @ (-p / np.linalg.norm(p))
    err = []
    for u, v in points:
        theta = np.arctan2(v - vc, u - uc)
        r_pt = np.hypot(u - uc, v - vc)
        f = lambda r: float(unproject(np.array([uc + r * np.cos(theta)]), np.array([vc + r * np.sin(theta)]))[0] @ e - np.cos(rho))
        r_limb = brentq(f, 0.0, 2000.0)
        err.append(r_pt - r_limb)
    return np.array(err)


def angle_from_sun(points, p, A, s):
    """Angle (deg) of each point around the projected disc centre, measured from the projected Sun direction."""
    _, (uc, vc), _, _ = earth_in_view(p, A)
    sc = A @ s
    sun = np.arctan2(sc[1], sc[0])
    ang = np.arctan2(points[:, 1] - vc, points[:, 0] - uc) - sun
    return np.degrees((ang + np.pi) % (2 * np.pi) - np.pi)


@pytest.mark.parametrize("dist, tol", [(GEO, 0.15), (FAR, 0.15), (LUNAR, 0.3)])
def test_limb_points_lie_on_the_true_limb_at_phase_90(dist, tol):
    """Noise-free sphere, Sun at 90 deg: the lit limb is a sharp step and every point must sit on the true curve.
    (The plan asked for 0.1 px; the 4x4 edge quantisation alone is 0.125 px, so 0.15 px is the honest bound.)"""
    p = np.array([dist, 0.0, 0.0])
    img, _ = render_earth(p, TOWARD_EARTH, SUN_90)
    pts = find_limb(img + BACKGROUND, TOWARD_EARTH, SUN_90, BACKGROUND, READ_NOISE)
    assert pts.shape[1] == 2 and len(pts) >= 100
    err = true_limb_error(pts, p, TOWARD_EARTH)
    assert np.abs(err).max() < tol, (np.abs(err).max(), np.median(err))


def test_no_point_on_the_terminator_or_dark_side():
    """Phase 90: all points within +-75 deg of the Sun direction (never near the terminator at +-90)."""
    p = np.array([GEO, 0.0, 0.0])
    img, _ = render_earth(p, TOWARD_EARTH, SUN_90)
    pts = find_limb(img + BACKGROUND, TOWARD_EARTH, SUN_90, BACKGROUND, READ_NOISE)
    ang = angle_from_sun(pts, p, TOWARD_EARTH, SUN_90)
    assert np.abs(ang).max() <= 75.0 + 1e-6
    r_px = F_PIX * np.tan(np.arcsin(R_EQ / GEO))
    assert np.all(np.abs(np.hypot(pts[:, 0] - C_PIX, pts[:, 1] - C_PIX) - r_px) < 1.0)   # none on the terminator line


def test_off_axis_disc_and_rolled_camera():
    """Earth 5 deg off the boresight, camera rolled 30 deg, Sun oblique: the limb is an ellipse, not a circle."""
    p = np.array([FAR, 0.0, 0.0])
    A = attitude_from_pointing(180.0, 5.0, 30.0)
    s = np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])
    img, _ = render_earth(p, A, s)
    pts = find_limb(img + BACKGROUND, A, s, BACKGROUND, READ_NOISE)
    assert len(pts) >= 100
    err = true_limb_error(pts, p, A)
    assert np.abs(err).max() < 0.3 and abs(np.median(err)) < 0.15


def test_full_phase_uses_the_whole_circle_with_a_known_inward_bias():
    """Phase 0: the whole limb is lit, but Lambertian shading fades to zero there, so a 50 %-of-plateau edge sits
    0.5-1.1 px inside the true limb (measured on Day 9 and documented, not hidden)."""
    p = np.array([GEO, 0.0, 0.0])
    img, _ = render_earth(p, TOWARD_EARTH, p / GEO)
    pts = find_limb(img + BACKGROUND, TOWARD_EARTH, p / GEO, BACKGROUND, READ_NOISE)
    assert len(pts) >= 150                                         # 180 lines all round the circle
    ang = np.degrees(np.arctan2(pts[:, 1] - C_PIX, pts[:, 0] - C_PIX))
    assert ang.min() < -170 and ang.max() > 170                    # all the way round
    err = true_limb_error(pts, p, TOWARD_EARTH)
    assert -1.2 < np.median(err) < 0.0


def test_no_solution_cases():
    p = np.array([GEO, 0.0, 0.0])
    img, _ = render_earth(p, TOWARD_EARTH, -p / GEO)              # phase 180: nothing lit
    assert len(find_limb(img + BACKGROUND, TOWARD_EARTH, -p / GEO, BACKGROUND, READ_NOISE)) == 0
    s = sun_at_phase(172.0)                                        # a sliver: too little lit limb
    img, _ = render_earth(p, TOWARD_EARTH, s)
    assert len(find_limb(img + BACKGROUND, TOWARD_EARTH, s, BACKGROUND, READ_NOISE)) == 0
    A = attitude_from_pointing(0.0, 0.0, 0.0)                      # Earth not in view
    img, _ = render_earth(p, A, SUN_90)
    assert len(find_limb(img + BACKGROUND, A, SUN_90, BACKGROUND, READ_NOISE)) == 0
    noise_only = BACKGROUND + np.random.default_rng(0).normal(0, READ_NOISE, (1024, 1024)).astype(np.float32)
    assert len(find_limb(noise_only, TOWARD_EARTH, SUN_90, BACKGROUND, READ_NOISE)) == 0


def test_with_noise_and_stars():
    cat = load_catalog(5.5)
    for dist, med_tol in [(GEO, 0.25), (LUNAR, 0.4)]:
        p = np.array([dist, 0.0, 0.0])
        img, mask, u, v, idx = render_scene(p, TOWARD_EARTH, SUN_90, cat, rng=np.random.default_rng(1))
        pts = find_limb(img, TOWARD_EARTH, SUN_90, BACKGROUND, READ_NOISE)
        assert len(pts) >= 100
        err = true_limb_error(pts, p, TOWARD_EARTH)
        assert abs(np.median(err)) < med_tol
        assert np.mean(np.abs(err) > 2.0) < 0.02                   # almost no outliers


def test_exclusion_mask_stops_the_disc_posing_as_stars():
    cat = load_catalog(5.5)
    p = np.array([FAR, 0.0, 0.0])
    img, mask, u, v, idx = render_scene(p, TOWARD_EARTH, SUN_90, cat, rng=np.random.default_rng(2))
    excl = exclusion_mask(img, BACKGROUND, READ_NOISE)
    bright_disc = mask & (img > BACKGROUND + 10 * READ_NOISE)     # at phase 90 only the lit half is bright
    assert excl.dtype == bool and excl[bright_disc].all() and excl.sum() > 1.1 * bright_disc.sum()   # covered, with a border
    du, dv, _ = detect(img, BACKGROUND, READ_NOISE, mask=excl)
    from scipy.spatial import cKDTree
    dist_to_true, _ = cKDTree(np.column_stack([u, v])).query(np.column_stack([du, dv]))
    assert (dist_to_true >= 1.5).sum() <= 1                        # at most one 5-sigma noise false alarm (~0.3 expected)
    assert len(detect(img, BACKGROUND, READ_NOISE)[0]) > len(du) + 20   # without the mask the limb makes false stars


# ---------------------------------------------------------------- Day 10 fix: predicted-profile edge fit

from tracker.limb import refine_limb                                   # noqa: E402
from tracker.opnav import fit_horizon_sphere, limb_directions, position_error, position_from_image   # noqa: E402


def first_pass_position(img, A, s):
    pts0 = find_limb(img, A, s, BACKGROUND, READ_NOISE)
    return fit_horizon_sphere(limb_directions(pts0, A), R_EQ)


@pytest.mark.parametrize("label, dist, sun, tol", [
    ("GEO phase 0, fading limb", GEO, np.array([1.0, 0.0, 0.0]), 0.15),
    ("lunar crescent, phase 135", LUNAR, sun_at_phase(135.0), 0.2),
    ("GEO phase 90", GEO, SUN_90, 0.15),
    ("lunar phase 90", LUNAR, SUN_90, 0.2),
])
def test_profile_fit_removes_the_shading_bias(label, dist, sun, tol):
    """Noise-free: the predicted-profile edge sits on the true limb where the 50 % rule was biased by up to 1 px."""
    p = np.array([dist, 0.0, 0.0])
    img = render_earth(p, TOWARD_EARTH, sun)[0] + BACKGROUND
    p0 = first_pass_position(img, TOWARD_EARTH, sun)
    pts = refine_limb(img, TOWARD_EARTH, sun, p0, BACKGROUND, READ_NOISE)
    assert len(pts) >= 100
    err = true_limb_error(pts, p, TOWARD_EARTH)
    assert abs(np.median(err)) < tol and np.abs(err).max() < 2 * tol, (label, np.median(err), np.abs(err).max())


def test_profile_fit_bias_is_even_along_the_arc():
    """The crescent's 50 % bias ran from +0.47 to +0.96 px along the arc (that unevenness is what broke the range);
    the profile fit must leave no trend: error spread below 0.1 px."""
    p = np.array([LUNAR, 0.0, 0.0])
    s = sun_at_phase(135.0)
    img = render_earth(p, TOWARD_EARTH, s)[0] + BACKGROUND
    pts = refine_limb(img, TOWARD_EARTH, s, first_pass_position(img, TOWARD_EARTH, s), BACKGROUND, READ_NOISE)
    err = true_limb_error(pts, p, TOWARD_EARTH)
    assert np.std(err) < 0.1 and np.ptp(err) < 0.4


def test_profile_fit_with_noise_lunar_crescent():
    cat = load_catalog(5.5)
    p = np.array([LUNAR, 0.0, 0.0])
    s = sun_at_phase(135.0)
    img, *_ = render_scene(p, TOWARD_EARTH, s, cat, rng=np.random.default_rng(11))
    pts = refine_limb(img, TOWARD_EARTH, s, first_pass_position(img, TOWARD_EARTH, s), BACKGROUND, READ_NOISE)
    err = true_limb_error(pts, p, TOWARD_EARTH)
    assert len(pts) >= 100 and abs(np.median(err)) < 0.25


def test_position_from_image_uses_the_profile_fit_by_default():
    """End to end at lunar distance, phase 135, noisy, true attitude, sphere Earth: range within 0.3 % (was +5.8 %
    with the 50 % rule, +0.7 % after one profile pass, ~0.1 % after the two passes the pipeline runs)."""
    cat = load_catalog(5.5)
    p = np.array([LUNAR, 0.0, 0.0])
    s = sun_at_phase(135.0)
    img, *_ = render_scene(p, TOWARD_EARTH, s, cat, rng=np.random.default_rng(12))
    p_est = position_from_image(img, TOWARD_EARTH, s, BACKGROUND, READ_NOISE, a=R_EQ, c=R_EQ)
    total, rng_err, cross = position_error(p_est, p)
    assert abs(rng_err) / LUNAR < 0.003 and cross / LUNAR < 0.002
    p_raw = position_from_image(img, TOWARD_EARTH, s, BACKGROUND, READ_NOISE, a=R_EQ, c=R_EQ, refine=False)
    assert abs(position_error(p_raw, p)[1]) > abs(rng_err)             # the old rule is worse
