"""Optical navigation: Earth's horizon in the image -> the spacecraft's position.

Every limb pixel is a world-frame direction d_i (unit). For a sphere of radius R centred at the origin, seen
from p, the limb rays make the angle rho = arcsin(R/|p|) with the direction to Earth's centre e = -p/|p|:
    d_i . e = cos(rho)   <=>   d_i . n = 1   with   n = e / cos(rho)        (|n| = 1 / cos rho)
That is LINEAR in n, so least squares gives n directly from any number of limb directions; then
    cos rho = 1/|n|,  sin rho = sqrt(n.n - 1)/|n|,  |p| = R / sin rho   =>   p = -R n / sqrt(n.n - 1).
For the WGS84 ellipsoid, scale space by S = diag(1/a, 1/a, 1/c): the ellipsoid becomes the unit sphere and
a tangent ray stays a tangent ray, so fit the unit-sphere problem to the scaled directions and unscale p.
All float64, on the CPU.
"""
import numpy as np

from tracker.camera import project, unproject
from tracker.detect_cpu import detect
from tracker.identify import identify
from tracker.limb import find_limb, refine_limb
from tracker.render_cpu import BACKGROUND, READ_NOISE, render_sky
from tracker.world import R_EQ, R_POL

MIN_LIMB_POINTS = 20


def limb_directions(points_uv, A):
    """(K, 2) pixel coordinates -> (K, 3) unit directions in the WORLD frame (rows: d = b @ A)."""
    points_uv = np.asarray(points_uv, dtype=np.float64)
    return unproject(points_uv[:, 0], points_uv[:, 1]) @ A


def _solve_n(d, robust, k_mad, max_iter):
    """Least squares for n in d . n = 1, with optional MAD-based outlier rejection."""
    keep = np.ones(len(d), dtype=bool)
    n = None
    for _ in range(max_iter if robust else 1):
        n = np.linalg.lstsq(d[keep], np.ones(keep.sum()), rcond=None)[0]
        res = np.abs(d @ n - 1.0)
        mad = np.median(np.abs(res[keep] - np.median(res[keep])))
        scale = 1.4826 * mad if mad > 0 else res[keep].max() + 1e-300
        new_keep = res <= np.median(res[keep]) + k_mad * scale
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return n, keep


def fit_horizon_sphere(d, R, robust=True, k_mad=4.0, max_iter=5):
    """Limb directions d (K, 3), unit, world frame; sphere radius R (km) -> position p (3,) km.
    Needs K >= 3 directions not all on one great circle (any real limb arc works)."""
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 3:
        raise ValueError("need at least 3 limb directions")
    n, _ = _solve_n(d, robust, k_mad, max_iter)
    nn = float(n @ n)
    if nn <= 1.0:
        raise ValueError("inconsistent limb directions: |n| <= 1 means rho is undefined")
    return -R * n / np.sqrt(nn - 1.0)


def fit_horizon_ellipsoid(d, a=R_EQ, c=R_POL, **kw):
    """Same as fit_horizon_sphere for the ellipsoid x^2/a^2 + y^2/a^2 + z^2/c^2 = 1: scale, fit the unit sphere, unscale."""
    S = np.array([1.0 / a, 1.0 / a, 1.0 / c])
    ds = np.asarray(d, dtype=np.float64) * S
    ds /= np.linalg.norm(ds, axis=1, keepdims=True)
    p_scaled = fit_horizon_sphere(ds, 1.0, **kw)
    return p_scaled / S


def position_error(p_est, p_true):
    """(total, range, cross_range) in km. range is SIGNED (|p_est| - |p_true|: positive = too far);
    cross_range is the distance of p_est from the true line of position, i.e. |p_true| times the angle between them."""
    p_est, p_true = np.asarray(p_est, dtype=np.float64), np.asarray(p_true, dtype=np.float64)
    total = float(np.linalg.norm(p_est - p_true))
    rng = float(np.linalg.norm(p_est) - np.linalg.norm(p_true))
    unit = p_true / np.linalg.norm(p_true)
    cross = float(np.linalg.norm(p_est - (p_est @ unit) * unit))
    return total, rng, cross


def _fit(pts, A_est, a, c):
    d = limb_directions(pts, A_est)
    return fit_horizon_ellipsoid(d, a, c) if c != a else fit_horizon_sphere(d, a)


REFINE_ITERATIONS = 2    # measured on Day 10: 1 pass leaves -0.13 px on a thin crescent (+0.66 % range), 2 passes -0.05 px (<0.1 %); 3 adds nothing


def position_from_image(image, A_est, s, background=BACKGROUND, read_noise=READ_NOISE, a=R_EQ, c=R_POL,
                        min_points=MIN_LIMB_POINTS, refine=True, iterations=REFINE_ITERATIONS, return_points=False):
    """The whole optical-navigation step for one image: limb -> directions -> fit. None = no solution.
    refine=True (default): the 50 %-rule limb gives a first position, which supplies the shading shape for the
    predicted-profile edge fit (`refine_limb`); the refined points are fitted again, and the pass is repeated
    `iterations` times so the shape comes from an ever better position. refine=False: the 50 % rule only."""
    pts = find_limb(image, A_est, s, background, read_noise)
    if len(pts) < min_points:
        return (None, pts) if return_points else None
    try:
        p = _fit(pts, A_est, a, c)
        if refine:
            for _ in range(iterations):
                pts2 = refine_limb(image, A_est, s, p, background, read_noise, R=a)
                if len(pts2) < min_points:
                    break
                pts, p = pts2, _fit(pts2, A_est, a, c)
    except (ValueError, np.linalg.LinAlgError):
        return (None, pts) if return_points else None
    return (p, pts) if return_points else p


def attitude_second_head(A_true, M, catalog, table, rng, read_noise=READ_NOISE):
    """Configuration (b): a second star-tracker head with fixed mounting M (camera2 = M @ camera1) that looks
    away from Earth. Render its star frame, run the lost-in-space tracker, map back: A_est = M.T @ A_head_est.
    Returns None if the tracker finds no solution."""
    A_head = M @ A_true
    img, u, v, idx = render_sky(A_head, catalog, rng=rng, read_noise=read_noise)
    du, dv, _ = detect(img, BACKGROUND, read_noise)
    sol = identify(unproject(du, dv), table)
    return None if sol is None else M.T @ sol.A
