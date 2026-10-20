"""Day 12: the optical-navigation Monte Carlo campaign engine. One cell = (distance, Sun phase angle, read
noise); one trial = one spacecraft position, Sun direction and camera attitude; one or two result rows per
trial (attitude from the second star-tracker head, and optionally the TRUE attitude, to separate the
attitude error's contribution).

Every trial is a pure function of (seed, trial number), as on Day 5:
  geometry   trial_geometry(seed, trial, distance, phase): default_rng([seed, trial, 0]) draws the direction to
             Earth, the Sun's azimuth around that direction, the camera's offset from Earth's centre and its
             roll. The draws do not depend on the cell, so the SAME trial number is the same sky, Sun azimuth and
             roll in every cell (paired comparisons).
  Earth-camera noise   counter noise (tracker/noise.py) with key (seed, trial)
  head-camera noise    counter noise with key (seed + HEAD_SEED_OFFSET, trial)
so a trial gives the same row alone, in any batch, in any process, and (to float32 rounding of the Earth
image) on the CPU or the GPU; an interrupted run resumes exactly.

Camera layout (configuration b of Day 10): the Earth camera points at Earth (the whole disc in the frame,
offset uniformly within the sensor minus a margin); a second star-tracker head with the fixed mounting
HEAD_MOUNT (camera2 = HEAD_MOUNT @ camera1) looks 90 deg away, renders its own star frame, and the lost-in-space
tracker solves it: A_est = HEAD_MOUNT.T @ A_head_est. The Earth frame also carries the stars that are not
occluded (realism; the limb finder must cope with them).

Outcome thresholds, fixed before the campaign (BUILD_LOG 2026-10-20):
  no solution   the head's tracker finds nothing, or fewer than MIN_LIMB_POINTS limb points, or the fit fails
  correct       total position error < CORRECT_FRACTION (1 %) of the true distance
  wrong         a position was returned and its error is >= 1 % of the true distance
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tracker.batch import pack_frames, project_frames
from tracker.camera import C_PIX, F_PIX, attitude_from_pointing, project, unproject
from tracker.detect_cpu import detect
from tracker.earth_render import occluded, render_earth
from tracker.evaluate import CORRECT, NO_SOLUTION, WRONG
from tracker.identify import identify
from tracker.opnav import MIN_LIMB_POINTS, fit_horizon_sphere, limb_directions, position_error, position_from_image
from tracker.render_cpu import BACKGROUND, render, star_flux
from tracker.solve import attitude_error
from tracker.world import R_EQ, R_POL, _unit_sphere

COLUMNS = ("trial", "distance", "phase", "read_noise", "att_src",
           "px", "py", "pz", "sx", "sy", "sz", "pixel_radius", "offset_px",
           "att_total", "att_pointing", "att_roll",
           "reason", "n_limb", "outcome", "total_km", "range_km", "range_pct", "cross_km",
           "total_sphere_km", "range_sphere_km")
ATT_TRACKER, ATT_TRUE = 0, 1                       # att_src: attitude from the second head's tracker / the truth
OK, NO_TRACKER, FEW_POINTS, FIT_FAILED = 0, 1, 2, 3  # reason
REASONS = ("ok", "tracker: no solution", "too few limb points", "fit failed")
CORRECT_FRACTION = 0.01                            # correct = total position error below 1 % of the true distance
EDGE_MARGIN_PX = 10.0                              # the disc stays this far inside the sensor
HEAD_MOUNT = np.array([[1.0, 0.0, 0.0],            # second head = Earth camera rotated 90 deg about its x axis
                       [0.0, 0.0, 1.0],            # (its boresight is the Earth camera's -y axis): looks 90 deg
                       [0.0, -1.0, 0.0]])          # away from Earth, always at stars
HEAD_SEED_OFFSET = 1


def _perpendicular_basis(e):
    u1 = np.cross(e, [0.0, 0.0, 1.0])
    if np.linalg.norm(u1) < 1e-9:
        u1 = np.cross(e, [1.0, 0.0, 0.0])
    u1 /= np.linalg.norm(u1)
    return u1, np.cross(e, u1)


def _pointing(boresight, roll_deg):
    ra = np.degrees(np.arctan2(boresight[1], boresight[0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(boresight[2], -1.0, 1.0)))
    return attitude_from_pointing(ra, dec, roll_deg)


def trial_geometry(seed, trial, distance, phase_deg):
    """-> (p, s, A, pixel_radius, offset_px). p: spacecraft position (km) at |p| = distance, direction uniform;
    s: Sun direction at the phase angle (angle at Earth between Earth->spacecraft and Earth->Sun), azimuth
    uniform; A: Earth-camera attitude, boresight = Earth's direction tilted by an offset uniform within the disc of
    radius C_PIX - EDGE_MARGIN_PX - pixel_radius (the whole Earth in the frame), roll uniform."""
    rng = np.random.default_rng([int(seed), int(trial), 0])
    p_hat = _unit_sphere(rng)
    az_sun, u_off, az_off, roll = rng.uniform(0, 2 * np.pi), rng.uniform(), rng.uniform(0, 2 * np.pi), rng.uniform(-180, 180)
    e1, e2 = _perpendicular_basis(p_hat)
    ph = np.radians(phase_deg)
    s = np.cos(ph) * p_hat + np.sin(ph) * (np.cos(az_sun) * e1 + np.sin(az_sun) * e2)
    rho = np.arcsin(R_EQ / distance)
    r_px = F_PIX * np.tan(rho)
    off_max = max(0.0, C_PIX - EDGE_MARGIN_PX - r_px)
    off_px = off_max * np.sqrt(u_off)                                      # uniform over the disc
    theta = np.arctan(off_px / F_PIX)
    boresight = np.cos(theta) * (-p_hat) + np.sin(theta) * (np.cos(az_off) * e1 + np.sin(az_off) * e2)
    return distance * p_hat, s, _pointing(boresight, roll), float(r_px), float(off_px)


def score_position(p_est, p_true):
    """-> (outcome, total_km, range_km, range_pct, cross_km); NaNs when there is no estimate."""
    if p_est is None:
        return NO_SOLUTION, np.nan, np.nan, np.nan, np.nan
    dist = float(np.linalg.norm(p_true))
    total, rng, cross = position_error(p_est, p_true)
    outcome = CORRECT if total < CORRECT_FRACTION * dist else WRONG
    return outcome, total, rng, 100.0 * rng / dist, cross


def solve_trial(image, p, s, A_est, read_noise, a=R_EQ, c=R_POL):
    """One image + one attitude -> (reason, n_limb, outcome, total_km, range_km, range_pct, cross_km,
    total_sphere_km, range_sphere_km). The sphere-model numbers refit the SAME limb points with a sphere of
    radius a (the oblateness comparison)."""
    nan6 = (np.nan,) * 6
    if A_est is None:
        return (NO_TRACKER, 0, NO_SOLUTION) + nan6
    p_est, pts = position_from_image(image, A_est, s, BACKGROUND, read_noise, a=a, c=c, return_points=True)
    if p_est is None:
        return (FEW_POINTS if len(pts) < MIN_LIMB_POINTS else FIT_FAILED, len(pts), NO_SOLUTION) + nan6
    outcome, total, rng, pct, cross = score_position(p_est, p)
    try:
        t_s, r_s, _ = position_error(fit_horizon_sphere(limb_directions(pts, A_est), a), p)
    except (ValueError, np.linalg.LinAlgError):
        t_s = r_s = np.nan
    return OK, len(pts), outcome, total, rng, pct, cross, t_s, r_s


def _earth_frame_stars(p, A, catalog, a, c):
    u, v, vis = project(A, catalog.r)
    k = np.flatnonzero(vis)
    k = k[~occluded(p, catalog.r[k], a, c)]
    return u[k], v[k], star_flux(catalog.mag[k])


def render_trials(trials, seed, distance, phase, read_noise, catalog, backend="cpu", a=R_EQ, c=R_POL):
    """Stage 1 -> (geometry, images, found): per trial its (p, s, A, r_px, off_px), the Earth-camera image
    (stars + Earth + background + noise, float32 NumPy) and the second head's detected star list (u, v, flux)."""
    trials = np.asarray(trials, dtype=np.int64)
    geometry = [trial_geometry(seed, t, distance, phase) for t in trials]
    heads = [HEAD_MOUNT @ g[2] for g in geometry]
    if backend == "gpu":
        import cupy as cp
        from tracker import gpu
        frames = [_earth_frame_stars(p, A, catalog, a, c) for p, s, A, *_ in geometry]
        imgs = gpu.render_batch(pack_frames(frames), seed=seed, frame_ids=trials, read_noise=read_noise)
        earth, _ = gpu.render_earth_batch(np.array([g[0] for g in geometry]), np.array([g[2] for g in geometry]),
                                          np.array([g[1] for g in geometry]), a=a, c=c)
        imgs += earth
        images = cp.asnumpy(imgs)
        del imgs, earth
        head_frames, _ = project_frames(heads, catalog)
        head_imgs = gpu.render_batch(pack_frames(head_frames), seed=seed + HEAD_SEED_OFFSET, frame_ids=trials,
                                     read_noise=read_noise)
        found = gpu.detect_batch(head_imgs, BACKGROUND, read_noise)
    elif backend == "cpu":
        images, found = [], []
        for k, (p, s, A, *_) in enumerate(geometry):
            u, v, f = _earth_frame_stars(p, A, catalog, a, c)
            img = render(u, v, f, read_noise=read_noise, noise_key=(seed, int(trials[k])))
            img += render_earth(p, A, s, a=a, c=c)[0]
            images.append(img)
            hu, hv, hvis = project(heads[k], catalog.r)
            j = np.flatnonzero(hvis)
            head_img = render(hu[j], hv[j], star_flux(catalog.mag[j]), read_noise=read_noise,
                              noise_key=(seed + HEAD_SEED_OFFSET, int(trials[k])))
            found.append(detect(head_img, BACKGROUND, read_noise))
        images = np.stack(images)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return geometry, images, found


def solve_rows(trials, geometry, images, found, phase, read_noise, table, true_attitude=True, a=R_EQ, c=R_POL):
    """Stage 2 (CPU, float64): identify the head's stars, map the attitude back, fit the horizon -> rows."""
    rows = []
    for k, t in enumerate(trials):
        p, s, A, r_px, off = geometry[k]
        u, v, _ = found[k]
        sol = identify(unproject(u, v), table)
        A_est = None if sol is None else HEAD_MOUNT.T @ sol.A
        att = attitude_error(A_est, A) if A_est is not None else (np.nan,) * 3
        variants = [(ATT_TRACKER, A_est, att)]
        if true_attitude:
            variants.append((ATT_TRUE, A, (0.0, 0.0, 0.0)))
        for src, A_use, (at, ap, ar) in variants:
            res = solve_trial(images[k], p, s, A_use, read_noise, a, c)
            rows.append((t, np.linalg.norm(p), phase, read_noise, src, *p, *s, r_px, off, at, ap, ar, *res))
    return np.array(rows, dtype=np.float64)


def run_batch(trials, seed, distance, phase, read_noise, catalog, table, backend="cpu", true_attitude=True,
              a=R_EQ, c=R_POL):
    """Run trials (int array) in one cell. Returns (rows, len(COLUMNS)) float64: one row per trial per attitude
    source (two per trial when true_attitude)."""
    trials = np.asarray(trials, dtype=np.int64)
    geometry, images, found = render_trials(trials, seed, distance, phase, read_noise, catalog, backend, a, c)
    return solve_rows(trials, geometry, images, found, phase, read_noise, table, true_attitude, a, c)


_WORKER = {}


def _init_worker():
    from tracker.catalog import load_catalog
    from tracker.identify import build_pair_table
    cat = load_catalog(5.5)
    _WORKER["catalog"], _WORKER["table"] = cat, build_pair_table(cat.r)


def _worker_batch(args):
    trials, seed, distance, phase, read_noise, true_attitude = args
    return run_batch(trials, seed, distance, phase, read_noise, _WORKER["catalog"], _WORKER["table"],
                     backend="cpu", true_attitude=true_attitude)


def run_condition(out_csv, n_trials, seed, distance, phase, read_noise, catalog, table, backend="cpu", batch=64,
                  workers=1, true_attitude=True, progress=None, pool=None):
    """Make sure out_csv holds trials 0..n_trials-1 of this cell; trials already there are skipped (resume);
    rows are appended after every batch. workers > 1 (CPU backend only) spreads each batch over that many
    processes, each rendering and solving its own trials (`pool`: an existing ProcessPoolExecutor to reuse).
    Returns the number of trials run now."""
    out_csv = Path(out_csv)
    done = set()
    if out_csv.exists() and out_csv.stat().st_size:
        done = set(np.loadtxt(out_csv, delimiter=",", skiprows=1, ndmin=2)[:, 0].astype(int).tolist())
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text(",".join(COLUMNS) + "\n")
    todo = np.array([t for t in range(n_trials) if t not in done], dtype=np.int64)
    own_pool = None
    if workers > 1 and backend == "cpu" and pool is None:
        own_pool = pool = ProcessPoolExecutor(workers, initializer=_init_worker)
    try:
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            if pool is not None and backend == "cpu":
                parts = [c for c in np.array_split(chunk, workers) if len(c)]
                rows = np.concatenate(list(pool.map(_worker_batch, [(c, seed, distance, phase, read_noise, true_attitude)
                                                                    for c in parts])))
            else:
                rows = run_batch(chunk, seed, distance, phase, read_noise, catalog, table, backend, true_attitude)
            with open(out_csv, "a") as fh:
                np.savetxt(fh, rows, delimiter=",", fmt="%.10g")
            if progress:
                progress(min(start + batch, len(todo)), len(todo))
    finally:
        if own_pool is not None:
            own_pool.shutdown()
    return len(todo)
