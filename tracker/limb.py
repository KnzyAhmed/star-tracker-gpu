"""Finding Earth's limb (the sharp edge of the lit disc) in the camera image.

    1. segment: the largest bright connected region is Earth; its centroid is the seed, its extent the scan range
    2. the Sun direction in the image (from the attitude) says which side of the disc is lit
    3. scan lines from the seed over the lit side only (never toward the terminator), sampled with bilinear
       interpolation; on each line the edge is the outermost crossing of 50 % of the LOCAL plateau, sub-pixel
    4. too few good lines -> an empty array: the honest "no solution"

Why 50 % of the local plateau and not of the line's peak or a fixed level (measured on Day 9, noise-free, px):
    fixed low threshold   +2 to +3   (the PSF tail pushes the crossing outward)
    maximum gradient      -0.3 to -1.6 and noisy
    50 % of local plateau  within +-0.16 at phase >= 45 deg; -0.5 to -1.1 at phase 0, where the Lambertian
                           limb fades to zero (a model limitation, documented in LEARNING.md)
"""
import numpy as np
from scipy.ndimage import binary_dilation, label, map_coordinates

from tracker.camera import N_PIX, unproject
from tracker.earth_render import _ray_hits
from tracker.render_cpu import SIGMA_PSF

SEG_SIGMA = 10.0       # segmentation threshold above background, in read-noise units
MIN_AREA = 200         # smallest bright region (px) accepted as Earth; a star covers ~30 px
STEP = 0.25            # scan sampling step along a line, px
PLATEAU_IN = 6.0       # the plateau is read this far (px) inside the coarse edge: past the PSF, before the shading fall-off
PLATEAU_WIDTH = 4.0    # ... averaged over this many px
MIN_CONTRAST = 20.0    # a line needs plateau - background > this many read-noise sigmas to be trusted


def segment_earth(image, background, read_noise):
    """Largest bright connected region. Returns (mask, (uc, vc) centroid, r_max) or None if nothing big enough."""
    bright = image > background + SEG_SIGMA * read_noise
    lab, n = label(bright)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    k = int(sizes.argmax())
    if sizes[k] < MIN_AREA:
        return None
    mask = lab == k
    vv, uu = np.nonzero(mask)
    uc, vc = uu.mean(), vv.mean()
    r_max = np.hypot(uu - uc, vv - vc).max()
    return mask, (float(uc), float(vc)), float(r_max)


def exclusion_mask(image, background, read_noise, pixels=6):
    """Pixels the star detector must ignore: Earth's bright region grown by `pixels`, so the disc's edge
    (and the PSF tail just outside it) cannot pose as stars. All-False if no Earth is found."""
    seg = segment_earth(image, background, read_noise)
    if seg is None:
        return np.zeros(image.shape, dtype=bool)
    return binary_dilation(seg[0], iterations=int(pixels))


def scan_lines(image, A_est, s, background, read_noise, n_lines=180, margin_deg=15.0):
    """The shared first half of both limb finders. Returns None (no usable Earth) or
    (seed (uc, vc), angles (n,), r (m,) sample radii, prof (n, m) background-subtracted samples)."""
    image = np.asarray(image, dtype=np.float64)
    seg = segment_earth(image, background, read_noise)
    if seg is None:
        return None
    _, (uc, vc), r_max = seg
    r_max = min(r_max * 1.3 + 12.0, 1.5 * N_PIX)
    # which side is lit: the Sun in the camera frame; phase angle = angle between Sun and the Earth->camera direction
    sc = A_est @ np.asarray(s, dtype=np.float64)
    phase = np.degrees(np.arccos(np.clip(-sc[2], -1.0, 1.0)))
    if phase > 180.0 - margin_deg:
        return None
    if phase < margin_deg:
        angles = np.linspace(-np.pi, np.pi, n_lines, endpoint=False)
    else:
        sun = np.arctan2(sc[1], sc[0])
        half = np.radians(90.0 - margin_deg)
        angles = sun + np.linspace(-half, half, n_lines)
    r = np.arange(0.0, r_max, STEP)
    uu = uc + r[None, :] * np.cos(angles)[:, None]
    vv = vc + r[None, :] * np.sin(angles)[:, None]
    prof = map_coordinates(image, [vv.ravel(), uu.ravel()], order=1, mode="constant", cval=background).reshape(uu.shape)
    return (uc, vc), angles, r, prof - background


def find_limb(image, A_est, s, background, read_noise, n_lines=180, margin_deg=15.0, min_points=20):
    """Limb points (K, 2) as (u, v) pixel coordinates, or an empty (0, 2) array: the 50 %-of-local-plateau rule.

    image: (H, W) counts. A_est: attitude (sky -> camera) from the star tracker. s: unit vector Earth -> Sun.
    Only the lit side of the disc is scanned: line angles within +-(90 - margin_deg) of the projected Sun
    direction. If the Sun is within margin_deg of the camera direction (phase ~0) the whole circle is lit and
    every direction is scanned; if the phase is within margin_deg of 180 nothing is lit.
    Unbiased where the limb is a sharp step; biased by up to ~1 px where Lambertian shading fades (phase ~0,
    thin crescents). `refine_limb` removes that bias given a first position estimate.
    """
    empty = np.zeros((0, 2))
    scan = scan_lines(image, A_est, s, background, read_noise, n_lines, margin_deg)
    if scan is None:
        return empty
    (uc, vc), angles, r, prof = scan
    pts = []
    coarse_level = SEG_SIGMA * read_noise
    n_in, n_w = int(round(PLATEAU_IN / STEP)), int(round(PLATEAU_WIDTH / STEP))
    for i in range(len(angles)):
        line = prof[i]
        above = np.flatnonzero(line > coarse_level)
        if len(above) == 0:
            continue
        j_coarse = above[-1]
        if j_coarse >= len(r) - 2 or j_coarse - n_in - n_w < 0:
            continue                                               # ran off the scan, or no room for a plateau
        plateau = line[j_coarse - n_in - n_w: j_coarse - n_in].mean()
        if plateau < MIN_CONTRAST * read_noise:
            continue                                               # too faint (terminator side, PSF tail)
        half_level = 0.5 * plateau
        inner = np.flatnonzero(line[: j_coarse + 3] > half_level)
        if len(inner) == 0:
            continue
        j = inner[-1]                                              # outermost sample above the half level
        if j + 1 >= len(r) or line[j] == line[j + 1]:
            continue
        frac = (line[j] - half_level) / (line[j] - line[j + 1])
        r_edge = r[j] + frac * STEP
        pts.append((uc + r_edge * np.cos(angles[i]), vc + r_edge * np.sin(angles[i])))
    pts = np.array(pts) if pts else empty
    return pts if len(pts) >= min_points else empty


FINE = 0.05            # px: sampling of the predicted profile before blurring
SHIFTS = np.arange(-4.0, 4.0 + 1e-9, 0.05)    # px: edge shifts tried per line
WINDOW_IN, WINDOW_OUT = 14.0, 8.0             # px: fit window around the predicted edge


def refine_limb(image, A_est, s, p0, background, read_noise, R=6378.137, sigma_psf=SIGMA_PSF,
                n_lines=180, margin_deg=15.0, min_points=20, min_contrast=MIN_CONTRAST):
    """Limb points by fitting the PREDICTED blurred edge profile on every scan line.

    p0: a first position estimate (km) from the 50 % rule; it supplies only the SHAPE of the Lambertian fade
    along each line (how brightness falls toward the limb), which is what biases a threshold crossing.
    Per line the model is  I(r) = A * [PSF * shape](r - delta): shape = Lambertian (n . s)+ of the sphere at p0
    along the line (0 beyond its limb), convolved with the Gaussian PSF; A (brightness) is linear, the shift
    delta is found on a 0.05 px grid and refined by a parabola. The edge = model edge + delta.
    """
    empty = np.zeros((0, 2))
    scan = scan_lines(image, A_est, s, background, read_noise, n_lines, margin_deg)
    if scan is None:
        return empty
    (uc, vc), angles, r, prof = scan
    p0 = np.asarray(p0, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    r_fine = np.arange(0.0, r[-1] + WINDOW_OUT + 6 * sigma_psf, FINE)
    kern_x = np.arange(-4 * sigma_psf, 4 * sigma_psf + 1e-9, FINE)
    kern = np.exp(-0.5 * (kern_x / sigma_psf) ** 2)
    kern /= kern.sum()
    pts = []
    for i, ang in enumerate(angles):
        # predicted unblurred shape along this line for a sphere at p0
        u = uc + r_fine * np.cos(ang)
        v = vc + r_fine * np.sin(ang)
        d = unproject(u, v) @ A_est
        hit, t = _ray_hits(p0, d, R, R)
        if hit.sum() < 2:
            continue
        shape = np.zeros_like(r_fine)
        x = p0 + t[hit, None] * d[hit]
        shape[hit] = np.maximum(0.0, (x / np.linalg.norm(x, axis=1, keepdims=True)) @ s)
        r_edge0 = r_fine[hit].max()                                # the model limb along this line
        blurred = np.convolve(shape, kern, mode="same")
        # observed samples in a window around the predicted edge
        win = (r > r_edge0 - WINDOW_IN) & (r < r_edge0 + WINDOW_OUT) & (r < r[-1] - 1.0)
        if win.sum() < 10:
            continue
        obs = prof[i, win]
        if obs.max() < min_contrast * read_noise:
            continue                                               # too faint to fit (terminator side)
        r_obs = r[win]
        # try every shift: model(r) = A * blurred(r - delta); A by linear least squares; pick the best residual
        best = None
        for delta in SHIFTS:
            m = np.interp(r_obs - delta, r_fine, blurred, left=blurred[0], right=0.0)
            mm = m @ m
            if mm <= 0:
                continue
            A_amp = (m @ obs) / mm
            res = float(((obs - A_amp * m) ** 2).sum())
            if best is None or res < best[0]:
                best = (res, delta, A_amp)
        if best is None or best[2] <= 0:
            continue
        # parabolic refinement of delta around the best grid point
        k = int(round((best[1] - SHIFTS[0]) / 0.05))
        if 0 < k < len(SHIFTS) - 1:
            resid = []
            for delta in SHIFTS[k - 1: k + 2]:
                m = np.interp(r_obs - delta, r_fine, blurred, left=blurred[0], right=0.0)
                A_amp = (m @ obs) / (m @ m)
                resid.append(float(((obs - A_amp * m) ** 2).sum()))
            den = resid[0] - 2 * resid[1] + resid[2]
            delta = best[1] + (0.05 * 0.5 * (resid[0] - resid[2]) / den if den > 0 else 0.0)
        else:
            delta = best[1]
        r_edge = r_edge0 + delta
        pts.append((uc + r_edge * np.cos(ang), vc + r_edge * np.sin(ang)))
    pts = np.array(pts) if pts else empty
    return pts if len(pts) >= min_points else empty
