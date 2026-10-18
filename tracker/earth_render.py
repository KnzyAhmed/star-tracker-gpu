"""CPU reference renderer for Earth: a sunlit ellipsoid seen by the pinhole camera.

For every pixel in the disc's bounding box, `supersample` x `supersample` sub-pixel rays are cast
from the spacecraft. Space is scaled by S = diag(1/a, 1/a, 1/c) so the ellipsoid becomes the unit
sphere and the ray-sphere quadratic applies; the hit point is shaded Lambertian, max(0, n . s), with
the TRUE ellipsoid normal (the gradient of x^T S^2 x). Sub-rays are averaged, then the same Gaussian
PSF as the stars is applied. The GPU kernel on Day 11 must reproduce this.

Image convention as everywhere: image[v, u], row 0 at the top, world directions d = b @ A.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from tracker.camera import N_PIX, project, unproject
from tracker.render_cpu import READ_NOISE, SIGMA_PSF, render, star_flux
from tracker.noise import pixel_normals
from tracker.world import EARTH_LEVEL, R_EQ, earth_in_view

HALO = 6     # extra pixels around the projected disc for the PSF tail (5 sigma) and sub-pixel edges


def _ray_hits(p, d, a, c):
    """Ray p + t d against the ellipsoid x^2/a^2 + y^2/a^2 + z^2/c^2 = 1, for many rays at once.
    p: (3,) km. d: (M, 3) directions (any length). Returns (hit (M,) bool, t (M,) km: nearest hit)."""
    inv = np.array([1.0 / a, 1.0 / a, 1.0 / c])
    ps, ds = p * inv, d * inv                           # scaled space: the ellipsoid is the unit sphere
    A = (ds * ds).sum(axis=1)
    B = 2.0 * ds @ ps
    C = ps @ ps - 1.0
    disc = B * B - 4.0 * A * C
    hit = disc > 0
    t = np.full(len(d), np.inf)
    t[hit] = (-B[hit] - np.sqrt(disc[hit])) / (2.0 * A[hit])
    hit &= t > 0
    return hit, t


def occluded(p, r, a=R_EQ, c=R_EQ):
    """Which star directions r (N, 3), unit, world frame are hidden behind Earth for a camera at p."""
    return _ray_hits(np.asarray(p, dtype=np.float64), np.asarray(r, dtype=np.float64), a, c)[0]


def subpixel_offsets(supersample):
    """Sub-ray offsets within a pixel: (k + 0.5)/n - 0.5 for k = 0..n-1 (the same list on CPU and GPU)."""
    return (np.arange(supersample) + 0.5) / supersample - 0.5


def earth_bbox(p, A, a, c, n_pix=N_PIX):
    """Pixel bounding box (u0, u1, v0, v1), inclusive, that contains the projected disc plus the PSF halo, or None
    if Earth is not in the frame. Computed on the HOST for both the CPU and the GPU renderer, so they draw the
    same pixels. Off-axis the disc stretches by 1/cos^2(theta) (theta <= the field's half-diagonal)."""
    p = np.asarray(p, dtype=np.float64)
    visible, (uc, vc), rho, pr = earth_in_view(p, A, R=max(a, c), n_pix=n_pix)
    if not visible:
        return None
    b_centre = A @ (-p / np.linalg.norm(p))
    stretch = 1.0 / max(b_centre[2], 0.5) ** 2
    half = pr * stretch + HALO
    u0, u1 = max(int(np.floor(uc - half)), 0), min(int(np.ceil(uc + half)), n_pix - 1)
    v0, v1 = max(int(np.floor(vc - half)), 0), min(int(np.ceil(vc + half)), n_pix - 1)
    if u1 < u0 or v1 < v0:
        return None
    return u0, u1, v0, v1


def render_earth(p, A, s, a=R_EQ, c=R_EQ, supersample=4, brightness=EARTH_LEVEL, shade=True,
                 n_pix=N_PIX, sigma_psf=SIGMA_PSF):
    """Earth as the camera sees it. p: spacecraft position (km), A: attitude (sky -> camera),
    s: unit vector Earth -> Sun. a, c: equatorial and polar radii (km); c = a is a sphere.
    shade=False draws a flat disc (geometry tests). Returns (image float32 (n_pix, n_pix) in counts,
    no background or noise; mask bool: at least one sub-ray of the pixel hit Earth)."""
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    image = np.zeros((n_pix, n_pix), dtype=np.float32)
    mask = np.zeros((n_pix, n_pix), dtype=bool)
    box = earth_bbox(p, A, a, c, n_pix)
    if box is None:
        return image, mask
    u0, u1, v0, v1 = box
    V, U = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    U, V = U.ravel().astype(np.float64), V.ravel().astype(np.float64)
    inv2 = np.array([1.0 / a**2, 1.0 / a**2, 1.0 / c**2])
    light = np.zeros(len(U))
    hits = np.zeros(len(U), dtype=np.int32)
    offsets = subpixel_offsets(supersample)
    for du in offsets:
        for dv in offsets:
            d = unproject(U + du, V + dv) @ A                       # world-frame ray directions, unit
            hit, t = _ray_hits(p, d, a, c)
            hits += hit
            if shade:
                x = p + t[hit, None] * d[hit]                      # hit points on the surface
                n = x * inv2                                       # gradient of the ellipsoid: the true normal
                n /= np.linalg.norm(n, axis=1, keepdims=True)
                light[hit] += np.maximum(0.0, n @ s)
            else:
                light[hit] += 1.0
    box = (brightness * light / supersample**2).reshape(v1 - v0 + 1, u1 - u0 + 1)
    image[v0:v1 + 1, u0:u1 + 1] = box
    mask[v0:v1 + 1, u0:u1 + 1] = (hits > 0).reshape(box.shape)
    if sigma_psf > 0:
        image = gaussian_filter(image, sigma_psf, truncate=4.0).astype(np.float32)
    return image, mask


def render_scene(p, A, s, catalog, rng=None, noise_key=None, a=R_EQ, c=R_EQ, read_noise=READ_NOISE, **earth_kw):
    """Stars + Earth + background + noise, one frame. Stars whose line of sight hits Earth are removed.
    Returns (image, earth_mask, u, v, idx): the drawn stars' true pixel positions and catalog indices."""
    u, v, vis = project(A, catalog.r)
    idx = np.flatnonzero(vis)
    idx = idx[~occluded(p, catalog.r[idx], a, c)]
    image = render(u[idx], v[idx], star_flux(catalog.mag[idx]), rng=rng, noise_key=noise_key, read_noise=read_noise)
    earth, mask = render_earth(p, A, s, a=a, c=c, **earth_kw)
    image += earth
    return image, mask, u[idx], v[idx], idx
