"""CPU reference renderer: star positions on the sensor -> a noisy image.

This is the slow-but-trustworthy version. The GPU kernel on Day 4 must match it.

Image convention: image[row, col] = image[v, u]. Row 0 is the top of the picture.
"""
import numpy as np

from tracker.camera import N_PIX, project
from tracker.noise import pixel_normals

SIGMA_PSF = 1.2        # star blur: Gaussian width in pixels (real trackers defocus on purpose)
BACKGROUND = 20.0      # sky + sensor background level, counts per pixel
READ_NOISE = 5.0       # Gaussian noise added to every pixel, counts
F0 = 60_000.0          # total counts from a magnitude-0 star
HALF_WINDOW = 5        # draw each star only within +-5 px of its centre (about 4 sigma)


def star_flux(mag):
    """Magnitude -> total counts. Each magnitude step is a factor 10**0.4 ~ 2.512 dimmer."""
    flux = F0 * 10**(-0.4 * mag)
    return flux

def star_windows(u, v, n_pix=N_PIX):
    """Integer pixel box each star is drawn in: +-HALF_WINDOW around it, clipped to the sensor.
    Returns int arrays (u_min, u_max, v_min, v_max), inclusive. Computed ONCE on the host and
    handed to the GPU too, so both draw exactly the same pixels (no float32 floor/ceil surprises)."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    u_min = np.maximum(np.floor(u - HALF_WINDOW), 0).astype(np.int32)
    u_max = np.minimum(np.ceil(u + HALF_WINDOW), n_pix - 1).astype(np.int32)
    v_min = np.maximum(np.floor(v - HALF_WINDOW), 0).astype(np.int32)
    v_max = np.minimum(np.ceil(v + HALF_WINDOW), n_pix - 1).astype(np.int32)
    return u_min, u_max, v_min, v_max


def render(u, v, flux, rng=None, sigma_psf=SIGMA_PSF, background=BACKGROUND,
           read_noise=READ_NOISE, n_pix=N_PIX, noise_key=None):
    """Draw stars at sub-pixel positions (u, v) with total brightness `flux`.

    u, v, flux: arrays of length N (N stars already known to be on the sensor).
    Returns a float32 image of shape (n_pix, n_pix).

    Steps:
      1. start from an image full of `background`
      2. for each star, add a 2-D Gaussian blob centred at (u, v) whose pixels
         sum to `flux`:  value = flux / (2 pi sigma^2) * exp(-(dx^2 + dy^2) / (2 sigma^2))
         Only touch pixels within HALF_WINDOW of the centre, and stay inside the image.
      3. if rng is given and read_noise > 0, add Gaussian noise to every pixel.
         OR, with noise_key=(seed, frame), add read_noise * pixel_normals(seed, frame):
         the counter-based noise the GPU reproduces (slower on CPU; used to compare).
    """
    if rng is not None and noise_key is not None:
        raise ValueError("give rng or noise_key, not both")
    image = np.full((n_pix, n_pix), background, dtype=np.float32)
    U_MIN, U_MAX, V_MIN, V_MAX = star_windows(u, v, n_pix)
    for star in range(len(u)):
        u0, v0, f = u[star], v[star], flux[star]
        u_min, u_max, v_min, v_max = U_MIN[star], U_MAX[star], V_MIN[star], V_MAX[star]
        dx = np.arange(u_min, u_max + 1)[None, :] - u0
        dy = np.arange(v_min, v_max + 1)[:, None] - v0
        image[v_min:v_max + 1, u_min:u_max + 1] += f / (2 * np.pi * sigma_psf**2) * np.exp(-(dx**2 + dy**2) / (2 * sigma_psf**2))
    if rng is not None and read_noise > 0:
        image += rng.normal(0, read_noise, size=image.shape)
    if noise_key is not None and read_noise > 0:
        image += np.float32(read_noise) * pixel_normals(*noise_key, n_pix=n_pix)
    return image

def add_hot_pixels(image, n, rng, lo=100.0, hi=2000.0):
    """Add n hot pixels (fake stars): n DISTINCT random pixels each get +uniform(lo, hi) counts.
    A real sensor has a few of these; they are single-pixel spikes with no PSF. Modifies and returns image."""
    flat = rng.choice(image.size, size=n, replace=False)
    image.flat[flat] += rng.uniform(lo, hi, size=n).astype(image.dtype)
    return image


def render_sky(A, catalog, rng=None, **kwargs):
    """Full pipeline for one frame: attitude + catalog -> image.

    Returns (image, u, v, idx) where idx are the catalog indices of the visible stars
    and u, v their true pixel positions. The truth is what we score against later.
    """
    u, v, visible = project(A, catalog.r)
    idx = np.flatnonzero(visible)
    image = render(u[idx], v[idx], star_flux(catalog.mag[idx]), rng=rng, **kwargs)
    return image, u[idx], v[idx], idx