"""CPU reference star detector: image -> list of star centroids (u, v, flux).

Two stages:
  1. candidates: pixels brighter than background + k_sigma * read_noise that are also
     the local maximum of their 3x3 neighbourhood.
  2. centroid: background-subtracted, brightness-weighted mean over a 5x5 window
     centred on each candidate pixel.

Image convention: image[v, u] (row, column), same as render_cpu.
The GPU kernel on Day 4 (detect.cu) must reproduce this exactly.
"""
import numpy as np

CENTROID_HALF = 2      # 5x5 centroid window: +-2 px around the peak pixel

# 3x3 neighbours, split by raster order (row by row, left to right).
# A candidate must be STRICTLY brighter than the neighbours that come before it and
# AT LEAST as bright as those that come after it. If two neighbouring pixels are exactly
# equal, only the earlier one passes, so a flat-topped star is counted exactly once.
_BEFORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
_AFTER = [(0, 1), (1, -1), (1, 0), (1, 1)]


def detect(image, background, read_noise, k_sigma=5):
    """Find stars in one image.

    image:       (H, W) array, counts. Indexed image[v, u].
    background:  flat background level, counts per pixel.
    read_noise:  per-pixel Gaussian noise sigma, counts.
    k_sigma:     detection threshold in units of read_noise above background.

    Returns (u, v, flux), float64 arrays of length N, sorted brightest first.
      u, v: centroid in pixels (column, row); flux: background-subtracted 5x5 sum, counts.
    """
    image = np.asarray(image)
    H, W = image.shape
    threshold = background + k_sigma * read_noise

    # 1a. threshold: one pass over the whole image, leaves a few hundred pixels at most
    vv, uu = np.nonzero(image > threshold)

    # 1b. local-max test, only on those pixels. Pad by 1 with -inf so edge pixels
    #     have neighbours that can never win.
    P = np.pad(image, 1, constant_values=-np.inf)
    c = image[vv, uu]
    keep = np.ones(len(vv), dtype=bool)
    for dv, du in _BEFORE:
        keep &= c > P[vv + 1 + dv, uu + 1 + du]
    for dv, du in _AFTER:
        keep &= c >= P[vv + 1 + dv, uu + 1 + du]
    vv, uu = vv[keep], uu[keep]

    # 2. centroid over a 5x5 window. Pad with the background value, so window pixels
    #    off the sensor contribute exactly 0 after background subtraction.
    h = CENTROID_HALF
    Q = np.pad(image.astype(np.float64), h, constant_values=float(background))
    d = np.arange(-h, h + 1)
    rows = vv[:, None, None] + h + d[None, :, None]           # (N, 5, 1)
    cols = uu[:, None, None] + h + d[None, None, :]           # (N, 1, 5)
    w = Q[rows, cols] - background                            # (N, 5, 5) weights
    flux = w.sum(axis=(1, 2))
    # Negative weights (noise below background) are kept: clipping them to 0 would add
    # a positive bias everywhere and drag every centroid toward the window centre.
    ok = flux > 0
    w, flux, vv, uu = w[ok], flux[ok], vv[ok], uu[ok]
    u = uu + (w.sum(axis=1) * d).sum(axis=1) / flux           # sum over rows -> column profile
    v = vv + (w.sum(axis=2) * d).sum(axis=1) / flux           # sum over cols -> row profile

    order = np.argsort(-flux, kind="stable")
    return u[order], v[order], flux[order]
