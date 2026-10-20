"""The ONLY Python module that touches CuPy. Loads tracker/kernels/*.cu and launches them.

Images stay on the GPU (CuPy arrays); only small star lists cross back to the host.
float32 on the GPU; the attitude solve stays float64 on the CPU.
"""
import functools
from pathlib import Path

import cupy as cp
import numpy as np

from tracker.camera import N_PIX
from tracker.render_cpu import BACKGROUND, READ_NOISE, SIGMA_PSF

KERNELS = Path(__file__).resolve().parent / "kernels"
BLOCK = 16                       # 16 x 16 threads per block (spec)


@functools.cache
def _kernel(filename, name, options=()):
    """Compile once per process (NVRTC); later calls reuse the compiled kernel."""
    return cp.RawKernel((KERNELS / filename).read_text(), name, options=tuple(options))


def render_batch(batch, seed=0, frame_ids=None, background=BACKGROUND, read_noise=READ_NOISE,
                 sigma_psf=SIGMA_PSF, n_pix=N_PIX):
    """StarBatch -> (F, n_pix, n_pix) float32 CuPy array of images, on the GPU.

    Noise is the counter-based noise of tracker.noise: frame f uses (seed, frame_ids[f]),
    so the CPU reference is render(..., noise_key=(seed, frame_ids[f])).
    frame_ids defaults to 0..F-1. read_noise=0 gives noise-free images.
    """
    F = batch.n_frames
    if n_pix % BLOCK:
        raise ValueError(f"n_pix must be a multiple of {BLOCK}")
    frame_ids = np.arange(F) if frame_ids is None else np.asarray(frame_ids)
    if len(frame_ids) != F:
        raise ValueError("need one frame id per frame")
    S = len(batch.du)
    # A batch with no stars at all still needs valid (non-null) device pointers.
    pad = lambda a, dt: cp.asarray(a if S else np.zeros(4 if dt == np.int32 else 1, dt), dtype=dt)
    out = cp.empty((F, n_pix, n_pix), dtype=cp.float32)
    _kernel("render.cu", "render_kernel")(
        (n_pix // BLOCK, n_pix // BLOCK, F), (BLOCK, BLOCK),
        (cp.asarray(batch.offsets, dtype=cp.int32),
         pad(batch.du, np.float32), pad(batch.dv, np.float32), pad(batch.amp, np.float32),
         pad(batch.box.ravel(), np.int32),
         cp.asarray(frame_ids, dtype=cp.uint32),
         np.float32(background), np.float32(read_noise),
         np.float32(1.0 / (2.0 * sigma_psf**2)), np.uint32(seed), np.int32(n_pix),
         out))
    return out


DETECT_CAPACITY = 256            # detection slots per frame (spec said 64; real frames reach 83+)


def detect_batch(images, background, read_noise, k_sigma=5, capacity=DETECT_CAPACITY):
    """(F, n_pix, n_pix) CuPy images -> list of F (u, v, flux) NumPy float64 tuples, brightest
    first: the same contract as detect_cpu.detect. Only the star lists leave the GPU.

    Raises RuntimeError if a frame has more than `capacity` detections: which ones would be
    kept depends on thread timing, so dropping them silently would make results irreproducible.
    """
    F, H, W = images.shape
    if H != W or H % BLOCK:
        raise ValueError(f"images must be square, a multiple of {BLOCK} px")
    images = cp.ascontiguousarray(images, dtype=cp.float32)
    threshold = np.float32(background + k_sigma * read_noise)      # same value as detect_cpu
    count = cp.zeros(F, dtype=cp.int32)
    out = cp.empty((F, capacity, 5), dtype=cp.float32)
    _kernel("detect.cu", "detect_kernel")(
        (H // BLOCK, H // BLOCK, F), (BLOCK, BLOCK),
        (images, np.float32(background), threshold, np.int32(H), np.int32(capacity), count, out))
    count = cp.asnumpy(count)
    if (count > capacity).any():
        bad = np.flatnonzero(count > capacity)
        raise RuntimeError(f"detection capacity {capacity} exceeded in frames {bad[:10].tolist()} "
                           f"(max {count.max()}); raise capacity")
    out = cp.asnumpy(out)                                           # (F, capacity, 5), small
    found = []
    for f in range(F):
        d = out[f, :count[f]].astype(np.float64)
        u, v, flux = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3], d[:, 4]
        order = np.lexsort((u, v, -flux))       # brightest first; ties broken by position, so stable
        found.append((u[order], v[order], flux[order]))
    return found


def add_hot_pixels_batch(images, draws):
    """Add hot pixels in place on the GPU. draws: one (flat_idx, values) per frame, from
    render_cpu.draw_hot_pixels, so the CPU reference adds exactly the same values.
    Indices within a frame are distinct, so a plain scatter-add is safe."""
    F, H, W = images.shape
    if len(draws) != F:
        raise ValueError("need one draw per frame")
    idx = [f * H * W + np.asarray(d[0], dtype=np.int64) for f, d in enumerate(draws)]
    val = [np.asarray(d[1], dtype=np.float32) for d in draws]
    if sum(len(i) for i in idx):
        flat = images.reshape(-1)                                  # a view: writes go to images
        g = cp.asarray(np.concatenate(idx))
        flat[g] += cp.asarray(np.concatenate(val))
    return images


def render_earth_batch(positions, attitudes, suns, a=None, c=None, supersample=4, brightness=None, shade=True,
                       sigma_psf=None, n_pix=N_PIX, precision="double"):
    """Earth for F frames on the GPU -> (images (F, n_pix, n_pix) float32 CuPy, masks (F, n_pix, n_pix) bool CuPy).
    Same contract as earth_render.render_earth per frame: Earth only, no background or noise, PSF-blurred.
    positions (F, 3) km, attitudes (F, 3, 3), suns (F, 3). Bounding boxes come from the host (earth_render.earth_bbox),
    one kernel launch per frame over its box (16 x 16 threads per block).
    precision: "double" (the shipped kernel: the limb is a hit/miss cliff at |p| ~ 1e5 km) or "float32", kept only to
    measure what single precision costs in limb accuracy and what double costs in time (Day 12 log)."""
    from cupyx.scipy.ndimage import gaussian_filter
    from tracker.camera import C_PIX, F_PIX
    from tracker.earth_render import earth_bbox
    from tracker.world import EARTH_LEVEL, R_EQ
    a = R_EQ if a is None else a
    c = a if c is None else c
    brightness = EARTH_LEVEL if brightness is None else brightness
    sigma_psf = SIGMA_PSF if sigma_psf is None else sigma_psf
    positions, attitudes, suns = (np.asarray(x, dtype=np.float64) for x in (positions, attitudes, suns))
    F = len(positions)
    images = cp.zeros((F, n_pix, n_pix), dtype=cp.float32)
    masks = cp.zeros((F, n_pix, n_pix), dtype=cp.uint8)
    if precision not in ("double", "float32"):
        raise ValueError("precision must be 'double' or 'float32'")
    kernel = _kernel("earth.cu", "earth_kernel", ("-DREAL_T=float",) if precision == "float32" else ())
    for f in range(F):
        box = earth_bbox(positions[f], attitudes[f], a, c, n_pix)
        if box is None:
            continue
        u0, u1, v0, v1 = box
        bw, bh = u1 - u0 + 1, v1 - v0 + 1
        grid = ((bw + BLOCK - 1) // BLOCK, (bh + BLOCK - 1) // BLOCK)
        kernel(grid, (BLOCK, BLOCK),
               (cp.asarray(positions[f]), cp.asarray(attitudes[f].ravel()), cp.asarray(suns[f]),
                np.float64(a), np.float64(c), np.float64(F_PIX), np.float64(C_PIX),
                np.int32(u0), np.int32(v0), np.int32(bw), np.int32(bh), np.int32(n_pix), np.int32(supersample),
                np.float64(brightness), np.int32(1 if shade else 0), images[f], masks[f]))
    if sigma_psf > 0:
        for f in range(F):
            images[f] = gaussian_filter(images[f], sigma_psf, truncate=4.0)
    return images, masks.astype(cp.bool_)
