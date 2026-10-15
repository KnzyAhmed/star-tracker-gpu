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
def _kernel(filename, name):
    """Compile once per process (NVRTC); later calls reuse the compiled kernel."""
    return cp.RawKernel((KERNELS / filename).read_text(), name)


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
