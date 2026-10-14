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
