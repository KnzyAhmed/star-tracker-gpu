"""Counter-based Gaussian noise: the same numbers on the CPU (NumPy) and the GPU (CUDA).

A normal RNG (rng.normal) is a sequence: number k depends on numbers 0..k-1, so a million
GPU threads can't each take "their" number. Here each pixel's noise is a pure function of
(seed, frame, pixel), so any thread can compute its own, in any order:

    key  = hash32(hash32(seed + GOLDEN) + frame)
    h1   = hash32(key + 2 * pixel),  h2 = hash32(key + 2 * pixel + 1)
    u1   = ((h1 >> 8) + 1) / 2^24       in (0, 1]   (never 0: log(0) = -inf)
    u2   =  (h2 >> 8)      / 2^24       in [0, 1)
    z    = sqrt(-2 ln u1) * cos(2 pi u2)              (Box-Muller, one normal per pixel)

pixel = v * n_pix + u (row-major, same as image.ravel()). Every step uses only uint32
arithmetic that wraps mod 2^32 and float32 maths, which NumPy and CUDA do identically
(up to ~1 ulp in log/cos). kernels/render.cu must implement exactly this.
"""
import numpy as np

from tracker.camera import N_PIX

GOLDEN = np.uint32(0x9E3779B9)       # 2^32 / golden ratio: breaks hash32(0) = 0 for seed 0
TWO_POW_M24 = np.float32(2.0**-24)
MAX_SIGMA = float(np.sqrt(-2 * np.log(np.float32(2.0**-24))))   # 5.77: largest |z| possible


def hash32(x):
    """'lowbias32' integer hash (C. Wellons): uint32 array -> uint32 array, well mixed.
    Multiplication of uint32 arrays wraps mod 2^32 in NumPy, exactly like `unsigned int` in C."""
    x = np.asarray(x, dtype=np.uint32).copy()
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def pixel_normals(seed, frame, n_pix=N_PIX):
    """(n_pix, n_pix) float32 standard normals for one frame, indexed [v, u]."""
    with np.errstate(over="ignore"):
        key = hash32(hash32(np.uint32(seed) + GOLDEN) + np.uint32(frame))
        pixel = np.arange(n_pix * n_pix, dtype=np.uint32)
        h1 = hash32(key + np.uint32(2) * pixel)
        h2 = hash32(key + np.uint32(2) * pixel + np.uint32(1))
    u1 = ((h1 >> np.uint32(8)) + np.uint32(1)).astype(np.float32) * TWO_POW_M24
    u2 = (h2 >> np.uint32(8)).astype(np.float32) * TWO_POW_M24
    z = np.sqrt(np.float32(-2) * np.log(u1)) * np.cos(np.float32(2 * np.pi) * u2)
    return z.astype(np.float32).reshape(n_pix, n_pix)
