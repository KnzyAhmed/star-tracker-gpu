"""Day 4: counter-based noise, identical on CPU and GPU. Run:  pytest tests/test_noise.py -q"""
import numpy as np

from tracker.camera import N_PIX
from tracker.noise import MAX_SIGMA, hash32, pixel_normals

M32 = 0xFFFFFFFF


def hash32_reference(x):
    """Same hash written with plain Python ints (no NumPy overflow behaviour involved)."""
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    x ^= x >> 16
    return x


def test_hash32_matches_plain_integer_maths():
    xs = [0, 1, 2, 12345, 2**31, M32] + list(np.random.default_rng(0).integers(0, 2**32, 50))
    got = hash32(np.array(xs, dtype=np.uint32))
    assert got.dtype == np.uint32
    assert [int(g) for g in got] == [hash32_reference(int(x)) for x in xs]


def test_shape_dtype_and_reproducible():
    z = pixel_normals(seed=7, frame=3)
    assert z.shape == (N_PIX, N_PIX) and z.dtype == np.float32
    assert np.array_equal(z, pixel_normals(seed=7, frame=3))


def test_standard_normal_statistics():
    z = pixel_normals(seed=1, frame=0).astype(np.float64)
    assert abs(z.mean()) < 0.005 and abs(z.std() - 1) < 0.005
    assert abs(np.mean(np.abs(z) > 1) - 0.3173) < 0.002     # 1-sigma two-sided tail
    assert abs(np.mean(np.abs(z) > 3) - 0.0027) < 0.0003
    assert np.isfinite(z).all() and np.abs(z).max() <= MAX_SIGMA


def test_seed_zero_frame_zero_pixel_zero_is_not_special():
    """hash32(0) == 0. Without the GOLDEN offset, (seed 0, frame 0, pixel 0) hashes to 0, gets the
    smallest possible u1, and comes out at -4.84 sigma (checked), in the same pixel every run."""
    assert hash32(np.array([0], dtype=np.uint32))[0] == 0
    assert abs(pixel_normals(seed=0, frame=0)[0, 0]) < 4


def test_frames_seeds_and_neighbours_are_uncorrelated():
    a = pixel_normals(seed=1, frame=0).astype(np.float64)
    for other in (pixel_normals(seed=1, frame=1), pixel_normals(seed=2, frame=0)):
        assert abs(np.corrcoef(a.ravel(), other.ravel())[0, 1]) < 0.005
    assert abs(np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]) < 0.005    # left-right
    assert abs(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1]) < 0.005          # up-down
