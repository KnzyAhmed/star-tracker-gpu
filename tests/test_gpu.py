"""Day 4 gate: GPU kernels vs the CPU reference. Needs an NVIDIA GPU + CuPy (Colab);
skipped elsewhere.  Run:  pytest tests/test_gpu.py -v"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from tracker import gpu                                     # noqa: E402
from tracker.batch import pack_frames, project_frames        # noqa: E402
from tracker.camera import random_attitude                   # noqa: E402
from tracker.catalog import load_catalog                     # noqa: E402
from tracker.render_cpu import BACKGROUND, READ_NOISE, render  # noqa: E402

RTOL, ATOL = 1e-5, 1e-3      # float32: a few ulp of accumulated rounding per pixel


@pytest.fixture(scope="module")
def sky():
    cat = load_catalog(5.5)
    rng = np.random.default_rng(0)
    frames, idx = project_frames([random_attitude(rng) for _ in range(8)], cat)
    return frames


def cpu_images(frames, **kw):
    return np.stack([render(u, v, f, **kw) for (u, v, f) in frames])


# ---------------------------------------------------------------- render

def test_render_noise_off_matches_cpu(sky):
    img = gpu.render_batch(pack_frames(sky), read_noise=0.0)
    assert isinstance(img, cp.ndarray) and img.shape == (8, 1024, 1024) and img.dtype == cp.float32
    np.testing.assert_allclose(cp.asnumpy(img), cpu_images(sky, read_noise=0.0), rtol=RTOL, atol=ATOL)


def test_render_counter_noise_matches_cpu(sky):
    """Same (seed, frame) -> same noise on both machines."""
    frame_ids = np.arange(100, 108)
    img = gpu.render_batch(pack_frames(sky), seed=5, frame_ids=frame_ids)
    cpu = np.stack([render(u, v, f, noise_key=(5, int(k))) for (u, v, f), k in zip(sky, frame_ids)])
    np.testing.assert_allclose(cp.asnumpy(img), cpu, rtol=RTOL, atol=ATOL)


def test_render_more_than_64_stars_in_a_frame():
    """The star list is processed in shared-memory chunks of 64; 150 stars = 3 chunks."""
    rng = np.random.default_rng(1)
    frame = (rng.uniform(0, 1023, 150), rng.uniform(0, 1023, 150), rng.uniform(100, 5000, 150))
    img = gpu.render_batch(pack_frames([frame]), read_noise=0.0)
    np.testing.assert_allclose(cp.asnumpy(img[0]), render(*frame, read_noise=0.0), rtol=RTOL, atol=ATOL)


def test_render_empty_frame_is_background():
    empty = (np.array([]), np.array([]), np.array([]))
    img = gpu.render_batch(pack_frames([empty, empty]), read_noise=0.0)
    assert float(img.min()) == float(img.max()) == BACKGROUND


def test_render_noise_statistics():
    empty = (np.array([]), np.array([]), np.array([]))
    img = cp.asnumpy(gpu.render_batch(pack_frames([empty]), seed=1, frame_ids=[0]))
    assert abs(img.mean() - BACKGROUND) < 0.05 and abs(img.std() - READ_NOISE) < 0.05
