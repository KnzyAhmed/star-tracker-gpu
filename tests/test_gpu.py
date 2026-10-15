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


# ---------------------------------------------------------------- detect

from tracker.detect_cpu import detect                       # noqa: E402
from tracker.render_cpu import add_hot_pixels              # noqa: E402


def assert_same_stars(gpu_out, cpu_out, pos_tol=1e-4, flux_rtol=1e-5):
    (ug, vg, fg), (uc, vc, fc) = gpu_out, cpu_out
    assert len(ug) == len(uc)
    order_g, order_c = np.lexsort((ug, vg)), np.lexsort((uc, vc))   # compare by position, not flux order
    assert np.abs(ug[order_g] - uc[order_c]).max(initial=0) < pos_tol
    assert np.abs(vg[order_g] - vc[order_c]).max(initial=0) < pos_tol
    np.testing.assert_allclose(fg[order_g], fc[order_c], rtol=flux_rtol, atol=1e-2)


def test_detect_matches_cpu_on_the_same_noisy_images(sky):
    """Upload CPU-rendered images (rng noise) and run both detectors on identical pixels."""
    rng = np.random.default_rng(2)
    imgs = np.stack([render(u, v, f, rng=rng) for (u, v, f) in sky])
    found = gpu.detect_batch(cp.asarray(imgs), BACKGROUND, READ_NOISE)
    assert len(found) == len(sky)
    for k in range(len(sky)):
        assert_same_stars(found[k], detect(imgs[k], BACKGROUND, READ_NOISE))
        assert np.all(np.diff(found[k][2]) <= 0)                     # brightest first, like the CPU


def test_detect_flat_top_and_edges():
    img = np.full((1024, 1024), BACKGROUND, dtype=np.float32)
    img[500, 500] = img[500, 501] = 1000.0          # tie: one star, not two
    img[0, 0] = 900.0                               # corner: window off the sensor
    img[1023, 600] = 800.0                          # bottom edge
    found = gpu.detect_batch(cp.asarray(img[None]), BACKGROUND, READ_NOISE)[0]
    assert_same_stars(found, detect(img, BACKGROUND, READ_NOISE))
    assert len(found[0]) == 3


def test_threshold_rounding_matches_cpu():
    """read_noise 5.13 -> threshold 45.65, which float32 rounds UP to 45.650001525878906. Both
    detectors compare pixel > float32(threshold): a pixel holding exactly that value is NOT
    detected, the next float32 above it is."""
    rn = 5.13
    t32 = np.float32(BACKGROUND + 5 * rn)
    assert float(t32) > BACKGROUND + 5 * rn                          # the rounding case we want
    img = np.full((1024, 1024), BACKGROUND, dtype=np.float32)
    img[300, 300] = t32                                              # equal: not detected
    img[600, 600] = np.nextafter(t32, np.float32(1e9))               # one ulp above: detected
    cpu = detect(img, BACKGROUND, rn)
    assert len(cpu[0]) == 1 and cpu[1][0] == 600
    assert_same_stars(gpu.detect_batch(cp.asarray(img[None]), BACKGROUND, rn)[0], cpu)


def test_render_then_detect_end_to_end_matches_cpu(sky):
    """GPU render (counter noise) -> GPU detect, against CPU render(noise_key) -> CPU detect."""
    ids = np.arange(len(sky))
    found = gpu.detect_batch(gpu.render_batch(pack_frames(sky), seed=9, frame_ids=ids), BACKGROUND, READ_NOISE)
    for (u, v, f), k in zip(sky, ids):
        cpu = detect(render(u, v, f, noise_key=(9, int(k))), BACKGROUND, READ_NOISE)
        assert_same_stars(found[k], cpu, pos_tol=1e-3, flux_rtol=1e-4)


def test_detect_overflow_fails_loudly():
    """Too many detections for the output buffer must raise, never silently drop arbitrary stars."""
    img = add_hot_pixels(np.full((1024, 1024), BACKGROUND, dtype=np.float32), 400, np.random.default_rng(3))
    with pytest.raises(RuntimeError, match="capacity"):
        gpu.detect_batch(cp.asarray(img[None]), BACKGROUND, READ_NOISE, capacity=256)
