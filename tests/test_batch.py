"""Day 4: packing many frames' star lists for the GPU. Run:  pytest tests/test_batch.py -q"""
import numpy as np

from tracker.batch import pack_frames, project_frames
from tracker.camera import random_attitude
from tracker.catalog import load_catalog
from tracker.render_cpu import SIGMA_PSF, star_flux, star_windows


def three_frames():
    return [(np.array([10.0, 20.5]), np.array([30.0, 40.25]), np.array([100.0, 200.0])),
            (np.array([]), np.array([]), np.array([])),                      # a frame with no stars
            (np.array([1000.7]), np.array([2.2]), np.array([50.0]))]


def test_offsets_and_order():
    b = pack_frames(three_frames())
    assert b.offsets.tolist() == [0, 2, 2, 3]
    assert b.offsets.dtype == np.int32
    assert b.du.dtype == b.dv.dtype == b.amp.dtype == np.float32
    assert b.n_frames == 3


def test_amplitude_is_peak_of_the_gaussian():
    """amp = flux / (2 pi sigma^2), the value at the centre of the blob."""
    b = pack_frames(three_frames())
    assert np.allclose(b.amp, np.array([100.0, 200.0, 50.0]) / (2 * np.pi * SIGMA_PSF**2), rtol=1e-6)


def test_boxes_are_the_cpu_windows():
    frames = three_frames()
    b = pack_frames(frames)
    u = np.concatenate([f[0] for f in frames])
    v = np.concatenate([f[1] for f in frames])
    assert b.box.dtype == np.int32 and b.box.shape == (3, 4)
    assert np.array_equal(b.box, np.column_stack(star_windows(u, v)))


def test_positions_are_relative_to_the_box_and_precise():
    """du = u - u_min. Absolute u = 1000.7 in float32 is off by 1.2e-5 px; relative 5.7 by 1.9e-7 (measured)."""
    frames = three_frames()
    b = pack_frames(frames)
    u = np.concatenate([f[0] for f in frames])
    v = np.concatenate([f[1] for f in frames])
    assert np.allclose(b.du + b.box[:, 0], u, atol=1e-5) and np.allclose(b.dv + b.box[:, 2], v, atol=1e-5)
    assert np.abs(b.du.astype(np.float64) - (u - b.box[:, 0])).max() < 1e-6


def test_project_frames_matches_render_sky_truth():
    cat = load_catalog(5.5)
    rng = np.random.default_rng(0)
    As = [random_attitude(rng) for _ in range(4)]
    frames, idx = project_frames(As, cat)
    assert len(frames) == len(idx) == 4
    for (u, v, f), k in zip(frames, idx):
        assert len(u) == len(v) == len(f) == len(k)
        assert np.allclose(f, star_flux(cat.mag[k]))
