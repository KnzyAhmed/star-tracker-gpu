"""Day 2: star detection and centroiding. Run:  pytest tests/test_detect.py -q"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from tracker.camera import N_PIX, attitude_from_pointing
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.render_cpu import BACKGROUND, READ_NOISE, render, render_sky


def stars(u, v, flux, rng=None):
    return render(np.atleast_1d(u).astype(float), np.atleast_1d(v).astype(float),
                  np.atleast_1d(flux).astype(float), rng=rng)


def flat(value=BACKGROUND):
    return np.full((N_PIX, N_PIX), value, dtype=np.float32)


def test_returns_three_equal_length_arrays():
    u, v, f = detect(stars(300.0, 400.0, 5000.0), BACKGROUND, READ_NOISE)
    assert len(u) == len(v) == len(f) == 1


@pytest.mark.parametrize("u0, v0", [(300.0, 400.0), (300.2, 400.1), (612.37, 101.81),
                                    (50.45, 900.45), (800.6, 800.3)])
def test_noise_free_centroid(u0, v0):
    """A 5x5 window clips the blob unevenly, pulling the answer toward the pixel centre
    by up to ~0.1 px at half-pixel offsets (pixel locking). 0.12 px allows for that."""
    u, v, f = detect(stars(u0, v0, 5000.0), BACKGROUND, READ_NOISE)
    assert len(u) == 1
    assert abs(u[0] - u0) < 0.12 and abs(v[0] - v0) < 0.12


def test_centred_star_is_exact():
    u, v, f = detect(stars(300.0, 400.0, 5000.0), BACKGROUND, READ_NOISE)
    assert abs(u[0] - 300.0) < 1e-4 and abs(v[0] - 400.0) < 1e-4


def test_flux_is_background_subtracted_window_sum():
    """5x5 window of a sigma=1.2 blob holds erf(2.5/(1.2*sqrt 2))^2 ~ 93 % of the light."""
    u, v, f = detect(stars(300.0, 400.0, 5000.0), BACKGROUND, READ_NOISE)
    assert 0.90 * 5000 < f[0] < 0.96 * 5000


def test_empty_noisy_sky_has_almost_no_detections():
    """5 sigma: P(noise > 5 sigma) = 2.9e-7 per pixel, so ~0.3 false stars per 1 Mpx frame."""
    rng = np.random.default_rng(0)
    u, v, f = detect(stars([], [], [], rng=rng), BACKGROUND, READ_NOISE)
    assert len(u) <= 2


def test_faint_star_below_threshold_is_not_detected():
    """Peak = flux / (2 pi sigma^2). Flux 150 -> peak 16.6 counts < 5 x 5 = 25 above background."""
    u, v, f = detect(stars(300.0, 400.0, 150.0), BACKGROUND, READ_NOISE)
    assert len(u) == 0


@pytest.mark.parametrize("dv, du", [(0, 1), (1, 0), (1, 1), (1, -1)])
def test_flat_topped_peak_is_counted_once(dv, du):
    """Two neighbouring pixels with EXACTLY the same value: one star, not two, not zero."""
    img = flat()
    img[500, 500] = img[500 + dv, 500 + du] = 1000.0
    u, v, f = detect(img, BACKGROUND, READ_NOISE)
    assert len(u) == 1
    assert abs(u[0] - (500 + du / 2)) < 1e-6 and abs(v[0] - (500 + dv / 2)) < 1e-6


def test_plateau_of_three_is_counted_once():
    img = flat()
    img[500, 500:503] = 1000.0
    assert len(detect(img, BACKGROUND, READ_NOISE)[0]) == 1


def test_two_separate_stars_sorted_brightest_first():
    u, v, f = detect(stars([200.0, 700.0], [300.0, 800.0], [2000.0, 8000.0]), BACKGROUND, READ_NOISE)
    assert len(u) == 2
    assert f[0] > f[1]
    assert abs(u[0] - 700.0) < 0.01 and abs(u[1] - 200.0) < 0.01


@pytest.mark.parametrize("u0, v0", [(0.0, 0.0), (1023.0, 512.0), (0.3, 1022.8), (512.0, 1.0)])
def test_star_at_the_edge(u0, v0):
    """Must not crash or wrap around; window pixels outside the sensor count as background."""
    u, v, f = detect(stars(u0, v0, 5000.0), BACKGROUND, READ_NOISE)
    assert len(u) == 1
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert abs(u[0] - u0) < 1.0 and abs(v[0] - v0) < 1.0


def test_k_sigma_changes_threshold():
    img = stars(300.0, 400.0, 400.0)       # peak ~ 44 counts above background
    assert len(detect(img, BACKGROUND, READ_NOISE, k_sigma=5)[0]) == 1
    assert len(detect(img, BACKGROUND, READ_NOISE, k_sigma=10)[0]) == 0


def test_orion_noisy_median_centroid_error():
    """The Day 2 gate on one real field: median error < 0.15 px."""
    cat = load_catalog(5.5)
    A = attitude_from_pointing(83.8, -1.0, 0.0)
    img, u_true, v_true, idx = render_sky(A, cat, rng=np.random.default_rng(1))
    u, v, f = detect(img, BACKGROUND, READ_NOISE)
    dist, j = cKDTree(np.column_stack([u_true, v_true])).query(np.column_stack([u, v]))
    matched = dist < 1.5
    assert matched.sum() >= 0.8 * len(idx)
    assert np.median(dist[matched]) < 0.15


def test_threshold_comparison_is_float32_whatever_the_input_type():
    """Same answer for Python-float and np.float64 inputs (the GPU compares in float32 too)."""
    rn = 5.13
    t32 = np.float32(BACKGROUND + 5 * rn)
    img = flat()
    img[300, 300] = t32                                              # equal to float32 threshold
    assert len(detect(img, BACKGROUND, rn)[0]) == 0
    assert len(detect(img, np.float64(BACKGROUND), np.float64(rn))[0]) == 0


def test_detect_mask_ignores_pixels():
    """Day 9: an optional mask of pixels to ignore (the Earth disc). Masked bright pixels produce no stars,
    stars outside the mask are unaffected, and mask=None reproduces the old behaviour exactly."""
    img = stars([200.0, 700.0], [300.0, 800.0], [2000.0, 8000.0])
    u0, v0, f0 = detect(img, BACKGROUND, READ_NOISE)
    assert np.array_equal(detect(img, BACKGROUND, READ_NOISE, mask=None)[0], u0)
    mask = np.zeros(img.shape, dtype=bool)
    mask[790:811, 690:711] = True                                  # hide the brighter star
    u, v, f = detect(img, BACKGROUND, READ_NOISE, mask=mask)
    assert len(u) == 1 and abs(u[0] - 200.0) < 0.01 and abs(f[0] - f0[1]) < 1e-6
    # a bright disc: without the mask its noisy edge makes false stars, with the mask none
    disc = flat()
    vv, uu = np.indices(disc.shape)
    disc[np.hypot(uu - 512, vv - 512) < 150] = 3000.0
    disc += np.random.default_rng(0).normal(0, READ_NOISE, disc.shape).astype(np.float32)
    assert len(detect(disc, BACKGROUND, READ_NOISE)[0]) > 50
    assert len(detect(disc, BACKGROUND, READ_NOISE, mask=np.hypot(uu - 512, vv - 512) < 158)[0]) <= 1
