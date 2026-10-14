"""Day 1b gate: the CPU renderer. Run:  pytest tests/test_render.py -q"""
import numpy as np

from tracker.camera import C_PIX, N_PIX, attitude_from_pointing
from tracker.catalog import load_catalog
from tracker.render_cpu import BACKGROUND, F0, READ_NOISE, render, render_sky, star_flux


def one_star(u, v, flux=10_000.0):
    """Noise-free, background-free image of a single star."""
    return render(np.array([u]), np.array([v]), np.array([flux]), rng=None, background=0.0)


def test_star_flux():
    assert np.isclose(star_flux(0.0), F0)
    assert np.isclose(star_flux(5.0), F0 / 100.0)          # 5 magnitudes = exactly 100x dimmer
    assert np.allclose(star_flux(np.array([1.0, 2.0])), F0 * 10 ** (-0.4 * np.array([1.0, 2.0])))


def test_image_shape_and_type():
    img = one_star(C_PIX, C_PIX)
    assert img.shape == (N_PIX, N_PIX)
    assert img.dtype == np.float32


def test_total_light_is_conserved():
    img = one_star(300.3, 700.7, flux=10_000.0)
    assert abs(img.sum() - 10_000.0) / 10_000.0 < 0.01


def test_peak_is_at_the_star():
    img = one_star(300.0, 700.0)                            # u = column 300, v = row 700
    row, col = np.unravel_index(np.argmax(img), img.shape)
    assert (row, col) == (700, 300)


def test_subpixel_position_is_preserved():
    """The brightness-weighted centre of the blob must sit at (u, v), to 0.01 px."""
    u0, v0 = 412.37, 268.81
    img = one_star(u0, v0).astype(np.float64)
    rows, cols = np.indices(img.shape)
    assert abs((img * cols).sum() / img.sum() - u0) < 0.01
    assert abs((img * rows).sum() / img.sum() - v0) < 0.01


def test_star_at_the_edge_does_not_crash():
    img = one_star(0.2, N_PIX - 1.1)
    assert np.isfinite(img).all() and img.max() > 0


def test_noise_statistics():
    rng = np.random.default_rng(0)
    img = render(np.array([]), np.array([]), np.array([]), rng=rng)
    assert abs(img.mean() - BACKGROUND) < 0.05
    assert abs(img.std() - READ_NOISE) < 0.05


def test_render_sky_orion():
    cat = load_catalog(5.5)
    A = attitude_from_pointing(83.8, -1.0, 0.0)
    img, u, v, idx = render_sky(A, cat, rng=np.random.default_rng(1))
    assert img.shape == (N_PIX, N_PIX)
    assert 10 <= len(idx) <= 80                             # a 20 deg field holds dozens of bright stars
    assert len(u) == len(v) == len(idx)
    assert 2061 in cat.hr[idx] and 1713 in cat.hr[idx]       # Betelgeuse and Rigel are in view


def test_add_hot_pixels():
    from tracker.render_cpu import add_hot_pixels
    img = np.full((N_PIX, N_PIX), BACKGROUND, dtype=np.float32)
    out = add_hot_pixels(img.copy(), 10, np.random.default_rng(0), lo=100.0, hi=2000.0)
    changed = out != img
    assert changed.sum() == 10                              # 10 distinct pixels
    extra = (out - img)[changed]
    assert extra.min() >= 100.0 and extra.max() <= 2000.0
    assert out.dtype == np.float32
    assert np.array_equal(add_hot_pixels(img.copy(), 0, np.random.default_rng(0)), img)
