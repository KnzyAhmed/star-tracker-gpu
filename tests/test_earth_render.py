"""Day 8: the CPU Earth renderer. Run:  pytest tests/test_earth_render.py -q"""
import numpy as np
import pytest

from tracker.camera import C_PIX, F_PIX, N_PIX, attitude_from_pointing, project
from tracker.catalog import load_catalog
from tracker.earth_render import occluded, render_earth, render_scene
from tracker.render_cpu import BACKGROUND, READ_NOISE
from tracker.world import EARTH_LEVEL, R_EQ, R_POL

GEO, LUNAR = 42_164.0, 384_400.0
TOWARD_EARTH = attitude_from_pointing(180.0, 0.0, 0.0)      # boresight along -x; p is on +x, so Earth is on-axis


def half_width_50(profile, x):
    """Half-width of a bump-shaped profile at 50 % of its peak, with linear interpolation on both sides."""
    half = 0.5 * profile.max()
    above = np.flatnonzero(profile > half)
    i, j = above[0], above[-1]
    left = x[i - 1] + (half - profile[i - 1]) / (profile[i] - profile[i - 1]) * (x[i] - x[i - 1])
    right = x[j] + (profile[j] - half) / (profile[j] - profile[j + 1]) * (x[j + 1] - x[j])
    return (right - left) / 2


@pytest.mark.parametrize("dist", [GEO, LUNAR])
@pytest.mark.parametrize("supersample, tol", [(4, 0.15), (8, 0.08)])
def test_unshaded_disc_edge_radius(dist, supersample, tol):
    """Geometry check: a flat disc blurred by the PSF crosses 50 % of its level at the geometric limb.
    n x n sub-rays quantise the edge to 1/n px, so the crossing can be off by up to 1/(2n) px: 0.125 px at 4x4."""
    p = np.array([dist, 0.0, 0.0])
    img, mask = render_earth(p, TOWARD_EARTH, s=p / dist, shade=False, supersample=supersample)
    expected = F_PIX * np.tan(np.arcsin(R_EQ / dist))
    row = (img[511] + img[512]) / 2                       # the centre sits between rows 511 and 512
    assert abs(half_width_50(row, np.arange(N_PIX)) - expected) < tol
    col = (img[:, 511] + img[:, 512]) / 2
    assert abs(half_width_50(col, np.arange(N_PIX)) - expected) < tol
    assert abs(img[480:544, 480:544].mean() - EARTH_LEVEL) < 1e-3 * EARTH_LEVEL   # flat top at the level


def test_mask_is_the_geometric_disc():
    p = np.array([GEO, 0.0, 0.0])
    img, mask = render_earth(p, TOWARD_EARTH, s=p / GEO)
    r_px = F_PIX * np.tan(np.arcsin(R_EQ / GEO))
    assert mask.dtype == bool and img.dtype == np.float32
    assert abs(mask.sum() - np.pi * r_px ** 2) / (np.pi * r_px ** 2) < 0.005
    vv, uu = np.indices(mask.shape)
    r = np.hypot(uu - C_PIX, vv - C_PIX)
    assert mask[r < r_px - 1].all() and not mask[r > r_px + 1].any()


def lambert_full_phase(D, R, r_px):
    """Brightness (0..1) at image radius r_px for a Lambertian sphere, Sun behind a camera at distance D.
    A ray at angle beta from the centre (tan beta = r / F) hits at t = D cos beta - sqrt(R^2 - D^2 sin^2 beta),
    and the normal's cosine to the Sun is (D - t cos beta) / R. At D -> infinity this becomes sqrt(1 - (r/R_px)^2)."""
    beta = np.arctan(r_px / F_PIX)
    t = D * np.cos(beta) - np.sqrt(R**2 - (D * np.sin(beta))**2)
    return (D - t * np.cos(beta)) / R


def test_full_phase_lambertian_shading():
    """Sun behind the camera: EARTH_LEVEL at the centre, falling toward the limb. At GEO (D/R = 6.6) the visible
    cap spans only 81 deg of normals, so the fall-off is gentler than the orthographic sqrt(1 - (r/R)^2):
    0.899 instead of 0.866 at half the radius."""
    p = np.array([GEO, 0.0, 0.0])
    img, mask = render_earth(p, TOWARD_EARTH, s=p / GEO)
    r_px = F_PIX * np.tan(np.arcsin(R_EQ / GEO))
    assert abs(img[511, 511] - EARTH_LEVEL) < 0.01 * EARTH_LEVEL
    assert abs(lambert_full_phase(GEO, R_EQ, 0.5 * r_px) - 0.8992) < 5e-4            # the hand value
    for frac in (0.5, 0.8):
        u = int(round(C_PIX + frac * r_px))
        expect = lambert_full_phase(GEO, R_EQ, u - C_PIX)
        assert abs((img[511, u] + img[512, u]) / 2 / EARTH_LEVEL - expect) < 0.01


def test_night_side_is_dark_but_still_masked():
    """Phase 180: the Sun is behind Earth. No light at all, but the disc still hides the sky."""
    p = np.array([GEO, 0.0, 0.0])
    img, mask = render_earth(p, TOWARD_EARTH, s=-p / GEO)
    assert img.max() == 0.0
    assert mask.sum() > 500_000


def test_half_phase_has_a_dark_half():
    """Phase 90, Sun along +y (camera image: east is left at roll 0, +y sky is RA 90 -> appears on one side)."""
    p = np.array([GEO, 0.0, 0.0])
    img, mask = render_earth(p, TOWARD_EARTH, s=np.array([0.0, 1.0, 0.0]))
    r_px = F_PIX * np.tan(np.arcsin(R_EQ / GEO))
    left, right = img[:, : int(C_PIX - 10)], img[:, int(C_PIX + 10):]
    lit, dark = (left, right) if left.sum() > right.sum() else (right, left)
    assert dark.max() < 0.02 * EARTH_LEVEL                 # only PSF leakage near the terminator
    assert lit.max() > 0.9 * EARTH_LEVEL


def test_oblate_earth_is_shorter_than_wide():
    """From the equatorial plane the polar (vertical) half-width is R_POL/R_EQ of the equatorial one: ~1.5 px at GEO."""
    p = np.array([GEO, 0.0, 0.0])
    img, mask = render_earth(p, TOWARD_EARTH, s=p / GEO, a=R_EQ, c=R_POL, shade=False, supersample=8)
    x = np.arange(N_PIX)
    wide = half_width_50((img[511] + img[512]) / 2, x)
    tall = half_width_50((img[:, 511] + img[:, 512]) / 2, x)
    assert abs(wide - F_PIX * np.tan(np.arcsin(R_EQ / GEO))) < 0.08
    assert abs(wide - tall - F_PIX * np.tan(np.arcsin(R_EQ / GEO)) * (1 - R_POL / R_EQ)) < 0.1
    sphere, _ = render_earth(p, TOWARD_EARTH, s=p / GEO, a=R_EQ, c=R_EQ, shade=False)
    assert np.array_equal(sphere, render_earth(p, TOWARD_EARTH, s=p / GEO, shade=False)[0])


def surface_integral_flux(p, s, R, n=400_000, seed=0):
    """Independent reference: total image flux of a Lambertian sphere by integrating over its SURFACE.
    Each visible patch contributes radiance (n.s)+ times its projected solid angle (n.v) dA / dist^2."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, n); phi = rng.uniform(0, 2 * np.pi, n)
    nrm = np.column_stack([np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z])
    x = R * nrm
    to_cam = p - x
    dist = np.linalg.norm(to_cam, axis=1)
    cos_v = (nrm * to_cam).sum(1) / dist
    lit = np.maximum(0.0, nrm @ s)
    vis = cos_v > 0
    return float((lit[vis] * cos_v[vis] / dist[vis]**2).sum() * 4 * np.pi * R**2 / n)


def test_flux_scales_with_lit_area():
    """Phase 0 vs 90 vs 135 deg. Against an independent surface integral at GEO (finite camera distance),
    and against the orthographic Lambert phase law (sin a + (pi - a) cos a) / pi from lunar distance, where
    D/R = 60 makes the camera effectively at infinity. At GEO the half-phase ratio is 0.26, not 1/pi = 0.32: the
    8.7 deg rim the finite camera cannot see holds the brightest part of the half-lit disc."""
    p = np.array([GEO, 0.0, 0.0])
    suns = {0: p / GEO, 90: np.array([0.0, 1.0, 0.0]), 135: np.array([np.cos(np.radians(135)), np.sin(np.radians(135)), 0.0])}
    rendered = {a: render_earth(p, TOWARD_EARTH, s=s)[0].sum() for a, s in suns.items()}
    reference = {a: surface_integral_flux(p, s, R_EQ) for a, s in suns.items()}
    for a in (90, 135):
        assert abs(rendered[a] / rendered[0] - reference[a] / reference[0]) < 0.01
    assert abs(rendered[90] / rendered[0] - 0.26) < 0.01
    p = np.array([LUNAR, 0.0, 0.0])
    far = {a: render_earth(p, TOWARD_EARTH, s=s)[0].sum() for a, s in {0: p / LUNAR, 90: suns[90], 135: suns[135]}.items()}
    for a in (90, 135):
        ar = np.radians(a)
        assert abs(far[a] / far[0] - (np.sin(ar) + (np.pi - ar) * np.cos(ar)) / np.pi) < 0.012


def test_not_in_view_gives_empty_image_fast():
    p = np.array([GEO, 0.0, 0.0])
    img, mask = render_earth(p, attitude_from_pointing(0.0, 0.0, 0.0), s=p / GEO)
    assert img.max() == 0.0 and not mask.any()


def test_occluded_stars():
    p = np.array([100_000.0, 0.0, 0.0])
    rho = np.arcsin(R_EQ / 100_000.0)
    r = np.array([[-1.0, 0.0, 0.0],                                                   # behind Earth's centre
                  [1.0, 0.0, 0.0],                                                    # opposite side of the sky
                  [-np.cos(rho * 0.98), np.sin(rho * 0.98), 0.0],                     # just inside the limb
                  [-np.cos(rho * 1.02), np.sin(rho * 1.02), 0.0]])                    # just outside
    assert occluded(p, r).tolist() == [True, False, True, False]
    assert occluded(p, r, a=R_EQ, c=R_POL).tolist()[:2] == [True, False]


def test_render_scene_hides_stars_behind_earth():
    """Put Earth (seen from GEO) right in front of Betelgeuse: Betelgeuse and its neighbours must vanish."""
    cat = load_catalog(5.5)
    k_bet = int(np.flatnonzero(cat.hr == 2061)[0])
    p = -GEO * cat.r[k_bet]                                # spacecraft on the far side of Earth from Betelgeuse
    ra = np.degrees(np.arctan2(cat.r[k_bet, 1], cat.r[k_bet, 0])) % 360
    dec = np.degrees(np.arcsin(cat.r[k_bet, 2]))
    A = attitude_from_pointing(ra, dec, 0.0)              # looking at Earth's centre, Betelgeuse behind it
    s = np.array([0.0, 0.0, 1.0])                         # Sun from the north: half-lit
    img, mask, u, v, idx = render_scene(p, A, s, cat, rng=np.random.default_rng(0))
    uu, vv, vis = project(A, cat.r)
    in_view = np.flatnonzero(vis)
    hidden = np.setdiff1d(in_view, idx)
    assert len(idx) + len(hidden) == len(in_view) and len(hidden) >= 1
    assert all(mask[int(round(vv[k])), int(round(uu[k]))] for k in hidden)     # every hidden star is under the disc
    assert not any(mask[int(round(vv[k])), int(round(uu[k]))] for k in idx)    # no drawn star is under the disc
    assert k_bet in hidden
    assert img.shape == (N_PIX, N_PIX) and abs(img[0:50, 0:50].mean() - BACKGROUND) < 1.0
    assert abs(img[0:50, 0:50].std() - READ_NOISE) < 0.6
    assert mask[511, 511] and img.max() > EARTH_LEVEL * 0.9
