"""The website's Python side. In the browser it runs inside Pyodide (Python compiled to WebAssembly)
on the REAL tracker/ modules; the page calls setup() once, then run_frame() per solve.

Everything returned is plain Python (dicts, lists, numbers, str, bytes) so it crosses into
JavaScript cleanly. The image goes over as 1024*1024 bytes (one grey level per pixel).
"""
import time

import numpy as np

from tracker.camera import attitude_from_pointing, pointing_from_attitude, project, unproject
from tracker.campaign import trial_attitude
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.evaluate import CORRECT_ARCSEC
from tracker.identify import EPS_ARCSEC, build_pair_table, identify, vote
from tracker.render_cpu import BACKGROUND, add_hot_pixels, render, star_flux
from tracker.solve import attitude_error

# Proper names for famous stars by HR number (the catalog stores Bayer/Flamsteed codes like "58Alp Ori").
PROPER = {2061: "Betelgeuse", 1713: "Rigel", 1790: "Bellatrix", 1903: "Alnilam", 1948: "Alnitak",
          1852: "Mintaka", 2004: "Saiph", 1899: "Hatysa", 1666: "Cursa", 2491: "Sirius", 2943: "Procyon",
          2326: "Canopus", 5340: "Arcturus", 7001: "Vega", 1708: "Capella", 1457: "Aldebaran",
          5056: "Spica", 6134: "Antares", 2990: "Pollux", 2891: "Castor", 3982: "Regulus", 7790: "Altair",
          7924: "Deneb", 8728: "Fomalhaut", 424: "Polaris", 472: "Achernar", 5459: "Rigil Kentaurus",
          4730: "Acrux", 4853: "Mimosa", 6527: "Shaula", 2618: "Adhara", 2693: "Wezen"}
DISPLAY_SCALE = 3 * 5.0           # asinh stretch knee in counts (3 x the default read noise), fixed so noise shows

_S = {}


def setup():
    """Load the catalog and build the pair table once (a few hundred ms)."""
    if not _S:
        cat = load_catalog(5.5)
        _S.update(cat=cat, table=build_pair_table(cat.r))
    return {"stars": len(_S["cat"]), "pairs": int(len(_S["table"].angle)), "eps_arcsec": EPS_ARCSEC}


def star_name(k):
    cat = _S["cat"]
    hr = int(cat.hr[k])
    return PROPER.get(hr, str(cat.name[k]).strip())


def _stretch(img):
    x = np.arcsinh(np.clip(img - BACKGROUND, 0, None) / DISPLAY_SCALE) / np.arcsinh(100.0)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8).tobytes()


def run_frame(ra, dec, roll, read_noise=5.0, hot_pixels=0, seed=0):
    """One lost-in-space solve. The tracker sees only the image; (ra, dec, roll) is the hidden truth."""
    setup()
    cat, table = _S["cat"], _S["table"]
    t = {}
    t0 = time.perf_counter()
    A = attitude_from_pointing(ra, dec, roll)
    u, v, vis = project(A, cat.r)
    k = np.flatnonzero(vis)
    t["prepare"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    img = render(u[k], v[k], star_flux(cat.mag[k]), rng=np.random.default_rng([seed, 0]), read_noise=read_noise)
    if hot_pixels:
        add_hot_pixels(img, int(hot_pixels), np.random.default_rng([seed, 1]))
    t["render"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    du, dv, flux = detect(img, BACKGROUND, read_noise)
    t["detect"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = unproject(du, dv)
    sol = identify(b, table)
    t["identify"] = time.perf_counter() - t0
    best, votes = vote(b, table, len(cat), EPS_ARCSEC)       # again, only to show the vote (not timed)

    verified = {}
    if sol is not None:
        verified = {int(d): int(c) for d, c in zip(sol.det, sol.cat)}
    dots = []
    for i in range(len(du)):
        c = verified.get(i, int(best[i]) if best[i] >= 0 else None)
        dots.append({"u": round(float(du[i]), 3), "v": round(float(dv[i]), 3), "flux": round(float(flux[i]), 1),
                     "verified": i in verified, "votes": int(votes[i]),
                     "hr": int(cat.hr[c]) if c is not None else None,
                     "name": star_name(c) if i in verified else "",
                     "mag": round(float(cat.mag[c]), 2) if c is not None else None})
    stars = [{"u": round(float(u[j]), 2), "v": round(float(v[j]), 2), "hr": int(cat.hr[j]),
              "name": star_name(j), "mag": round(float(cat.mag[j]), 2)} for j in k]

    out = {"image": _stretch(img), "dots": dots, "stars": stars,
           "n_in_view": int(len(k)), "n_detected": int(len(du)), "n_verified": len(verified),
           "n_pairs": int(len(du) * (len(du) - 1) // 2),
           "truth": {"ra": float(ra), "dec": float(dec), "roll": float(roll)},
           "timing_ms": {s: round(1000 * x, 2) for s, x in t.items()},
           "estimate": None, "error_arcsec": None, "outcome": "no solution"}
    if sol is not None:
        e_ra, e_dec, e_roll = pointing_from_attitude(sol.A)
        total, pointing, roll_err = attitude_error(sol.A, A)
        out["estimate"] = {"ra": e_ra, "dec": e_dec, "roll": e_roll}
        out["error_arcsec"] = {"total": round(float(total), 2), "pointing": round(float(pointing), 2), "roll": round(float(roll_err), 2)}
        out["outcome"] = "correct" if total < CORRECT_ARCSEC else "wrong"
    return out


def random_pointing(seed):
    """A uniformly random attitude (Shoemake), as (ra, dec, roll) in degrees: 'lost in space'."""
    ra, dec, roll = pointing_from_attitude(trial_attitude(20261015, int(seed)))
    return {"ra": ra, "dec": dec, "roll": roll}


def sky_stars():
    """The whole catalog for the sky globe: RA, Dec (deg), magnitude, and names of the bright ones."""
    setup()
    cat = _S["cat"]
    r = cat.r
    ra = (np.degrees(np.arctan2(r[:, 1], r[:, 0])) % 360).round(3)
    dec = np.degrees(np.arcsin(np.clip(r[:, 2], -1, 1))).round(3)
    names = [star_name(k) if cat.mag[k] < 1.6 else "" for k in range(len(cat))]
    return {"ra": ra.tolist(), "dec": dec.tolist(), "mag": cat.mag.round(2).tolist(), "name": names}
