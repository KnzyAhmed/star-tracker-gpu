"""Day 10: how the Day 9 limb biases turn into position error, with the REAL fit and the TRUE attitude.

    python -m experiments.day10_bias          # prints the table, saves docs/results/day10/bias_table.md

Cases are the Day 9 frames (sphere Earth, stars, noise) plus their noise-free twins. For each: run find_limb and
fit_horizon_sphere, split the error into range and cross-range, and compare the range error with the hand
estimate dD/D = -dr/r (dr = median radial limb bias, r = pixel radius). The hand estimate assumes the whole
circle is biased uniformly; a short, unevenly biased arc behaves differently, which is the point of the table.
"""
from pathlib import Path

import numpy as np

from tests.test_limb import true_limb_error
from tracker.camera import attitude_from_pointing
from tracker.catalog import load_catalog
from tracker.earth_render import render_earth, render_scene
from tracker.limb import find_limb
from tracker.opnav import fit_horizon_sphere, limb_directions, position_error
from tracker.render_cpu import BACKGROUND, READ_NOISE
from tracker.world import R_EQ, earth_in_view

CASES = [
    ("GEO, phase 0 (fading limb)", 42_164.0, attitude_from_pointing(180.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0])),
    ("GEO, phase 90", 42_164.0, attitude_from_pointing(180.0, 0.0, 0.0), np.array([0.0, 1.0, 0.0])),
    ("100,000 km, off-axis, oblique Sun", 100_000.0, attitude_from_pointing(180.0, 5.0, 30.0),
     np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])),
    ("lunar, phase 135 (thin crescent)", 384_400.0, attitude_from_pointing(180.0, 0.0, 0.0),
     np.array([np.cos(np.radians(135)), np.sin(np.radians(135)), 0.0])),
    ("lunar, phase 90", 384_400.0, attitude_from_pointing(180.0, 0.0, 0.0), np.array([0.0, 1.0, 0.0])),
]


def run(label, D, A, s, noisy, cat, seed):
    p = np.array([D, 0.0, 0.0])
    if noisy:
        img, *_ = render_scene(p, A, s, cat, rng=np.random.default_rng(seed))
    else:
        img = render_earth(p, A, s)[0] + BACKGROUND
    pts = find_limb(img, A, s, BACKGROUND, READ_NOISE)
    _, (uc, vc), rho, r_px = earth_in_view(p, A)
    if len(pts) < 20:
        return dict(label=label, noisy=noisy, n=len(pts))
    err = true_limb_error(pts, p, A)
    ang = np.degrees(np.arctan2(pts[:, 1] - vc, pts[:, 0] - uc))
    p_est = fit_horizon_sphere(limb_directions(pts, A), R_EQ)
    total, rng_err, cross = position_error(p_est, p)
    dr = float(np.median(err))
    hand = -dr / r_px * D                                   # dD = -D * dr / r
    return dict(label=label, noisy=noisy, n=len(pts), arc=float(np.ptp(ang)), dr=dr, dr_lo=float(err.min()),
                dr_hi=float(err.max()), r_px=r_px, D=D, range_km=rng_err, range_pct=100 * rng_err / D,
                cross_km=cross, cross_pct=100 * cross / D, total_km=total, hand_km=hand, hand_pct=100 * hand / D)


def main():
    cat = load_catalog(5.5)
    rows = []
    for k, (label, D, A, s) in enumerate(CASES):
        for noisy in (False, True):
            rows.append(run(label, D, A, s, noisy, cat, 9 + k))
    hdr = (f"{'case':36s} {'noise':5s} {'pts':>4s} {'arc°':>5s} {'limb bias px (min..max)':>26s} "
           f"{'range km':>10s} {'range %':>8s} {'cross km':>9s} {'cross %':>8s} {'hand km':>9s} {'hand %':>7s}")
    lines = [hdr, "-" * len(hdr)]
    md = ["| case | noise | points | arc (deg) | median limb bias (px) | range error (km) | range (%) | cross-range (km) | cross (%) | hand estimate (km) | hand (%) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "dr" not in r:
            lines.append(f"{r['label']:36s} {'on' if r['noisy'] else 'off':5s} {r['n']:>4d}  no solution")
            md.append(f"| {r['label']} | {'on' if r['noisy'] else 'off'} | {r['n']} | | | no solution | | | | | |")
            continue
        lines.append(f"{r['label']:36s} {'on' if r['noisy'] else 'off':5s} {r['n']:>4d} {r['arc']:>5.0f} "
                     f"{r['dr']:>+8.2f} ({r['dr_lo']:+.2f}..{r['dr_hi']:+.2f})   {r['range_km']:>+10.1f} {r['range_pct']:>+8.3f} "
                     f"{r['cross_km']:>9.1f} {r['cross_pct']:>8.3f} {r['hand_km']:>+9.1f} {r['hand_pct']:>+7.3f}")
        md.append(f"| {r['label']} | {'on' if r['noisy'] else 'off'} | {r['n']} | {r['arc']:.0f} | {r['dr']:+.2f} ({r['dr_lo']:+.2f} to {r['dr_hi']:+.2f}) | "
                  f"{r['range_km']:+.1f} | {r['range_pct']:+.3f} | {r['cross_km']:.1f} | {r['cross_pct']:.3f} | {r['hand_km']:+.1f} | {r['hand_pct']:+.3f} |")
    print("\n".join(lines))
    out = Path("docs/results/day10")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bias_table.md").write_text("# Day 10: limb bias -> position error (true attitude, sphere model)\n\n" + "\n".join(md) + "\n")
    print(f"saved {out}/bias_table.md")


if __name__ == "__main__":
    main()
