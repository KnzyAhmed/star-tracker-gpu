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
from tracker.limb import find_limb, refine_limb
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
    # the Day 10 fix, as the pipeline runs it: two passes of the predicted-profile fit, shape from the latest position
    p2 = p_est
    for _ in range(2):
        pts2 = refine_limb(img, A, s, p2, BACKGROUND, READ_NOISE)
        p2 = fit_horizon_sphere(limb_directions(pts2, A), R_EQ)
    err2 = true_limb_error(pts2, p, A)
    total2, rng2, cross2 = position_error(p2, p)
    return dict(label=label, noisy=noisy, n=len(pts), arc=float(np.ptp(ang)), dr=dr, dr_lo=float(err.min()),
                dr_hi=float(err.max()), r_px=r_px, D=D, range_km=rng_err, range_pct=100 * rng_err / D,
                cross_km=cross, cross_pct=100 * cross / D, total_km=total, hand_km=hand, hand_pct=100 * hand / D,
                dr2=float(np.median(err2)), dr2_lo=float(err2.min()), dr2_hi=float(err2.max()),
                range2_km=rng2, range2_pct=100 * rng2 / D, cross2_km=cross2, cross2_pct=100 * cross2 / D)


def main():
    cat = load_catalog(5.5)
    rows = []
    for k, (label, D, A, s) in enumerate(CASES):
        for noisy in (False, True):
            rows.append(run(label, D, A, s, noisy, cat, 9 + k))
    hdr = (f"{'case':36s} {'noise':5s} {'pts':>4s} {'arc°':>5s} {'50% rule: limb bias px':>24s} "
           f"{'range km':>10s} {'range %':>8s} {'cross km':>9s} {'hand km':>9s} | {'PROFILE FIT: bias px':>22s} {'range km':>10s} {'range %':>8s} {'cross km':>9s}")
    lines = [hdr, "-" * len(hdr)]
    md = ["| case | noise | points | arc (deg) | 50 % rule: median limb bias (px) | range (km) | range (%) | cross-range (km) | hand −Δr/r (km) | "
          "profile fit: median limb bias (px) | range (km) | range (%) | cross-range (km) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "dr" not in r:
            lines.append(f"{r['label']:36s} {'on' if r['noisy'] else 'off':5s} {r['n']:>4d}  no solution")
            md.append(f"| {r['label']} | {'on' if r['noisy'] else 'off'} | {r['n']} | | | no solution | | | | | |")
            continue
        lines.append(f"{r['label']:36s} {'on' if r['noisy'] else 'off':5s} {r['n']:>4d} {r['arc']:>5.0f} "
                     f"{r['dr']:>+6.2f} ({r['dr_lo']:+.2f}..{r['dr_hi']:+.2f}) {r['range_km']:>+10.1f} {r['range_pct']:>+8.3f} "
                     f"{r['cross_km']:>9.1f} {r['hand_km']:>+9.1f} | {r['dr2']:>+6.2f} ({r['dr2_lo']:+.2f}..{r['dr2_hi']:+.2f}) "
                     f"{r['range2_km']:>+10.1f} {r['range2_pct']:>+8.3f} {r['cross2_km']:>9.1f}")
        md.append(f"| {r['label']} | {'on' if r['noisy'] else 'off'} | {r['n']} | {r['arc']:.0f} | {r['dr']:+.2f} ({r['dr_lo']:+.2f} to {r['dr_hi']:+.2f}) | "
                  f"{r['range_km']:+.1f} | {r['range_pct']:+.3f} | {r['cross_km']:.1f} | {r['hand_km']:+.1f} | "
                  f"{r['dr2']:+.2f} ({r['dr2_lo']:+.2f} to {r['dr2_hi']:+.2f}) | {r['range2_km']:+.1f} | {r['range2_pct']:+.3f} | {r['cross2_km']:.1f} |")
    print("\n".join(lines))
    out = Path("docs/results/day10")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bias_table.md").write_text("# Day 10: limb bias -> position error (true attitude, sphere model), 50 % rule vs predicted-profile fit\n\n"
                                       + "\n".join(md) + "\n")
    print(f"saved {out}/bias_table.md")


if __name__ == "__main__":
    main()
