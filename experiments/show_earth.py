"""Day 8 gate: look at Earth. A 3 x 3 grid of frames: distance (GEO, 100,000 km, lunar) x Sun phase angle
(0, 90, 135 deg), each with the real star field behind it, stars occluded, noise on.

    python -m experiments.show_earth                 # -> docs/results/day8/earth_grid.png + one full-res frame

The camera looks straight at Earth's centre from p = D * (1, 0, 0); the Sun is placed in the x-y plane at the
given phase angle from the camera direction. Printed per frame: apparent radius, pixel radius, lit fraction,
stars in view and hidden, and the render time.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.camera import attitude_from_pointing
from tracker.catalog import load_catalog
from tracker.earth_render import render_scene
from tracker.render_cpu import BACKGROUND, READ_NOISE
from tracker.world import R_EQ, R_POL, earth_in_view

DISTANCES = [("GEO", 42_164.0), ("100,000 km", 100_000.0), ("lunar distance", 384_400.0)]
PHASES = [0, 90, 135]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/results/day8")
    p.add_argument("--seed", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = load_catalog(5.5)
    A = attitude_from_pointing(180.0, 0.0, 0.0)          # boresight toward -x: at Earth's centre
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5), dpi=90, facecolor="#04070f")
    rows = []
    for i, (label, D) in enumerate(DISTANCES):
        pos = np.array([D, 0.0, 0.0])
        for j, phase in enumerate(PHASES):
            ph = np.radians(phase)
            sun = np.array([np.cos(ph), np.sin(ph), 0.0])  # angle `phase` between Earth->camera and Earth->Sun
            t0 = time.perf_counter()
            img, mask, u, v, idx = render_scene(pos, A, sun, cat, rng=np.random.default_rng(a.seed + 10 * i + j),
                                                a=R_EQ, c=R_POL)
            dt = time.perf_counter() - t0
            _, _, rho, pr = earth_in_view(pos, A)
            lit = (img[mask] > BACKGROUND + 10 * READ_NOISE).mean()
            rows.append((label, phase, np.degrees(rho), pr, lit, len(idx), dt))
            shown = np.arcsinh(np.clip(img - BACKGROUND, 0, None) / (3 * READ_NOISE))
            ax = axes[i, j]
            ax.imshow(shown, cmap="gray", origin="upper", interpolation="nearest", vmin=0, vmax=np.arcsinh(200))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label}  ·  phase {phase}°  ·  radius {pr:.0f} px", color="#e8f0ff", fontsize=10)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e3356")
            if (label, phase) == ("GEO", 90):
                plt.imsave(out / "earth_geo_phase90.png", shown, cmap="gray", vmin=0, vmax=np.arcsinh(200))
    fig.suptitle("Earth + stars as the camera sees them (asinh stretch, stars occluded, noise on)", color="#e8f0ff", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "earth_grid.png", facecolor=fig.get_facecolor())
    print(f"{'distance':>15s} {'phase':>6s} {'rho deg':>8s} {'px':>7s} {'lit frac':>9s} {'stars':>6s} {'render s':>9s}")
    for label, phase, rho, pr, lit, n, dt in rows:
        print(f"{label:>15s} {phase:>6d} {rho:>8.3f} {pr:>7.1f} {lit:>9.2f} {n:>6d} {dt:>9.2f}")
    print(f"saved {out}/earth_grid.png, earth_geo_phase90.png")


if __name__ == "__main__":
    main()
