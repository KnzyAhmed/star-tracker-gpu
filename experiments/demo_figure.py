"""README hero image: one lost-in-space solve, end to end, on a rendered frame.

    python -m experiments.demo_figure            # Orion -> docs/results/demo.png
    python -m experiments.demo_figure --ra 101.3 --dec -16.7 --roll 30

Left: the simulated camera image. Right: the same image with every detected dot circled,
verified dots in blue, unverified in orange, identified named stars labelled, and the error.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.camera import attitude_from_pointing, unproject
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.identify import build_pair_table, identify
from tracker.render_cpu import BACKGROUND, READ_NOISE, render_sky
from tracker.solve import attitude_error

SURFACE, INK, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#3987e5", "#eb6834"
# proper names for a few famous stars, by HR number (the catalog stores Bayer/Flamsteed codes)
PROPER = {2061: "Betelgeuse", 1713: "Rigel", 1790: "Bellatrix", 1903: "Alnilam", 1948: "Alnitak",
          1852: "Mintaka", 2004: "Saiph", 1899: "Hatysa", 1666: "Cursa", 2491: "Sirius", 2943: "Procyon"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ra", type=float, default=83.8)
    p.add_argument("--dec", type=float, default=-1.0)
    p.add_argument("--roll", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="docs/results/demo.png")
    a = p.parse_args()

    cat = load_catalog(5.5)
    table = build_pair_table(cat.r)
    A = attitude_from_pointing(a.ra, a.dec, a.roll)
    img, _, _, idx = render_sky(A, cat, rng=np.random.default_rng(a.seed))
    u, v, f = detect(img, BACKGROUND, READ_NOISE)
    sol = identify(unproject(u, v), table)
    shown = np.arcsinh(np.clip(img - BACKGROUND, 0, None) / (3 * READ_NOISE))

    fig, ax = plt.subplots(1, 2, figsize=(13, 7.0), dpi=120, facecolor=SURFACE)
    for axis in ax:
        axis.imshow(shown, cmap="gray", origin="upper", interpolation="nearest")
        axis.set_xticks([]), axis.set_yticks([])
    ax[0].set_title(f"1. What the camera sees\n{len(idx)} catalog stars in a 20° field, read noise {READ_NOISE:g}", color=INK)
    verified = np.zeros(len(u), bool)
    if sol is not None:
        verified[sol.det] = True
    ax[1].scatter(u[~verified], v[~verified], s=160, facecolors="none", edgecolors=ORANGE, linewidths=1.2)
    ax[1].scatter(u[verified], v[verified], s=160, facecolors="none", edgecolors=BLUE, linewidths=1.4)
    if sol is not None:
        for d, c in zip(sol.det, sol.cat):
            name = PROPER.get(int(cat.hr[c]), cat.name[c].strip())
            if name and cat.mag[c] < 3.0:
                ax[1].annotate(name, (u[d], v[d]), xytext=(9, -9), textcoords="offset points", color="#ffcc66", fontsize=8)
        total, pointing, roll = attitude_error(sol.A, A)
        result = (f"2. Lost-in-space solve: {len(u)} dots, {len(sol.det)} verified\n"
                  f"pointing error {pointing:.1f}\", roll error {roll:.1f}\"")
    else:
        result = f"2. Lost-in-space solve: {len(u)} dots -> no solution"
    ax[1].set_title(result, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.025, f"Truth: RA {a.ra:g}°, Dec {a.dec:g}°, roll {a.roll:g}°.  Blue circle = detected and verified "
             "against the catalog; orange = detected, not verified.  The tracker is told nothing about the pointing.",
             ha="center", color="#52514e", fontsize=9)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {a.out}: {result}")


if __name__ == "__main__":
    main()
