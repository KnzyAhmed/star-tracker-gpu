"""Render one pointing and save it as a PNG, with the brightest stars labelled.

    python -m experiments.show_sky                      # Orion, the Day 1 gate
    python -m experiments.show_sky --ra 101.3 --dec -16.7 --roll 30 --out results/sirius.png

Compare with Stellarium: field of view 20 deg, centred on the same RA/Dec,
equatorial view (north up). The star pattern must match, including left/right.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.camera import attitude_from_pointing
from tracker.catalog import load_catalog
from tracker.render_cpu import BACKGROUND, READ_NOISE, render_sky


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ra", type=float, default=83.8)
    p.add_argument("--dec", type=float, default=-1.0)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-mag", type=float, default=2.5, help="label stars brighter than this")
    p.add_argument("--out", default="results/sky.png")
    a = p.parse_args()

    cat = load_catalog(5.5)
    A = attitude_from_pointing(a.ra, a.dec, a.roll)
    img, u, v, idx = render_sky(A, cat, rng=np.random.default_rng(a.seed))

    # asinh stretch: shows faint stars without blowing out bright ones
    shown = np.arcsinh(np.clip(img - BACKGROUND, 0, None) / (3 * READ_NOISE))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(shown, cmap="gray", origin="upper", interpolation="nearest")
    for ui, vi, k in zip(u, v, idx):
        if cat.mag[k] < a.label_mag:
            label = cat.name[k] or f"HR {cat.hr[k]}"
            ax.annotate(f"{label} ({cat.mag[k]:.1f})", (ui, vi), xytext=(8, -8),
                        textcoords="offset points", color="#ffcc66", fontsize=8)
    ax.set_title(f"RA {a.ra:.2f}  Dec {a.dec:.2f}  roll {a.roll:.1f}   |   "
                 f"{len(idx)} stars brighter than 5.5 in view", fontsize=9)
    ax.set_xlabel("u (column) ->   east is left, west is right")
    ax.set_ylabel("v (row) ->   north is up")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"saved {a.out}: {len(idx)} stars in view")
    for k in idx[np.argsort(cat.mag[idx])][:8]:
        print(f"  HR {cat.hr[k]:>5}  mag {cat.mag[k]:5.2f}  {cat.name[k]}")


if __name__ == "__main__":
    main()
