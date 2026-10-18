"""Day 9 gate: look at the limb points. Three noisy frames with stars (GEO phase 90, 100,000 km off-axis and
rolled with an oblique Sun, lunar distance phase 135), the limb points found, the true limb curve, and the
radial error of every point.

    python -m experiments.show_limb        # -> docs/results/day9/limb_grid.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tests.test_limb import true_limb_error                      # the same reference the tests use
from tracker.camera import attitude_from_pointing
from tracker.catalog import load_catalog
from tracker.earth_render import render_scene
from tracker.limb import find_limb
from tracker.render_cpu import BACKGROUND, READ_NOISE
from tracker.world import earth_in_view

CASES = [
    ("GEO · phase 90°", 42_164.0, attitude_from_pointing(180.0, 0.0, 0.0), np.array([0.0, 1.0, 0.0])),
    ("100,000 km · 5° off-axis, roll 30°, oblique Sun", 100_000.0, attitude_from_pointing(180.0, 5.0, 30.0),
     np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])),
    ("lunar distance · phase 135°", 384_400.0, attitude_from_pointing(180.0, 0.0, 0.0),
     np.array([np.cos(np.radians(135)), np.sin(np.radians(135)), 0.0])),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results/day9")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = load_catalog(5.5)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=90, facecolor="#04070f",
                             gridspec_kw={"height_ratios": [3, 1.2]})
    for k, (label, D, A, s) in enumerate(CASES):
        p = np.array([D, 0.0, 0.0])
        img, mask, u, v, idx = render_scene(p, A, s, cat, rng=np.random.default_rng(9 + k))
        pts = find_limb(img, A, s, BACKGROUND, READ_NOISE)
        err = true_limb_error(pts, p, A) if len(pts) else np.array([])
        _, (uc, vc), rho, pr = earth_in_view(p, A)
        zoom = min(pr * 1.35 + 20, 512)
        ax = axes[0, k]
        shown = np.arcsinh(np.clip(img - BACKGROUND, 0, None) / (3 * READ_NOISE))
        ax.imshow(shown, cmap="gray", origin="upper", interpolation="nearest", vmin=0, vmax=np.arcsinh(200))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=6, color="#4aa3ff", label=f"{len(pts)} limb points")
        ax.scatter([uc], [vc], marker="+", s=80, color="#5fe3c0", label="true centre")
        ax.set_xlim(uc - zoom, uc + zoom); ax.set_ylim(vc + zoom, vc - zoom)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, color="#e8f0ff", fontsize=10)
        ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor="#e8f0ff")
        ax2 = axes[1, k]
        ax2.set_facecolor("#0b1226")
        if len(pts):
            ang = np.degrees(np.arctan2(pts[:, 1] - vc, pts[:, 0] - uc))
            ax2.scatter(ang, err, s=8, color="#4aa3ff")
            ax2.axhline(np.median(err), color="#ffb067", lw=1, label=f"median {np.median(err):+.2f} px")
            ax2.axhline(0, color="#7489a8", lw=0.8)
            ax2.set_ylim(-1.5, 1.5)
            ax2.legend(fontsize=8, frameon=False, labelcolor="#e8f0ff", loc="upper right")
        ax2.set_xlabel("angle around the disc (deg)", color="#b3c3dc", fontsize=9)
        ax2.set_ylabel("radial error (px)", color="#b3c3dc", fontsize=9)
        ax2.tick_params(colors="#b3c3dc", labelsize=8)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#1e3356")
        print(f"{label:50s} points {len(pts):4d}  median {np.median(err) if len(err) else float('nan'):+.3f} px  "
              f"p95 |err| {np.percentile(np.abs(err), 95) if len(err) else float('nan'):.3f} px")
    fig.suptitle("Limb points (blue) on noisy frames with stars, and their radial error from the true limb",
                 color="#e8f0ff", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "limb_grid.png", facecolor=fig.get_facecolor())
    print(f"saved {out}/limb_grid.png")


if __name__ == "__main__":
    main()
