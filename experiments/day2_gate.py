"""Day 2 gate over many random skies: centroid accuracy, and attitude with TRUE identities.

    python -m experiments.day2_gate                 # 1000 random attitudes
    python -m experiments.day2_gate --trials 200

For every frame: render (with noise) -> detect -> match each detection to the nearest true
star within 1.5 px -> solve with those true identities -> attitude error.
Also measures the error of every star-pair ANGLE, which is what Day 3 identification
compares against the catalog, so eps for Day 3 comes from here.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from tracker.camera import F_PIX, random_attitude, unproject
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.render_cpu import BACKGROUND, READ_NOISE, render_sky
from tracker.solve import RAD_TO_ARCSEC, attitude_error, solve_attitude

MATCH_PX = 1.5
PX_TO_ARCSEC = RAD_TO_ARCSEC / F_PIX


def pair_angle_errors(b, r):
    """Measured minus true angle (arcsec) for every pair of matched stars."""
    i, j = np.triu_indices(len(b), k=1)
    ang_b = np.arctan2(np.linalg.norm(np.cross(b[i], b[j]), axis=1), (b[i] * b[j]).sum(1))
    ang_r = np.arctan2(np.linalg.norm(np.cross(r[i], r[j]), axis=1), (r[i] * r[j]).sum(1))
    return (ang_b - ang_r) * RAD_TO_ARCSEC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default="results/day2_gate")
    a = p.parse_args()

    cat = load_catalog(5.5)
    rng = np.random.default_rng(a.seed)
    cen_err, cen_mag, pair_err, rows = [], [], [], []
    n_true_total = n_det_total = n_matched_total = 0
    t0 = time.perf_counter()
    for _ in range(a.trials):
        A = random_attitude(rng)
        img, u_true, v_true, idx = render_sky(A, cat, rng=rng)
        u, v, f = detect(img, BACKGROUND, READ_NOISE)
        n_true_total += len(idx)
        n_det_total += len(u)
        if len(idx) == 0 or len(u) == 0:
            rows.append((len(idx), len(u), 0, np.nan, np.nan, np.nan))
            continue
        dist, j = cKDTree(np.column_stack([u_true, v_true])).query(np.column_stack([u, v]))
        m = dist < MATCH_PX
        n_matched_total += m.sum()
        cen_err.append(dist[m])
        cen_mag.append(cat.mag[idx[j[m]]])
        if m.sum() < 2:
            rows.append((len(idx), len(u), m.sum(), np.nan, np.nan, np.nan))
            continue
        b, r = unproject(u[m], v[m]), cat.r[idx[j[m]]]
        pair_err.append(pair_angle_errors(b, r))
        rows.append((len(idx), len(u), m.sum(), *attitude_error(solve_attitude(b, r), A)))
    elapsed = time.perf_counter() - t0

    cen_err, cen_mag = np.concatenate(cen_err), np.concatenate(cen_mag)
    pair_err = np.abs(np.concatenate(pair_err))
    rows = np.array(rows, dtype=float)
    solved = np.isfinite(rows[:, 3])
    tot, poi, rol = rows[solved, 3], rows[solved, 4], rows[solved, 5]

    pct = lambda x, q: np.percentile(x, q)
    lines = [
        f"Day 2 gate: {a.trials} random attitudes, seed {a.seed}, {elapsed:.1f} s "
        f"({1000 * elapsed / a.trials:.1f} ms/frame incl. render)",
        f"stars in view (mag < 5.5): total {n_true_total}, mean {n_true_total / a.trials:.1f}/frame, "
        f"min {rows[:, 0].min():.0f}, max {rows[:, 0].max():.0f}",
        f"detected {n_det_total}, matched within {MATCH_PX} px: {n_matched_total} "
        f"({100 * n_matched_total / n_true_total:.1f} % of true, "
        f"{n_det_total - n_matched_total} unmatched detections)",
        "",
        "CENTROID ERROR (radial, px)   median {:.3f}   mean {:.3f}   p90 {:.3f}   p99 {:.3f}   p99.9 {:.3f}".format(
            np.median(cen_err), cen_err.mean(), pct(cen_err, 90), pct(cen_err, 99), pct(cen_err, 99.9)),
        "                    (arcsec)  median {:.1f}   p99 {:.1f}".format(
            np.median(cen_err) * PX_TO_ARCSEC, pct(cen_err, 99) * PX_TO_ARCSEC),
        "  by magnitude:  " + "   ".join(
            f"[{lo},{lo + 1}) {np.median(cen_err[(cen_mag >= lo) & (cen_mag < lo + 1)]):.3f}"
            for lo in range(-1, 6) if ((cen_mag >= lo) & (cen_mag < lo + 1)).any()),
        "PAIR-ANGLE ERROR |measured - catalog| (arcsec)   median {:.1f}   p99 {:.1f}   p99.9 {:.1f}   max {:.1f}".format(
            np.median(pair_err), pct(pair_err, 99), pct(pair_err, 99.9), pair_err.max()),
        "",
        f"ATTITUDE WITH TRUE IDENTITIES: solved {solved.sum()} / {a.trials} (rest have < 2 matched stars)",
        "  arcsec       median    p90    p99",
        f"  total     {np.median(tot):8.2f} {pct(tot, 90):6.2f} {pct(tot, 99):6.2f}",
        f"  pointing  {np.median(poi):8.2f} {pct(poi, 90):6.2f} {pct(poi, 99):6.2f}",
        f"  roll      {np.median(rol):8.2f} {pct(rol, 90):6.2f} {pct(rol, 99):6.2f}",
        "",
        f"GATE  median centroid error < 0.15 px: {'PASS' if np.median(cen_err) < 0.15 else 'FAIL'} "
        f"({np.median(cen_err):.3f})",
        f"GATE  true-identity pointing a few arcsec: {'PASS' if np.median(poi) < 5 else 'FAIL'} "
        f"(median {np.median(poi):.2f})",
    ]
    report = "\n".join(lines)
    print(report)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(report + "\n")
    np.savetxt(out.with_suffix(".csv"), rows, delimiter=",", fmt="%.6g",
               header="n_true,n_detected,n_matched,total_arcsec,pointing_arcsec,roll_arcsec")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=110)
    ax[0].hist(cen_err, bins=100, range=(0, 0.6))
    ax[0].axvline(np.median(cen_err), color="k", ls="--", label=f"median {np.median(cen_err):.3f} px")
    ax[0].set(xlabel="radial centroid error (px)", ylabel="stars", title="Centroid error")
    ax[0].legend()
    ax[1].hist(pair_err, bins=100, range=(0, pct(pair_err, 99.9)))
    ax[1].axvline(pct(pair_err, 99), color="r", ls="--", label=f"p99 {pct(pair_err, 99):.1f}\"")
    ax[1].set(xlabel="|pair angle error| (arcsec)", title="Pair-angle error (sets Day 3 eps)")
    ax[1].legend()
    hi = pct(rol, 99)
    ax[2].hist(poi, bins=80, range=(0, hi), alpha=0.7, label=f"pointing, median {np.median(poi):.2f}\"")
    ax[2].hist(rol, bins=80, range=(0, hi), alpha=0.7, label=f"roll, median {np.median(rol):.2f}\"")
    ax[2].set(xlabel="attitude error (arcsec)", title="Attitude error, true identities")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"))
    print(f"saved {out}.txt/.csv/.png")


if __name__ == "__main__":
    main()
