"""Day 12 analysis: results/opnav/*.csv -> report, summary.json and figures.

    python -m experiments.analyze_opnav                       # results/opnav -> results/opnav/analysis
    python -m experiments.analyze_opnav --dir results/opnav_smoke
    python -m experiments.analyze_opnav --theme dark          # transparent figures for the website

Error bars: rates get 95 % Wilson intervals; medians get a 95 % bootstrap interval over TRIALS (resampling trials,
never limb points: the ~180 limb points of one frame share the PSF blur and are not independent, so a per-fit
covariance would be far too optimistic; the spread over Monte Carlo trials is the honest one).
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import analyze_campaign as ac
from tracker.evaluate import CORRECT, NO_SOLUTION, WRONG
from tracker.opnav_campaign import ATT_TRACKER, ATT_TRUE, COLUMNS, REASONS
from tracker.stats import outcome_rates, wilson_interval

C = {name: k for k, name in enumerate(COLUMNS)}
PHASE_COLORS = {0.0: "#6da7ec", 45.0: "#2a78d6", 90.0: "#eb6834", 135.0: "#d03b3b"}
Q = (50, 95)


def load_all(d):
    files = sorted(glob.glob(str(Path(d) / "d*_ph*_n*.csv")))
    rows = [np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2) for f in files]
    rows = [r for r in rows if len(r)]
    return np.vstack(rows) if rows else np.zeros((0, len(COLUMNS)))


def boot_median(x, n_boot=1000, seed=0):
    """(median, lo, hi): 95 % bootstrap interval of the median over trials."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def select(rows, att=ATT_TRACKER, noise=None, phase=None, solved=None):
    m = rows[:, C["att_src"]] == att
    if noise is not None:
        m &= rows[:, C["read_noise"]] == noise
    if phase is not None:
        m &= rows[:, C["phase"]] == phase
    if solved:
        m &= rows[:, C["outcome"]] != NO_SOLUTION
    return rows[m]


def per_distance(rows, column, absolute=True, stat="median"):
    """-> distances, value, lo, hi (bootstrap) and p95, over the solved trials of each distance level."""
    dist = np.unique(rows[:, C["distance"]])
    out = []
    for d in dist:
        x = rows[rows[:, C["distance"]] == d, C[column]]
        x = np.abs(x) if absolute else x
        x = x[np.isfinite(x)]
        med, lo, hi = boot_median(x)
        out.append((med, lo, hi, np.percentile(x, 95) if len(x) else np.nan, len(x)))
    return dist, np.array(out)


def band(ax, x, med, lo, hi, p95, color, label, ls="-"):
    ax.plot(x, med, color=color, lw=2, marker="o", ms=5, mec=ac.SURFACE, mew=1.2, label=label, ls=ls)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    if p95 is not None:
        ax.plot(x, p95, color=color, lw=1, ls=":", alpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/opnav")
    p.add_argument("--theme", choices=ac.THEMES, default="light")
    a = p.parse_args()
    ac.set_theme(a.theme)
    d = Path(a.dir)
    out = d / ("analysis" if a.theme == "light" else "analysis_dark")
    out.mkdir(exist_ok=True)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    rows = load_all(d)
    nominal = meta.get("read_noise", [5.0, 10.0])[0]
    noises = sorted(set(rows[:, C["read_noise"]].tolist()))
    phases = sorted(set(rows[:, C["phase"]].tolist()))
    dists = np.unique(rows[:, C["distance"]])
    L, S = [], {"meta": meta, "distances_km": dists.tolist(), "phases_deg": phases, "read_noise": noises}
    trk = select(rows, ATT_TRACKER)
    n_trials = len(trk)
    L += [f"OPTICAL NAVIGATION CAMPAIGN  ({meta.get('backend', '?')}, {meta.get('machine', '')}, seed {meta.get('seed', '?')})",
          f"{n_trials:,} trials in {len(dists)} distances x {len(phases)} phases x {len(noises)} noise levels; "
          f"{len(select(rows, ATT_TRUE)):,} extra true-attitude solves",
          f"thresholds: correct = total position error < {100 * meta.get('correct_fraction', 0.01):g} % of the distance; "
          f"no solution = tracker fails / < {meta.get('min_limb_points', 20)} limb points / fit fails", ""]

    # ---------------- outcomes per distance (tracker attitude), nominal noise and 2x, with reasons
    S["outcomes"] = {}
    fig, axes = plt.subplots(1, len(noises), figsize=(5.5 * len(noises), 4.2), dpi=130, sharey=True, squeeze=False)
    for j, noise in enumerate(noises):
        r = select(rows, ATT_TRACKER, noise=noise)
        L += [f"OUTCOMES, read noise {noise:g}, attitude from the second head's tracker"]
        ax = axes[0, j]
        for ph in phases:
            rp = r[r[:, C["phase"]] == ph]
            t = outcome_rates(rp[:, C["outcome"]], rp[:, C["distance"]])
            ac.errorbar_rates(ax, t["group"], t, PHASE_COLORS.get(ph, ac.BLUE), f"phase {ph:g}°")
            L += [f"  phase {ph:g}°"] + [ac.rate_line(f"{g:,.0f} km", t, i) for i, g in enumerate(t["group"])]
            S["outcomes"][f"noise{noise:g}_phase{ph:g}"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in t.items()}
        reasons = {REASONS[i]: int((r[:, C["reason"]] == i).sum()) for i in range(len(REASONS))}
        wrong = int((r[:, C["outcome"]] == WRONG).sum())
        L += [f"  reasons for no solution: " + ", ".join(f"{k}: {v:,}" for k, v in reasons.items() if k != "ok")
              + f";  wrong (>= 1 %): {wrong:,} of {len(r):,}", ""]
        S["outcomes"][f"noise{noise:g}_reasons"] = reasons
        ax.set_xscale("log")
        ax.set(xlabel="distance from Earth's centre (km)", ylabel="correct (%)" if j == 0 else "",
               title=f"Correct rate vs distance, read noise {noise:g} (95 % CI)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "outcomes_vs_distance.png", **ac.SAVE)
    plt.close(fig)

    # ---------------- range error vs distance per phase (nominal noise, tracker attitude, solved trials)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), dpi=130)
    S["range_vs_distance"] = {}
    L += [f"RANGE ERROR |dD|/D (%) of solved trials, read noise {nominal:g}, tracker attitude: median [bootstrap 95 %], p95"]
    for ph in phases:
        r = select(rows, ATT_TRACKER, noise=nominal, phase=ph, solved=True)
        x, v = per_distance(r, "range_pct")
        band(ax[0], x, v[:, 0], v[:, 1], v[:, 2], v[:, 3], PHASE_COLORS.get(ph, ac.BLUE), f"phase {ph:g}°")
        xk, vk = per_distance(r, "range_km")
        band(ax[1], xk, vk[:, 0], vk[:, 1], vk[:, 2], vk[:, 3], PHASE_COLORS.get(ph, ac.BLUE), f"phase {ph:g}°")
        L += [f"  phase {ph:g}°: " + "  ".join(f"{d:,.0f}: {m:.3f} [{lo:.3f},{hi:.3f}] p95 {p9:.2f}" for d, (m, lo, hi, p9, n) in zip(x, v))]
        S["range_vs_distance"][f"phase{ph:g}"] = {"distance": x.tolist(), "median_pct": v[:, 0].tolist(), "lo": v[:, 1].tolist(),
                                                  "hi": v[:, 2].tolist(), "p95_pct": v[:, 3].tolist(), "n": v[:, 4].tolist(),
                                                  "median_km": vk[:, 0].tolist(), "p95_km": vk[:, 3].tolist()}
    for axis, unit in ((ax[0], "%"), (ax[1], "km")):
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set(xlabel="distance from Earth's centre (km)", ylabel=f"|range error| ({unit})",
                 title=f"Range error vs distance ({unit}): median, 95 % band, dotted = p95")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "range_vs_distance.png", **ac.SAVE)
    plt.close(fig)
    L += [""]

    # ---------------- attitude contribution: tracker vs true attitude (nominal noise, all phases)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), dpi=130)
    S["attitude"] = {}
    L += [f"ATTITUDE CONTRIBUTION (read noise {nominal:g}, all phases, solved trials): tracker vs TRUE attitude"]
    for att, color, label in ((ATT_TRACKER, ac.ORANGE, "attitude from the star tracker"), (ATT_TRUE, ac.BLUE, "true attitude")):
        r = select(rows, att, noise=nominal, solved=True)
        for k, (col, absolute) in enumerate((("cross_km", False), ("range_pct", True))):
            x, v = per_distance(r, col, absolute)
            band(ax[k], x, v[:, 0], v[:, 1], v[:, 2], v[:, 3], color, label)
            S["attitude"][f"{'tracker' if att == ATT_TRACKER else 'true'}_{col}"] = {"distance": x.tolist(), "median": v[:, 0].tolist(),
                                                                                    "lo": v[:, 1].tolist(), "hi": v[:, 2].tolist(), "p95": v[:, 3].tolist()}
            L += [f"  {label:32s} {col:9s}: " + "  ".join(f"{d:,.0f}: {m:.3g} (p95 {p9:.3g})" for d, (m, lo, hi, p9, n) in zip(x, v))]
    att_err = select(rows, ATT_TRACKER, noise=nominal)
    ap, ar = att_err[:, C["att_pointing"]], att_err[:, C["att_roll"]]
    L += [f"  tracker attitude error used (arcsec): pointing median {np.nanmedian(ap):.2f} p95 {np.nanpercentile(ap, 95):.2f}; "
          f"roll median {np.nanmedian(ar):.2f} p95 {np.nanpercentile(ar, 95):.2f}", ""]
    S["attitude"]["tracker_error_arcsec"] = {"pointing_median": float(np.nanmedian(ap)), "pointing_p95": float(np.nanpercentile(ap, 95)),
                                             "roll_median": float(np.nanmedian(ar)), "roll_p95": float(np.nanpercentile(ar, 95))}
    ax[0].set(ylabel="cross-range error (km)", title="Cross-range error: the star tracker's share")
    ax[1].set(ylabel="|range error| (%)", title="Range error: unchanged by the attitude source")
    for axis in ax:
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set(xlabel="distance from Earth's centre (km)")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "attitude_contribution.png", **ac.SAVE)
    plt.close(fig)

    # ---------------- sphere vs ellipsoid model (signed range error: the oblateness bias)
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=130)
    r = select(rows, ATT_TRUE, noise=nominal, solved=True)
    if len(r) == 0:
        r = select(rows, ATT_TRACKER, noise=nominal, solved=True)
    S["sphere_vs_ellipsoid"] = {}
    L += [f"SPHERE vs ELLIPSOID MODEL on the same limb points (signed range error %, median [bootstrap 95 %]), read noise {nominal:g}"]
    for col, color, label in (("range_pct", ac.BLUE, "ellipsoid (WGS84) model"), ("range_sphere_pct", ac.ORANGE, "sphere model, R = R_EQ")):
        if col == "range_sphere_pct":
            vals = 100 * r[:, C["range_sphere_km"]] / r[:, C["distance"]]
            rr = r.copy(); rr[:, C["range_pct"]] = vals
        else:
            rr = r
        x, v = per_distance(rr, "range_pct", absolute=False)
        band(ax, x, v[:, 0], v[:, 1], v[:, 2], None, color, label)
        S["sphere_vs_ellipsoid"][label] = {"distance": x.tolist(), "median_pct": v[:, 0].tolist(), "lo": v[:, 1].tolist(), "hi": v[:, 2].tolist()}
        L += [f"  {label:26s}: " + "  ".join(f"{d:,.0f}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]" for d, (m, lo, hi, p9, n) in zip(x, v))]
    ax.axhline(0, color=ac.INK2, lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="distance from Earth's centre (km)", ylabel="signed range error (%)",
           title="Oblateness: fitting a sphere to the real (oblate) Earth biases the range")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sphere_vs_ellipsoid.png", **ac.SAVE)
    plt.close(fig)
    L += [""]

    # ---------------- noise: nominal vs 2x (tracker attitude, all phases)
    if len(noises) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=130)
        S["noise"] = {}
        L += ["READ NOISE: |range error| (%) median, all phases, tracker attitude"]
        for noise, color in zip(noises, (ac.BLUE, ac.ORANGE)):
            r = select(rows, ATT_TRACKER, noise=noise, solved=True)
            x, v = per_distance(r, "range_pct")
            band(ax, x, v[:, 0], v[:, 1], v[:, 2], v[:, 3], color, f"read noise {noise:g}")
            S["noise"][f"noise{noise:g}"] = {"distance": x.tolist(), "median_pct": v[:, 0].tolist(), "lo": v[:, 1].tolist(),
                                             "hi": v[:, 2].tolist(), "p95_pct": v[:, 3].tolist()}
            L += [f"  noise {noise:g}: " + "  ".join(f"{d:,.0f}: {m:.3f}" for d, (m, lo, hi, p9, n) in zip(x, v))]
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlabel="distance from Earth's centre (km)", ylabel="|range error| (%)", title="Range error vs distance: nominal vs 2x read noise")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "noise.png", **ac.SAVE)
        plt.close(fig)
        L += [""]

    # ---------------- headline numbers
    geo = select(rows, ATT_TRACKER, noise=nominal, solved=True)
    geo = geo[geo[:, C["distance"]] == dists.min()]
    lunar_like = select(rows, ATT_TRACKER, noise=nominal, solved=True)
    d_l = dists[np.argmin(np.abs(dists - 384_400))]
    lunar_like = lunar_like[lunar_like[:, C["distance"]] == d_l]
    S["headline"] = {"geo_km": float(dists.min()), "geo_range_pct_median": float(np.nanmedian(np.abs(geo[:, C["range_pct"]]))) if len(geo) else None,
                     "geo_total_km_median": float(np.nanmedian(geo[:, C["total_km"]])) if len(geo) else None,
                     "near_lunar_km": float(d_l), "near_lunar_range_pct_median": float(np.nanmedian(np.abs(lunar_like[:, C["range_pct"]]))) if len(lunar_like) else None,
                     "near_lunar_total_km_median": float(np.nanmedian(lunar_like[:, C["total_km"]])) if len(lunar_like) else None,
                     "n_trials": int(n_trials)}
    L += ["HEADLINE: " + json.dumps(S["headline"])]

    report = "\n".join(L)
    print(report)
    if a.theme == "light":
        (out / "report.txt").write_text(report + "\n")
        (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(f"\nsaved {out}/report.txt, summary.json, outcomes_vs_distance.png, range_vs_distance.png, attitude_contribution.png, "
          f"sphere_vs_ellipsoid.png, noise.png")


if __name__ == "__main__":
    main()
