"""Day 5 analysis: results/campaign/*.csv -> report, summary.json and figures.

    python -m experiments.analyze_campaign                        # results/campaign
    python -m experiments.analyze_campaign --dir results/campaign_smoke

Every rate is printed with its trial count and 95 % Wilson interval. The noise sweep and the
hot-pixel sweep include the first N_SWEEP baseline trials as their noise-5 / 0-hot-pixel point
(same skies: the campaign uses common random numbers).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from experiments.campaign import HOT_SWEEP, NOISE_SWEEP
from tracker.campaign import COLUMNS
from tracker.evaluate import CORRECT, OUTCOMES
from tracker.render_cpu import READ_NOISE
from tracker.stats import outcome_rates

# reference palette (dataviz skill): light surface, ink, series 1/2, status critical
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
C = {name: k for k, name in enumerate(COLUMNS)}
SAVE = {}                       # extra savefig kwargs (dark theme: transparent background)

THEMES = {  # the website's dark "lost in space" theme: same series hues stepped for a dark surface
    "light": dict(SURFACE="#fcfcfb", INK="#0b0b0b", INK2="#52514e", GRID="#e4e3df",
                  BLUE="#2a78d6", ORANGE="#eb6834", RED="#d03b3b",
                  seq=["#cde2fb", "#6da7ec", "#256abf", "#0d366b"], transparent=False),
    "dark": dict(SURFACE="#0b0e24", INK="#eef0ff", INK2="#b8bce0", GRID="#2a2f58",
                 BLUE="#9db4ff", ORANGE="#ff9a6e", RED="#ff6b6b",
                 seq=["#241a4d", "#5a3bb5", "#9b7bff", "#e6dcff"], transparent=True),
}


def set_theme(name):
    t = THEMES[name]
    globals().update({k: v for k, v in t.items() if k.isupper()})
    globals()["SEQ"] = matplotlib.colors.LinearSegmentedColormap.from_list(name, t["seq"])
    SAVE.clear()
    SAVE.update({"transparent": True} if t["transparent"] else {})
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "legend.labelcolor": INK, "savefig.facecolor": SURFACE})


set_theme("light")


def load(d, name):
    path = d / f"{name}.csv"
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2) if path.exists() else None


def pct(lo, hi):
    return f"[{100 * lo:6.3f}, {100 * hi:6.3f}] %"


def rate_line(label, t, i=0):
    n = t["n"][i]
    parts = [f"{label:>14s}  n={n:>7,d}"]
    for name in ("correct", "wrong", "none"):
        k = t[name][i]
        parts.append(f"{name} {k:>6,d} = {100 * k / n:7.3f} % {pct(t[name + '_lo'][i], t[name + '_hi'][i])}")
    return "  ".join(parts)


def sweep(base, runs, key, levels, base_level, n_sweep):
    """Rows for a sweep: the baseline's first n_sweep trials stand in for base_level."""
    pts = [(base_level, base[base[:, C["trial"]] < n_sweep])]
    pts += [(lv, r) for lv, r in zip(levels, runs) if r is not None]
    pts.sort(key=lambda p: p[0])
    lv = np.array([p[0] for p in pts], dtype=float)
    rows = np.vstack([p[1] for p in pts])
    return lv, rows


def errorbar_rates(ax, x, t, color, label=None):
    rate = 100 * t["correct"] / t["n"]
    yerr = np.clip([rate - 100 * t["correct_lo"], 100 * t["correct_hi"] - rate], 0, None)   # 100 %: -1e-14
    ax.errorbar(x, rate, yerr=yerr, color=color, lw=2, marker="o", ms=7, capsize=3, label=label,
                mec=SURFACE, mew=1.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/campaign")
    p.add_argument("--theme", choices=THEMES, default="light", help="dark: transparent figures for the website")
    a = p.parse_args()
    set_theme(a.theme)
    d = Path(a.dir)
    out = d / ("analysis" if a.theme == "light" else "analysis_dark")
    out.mkdir(exist_ok=True)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}

    base = load(d, "baseline")
    noise_runs = [load(d, f"noise_{x:g}") for x in NOISE_SWEEP]
    hot_runs = [load(d, f"hot_{n}") for n in HOT_SWEEP]
    n_sweep = max((len(r) for r in noise_runs + hot_runs if r is not None), default=len(base))
    L, S = [], {"meta": meta}

    # ---------------- baseline
    tb = outcome_rates(base[:, C["outcome"]])
    ok = base[:, C["outcome"]] == CORRECT
    poi, rol, tot = base[ok, C["pointing"]], base[ok, C["roll_err"]], base[ok, C["total"]]
    q = [50, 90, 99, 99.9]
    L += [f"STAR TRACKER CAMPAIGN  ({meta.get('backend', '?')}, {meta.get('gpu', meta.get('machine', ''))}, "
          f"seed {meta.get('seed', '?')}, eps {meta.get('eps_arcsec', '?')}\")", "",
          "BASELINE (read noise 5, no hot pixels)", rate_line("baseline", tb),
          f"  stars in view: mean {base[:, C['n_in_view']].mean():.1f}, min {base[:, C['n_in_view']].min():.0f}, "
          f"max {base[:, C['n_in_view']].max():.0f};  detected mean {base[:, C['n_detected']].mean():.1f}",
          "  accuracy of correct trials (arcsec):   " + "   ".join(f"p{x:g}" for x in q),
          "    pointing  " + "  ".join(f"{np.percentile(poi, x):7.2f}" for x in q),
          "    roll      " + "  ".join(f"{np.percentile(rol, x):7.2f}" for x in q),
          "    total     " + "  ".join(f"{np.percentile(tot, x):7.2f}" for x in q)]
    if (~ok).any():
        L += [f"  worst correct total {tot.max():.1f}\";  non-correct trials: "
              + ", ".join(f"#{int(r[C['trial']])} ({OUTCOMES[int(r[C['outcome']])]}, {int(r[C['n_in_view']])} in view, "
                          f"{int(r[C['n_detected']])} detected)" for r in base[~ok][:10])]
    S["baseline"] = {"n": int(tb["n"][0]), "correct": int(tb["correct"][0]), "wrong": int(tb["wrong"][0]),
                     "no_solution": int(tb["none"][0]),
                     "correct_ci": [float(tb["correct_lo"][0]), float(tb["correct_hi"][0])],
                     "wrong_ci": [float(tb["wrong_lo"][0]), float(tb["wrong_hi"][0])],
                     "pointing_pct": {f"p{x:g}": float(np.percentile(poi, x)) for x in q},
                     "roll_pct": {f"p{x:g}": float(np.percentile(rol, x)) for x in q}}

    # ---------------- sweeps
    for title, key, levels, runs, base_level, unit in [
            ("NOISE SWEEP", "read_noise", NOISE_SWEEP, noise_runs, READ_NOISE, "read noise"),
            ("HOT-PIXEL SWEEP (read noise 5)", "hot_pixels", HOT_SWEEP, hot_runs, 0, "hot pixels")]:
        lv, rows = sweep(base, runs, key, levels, base_level, n_sweep)
        t = outcome_rates(rows[:, C["outcome"]], rows[:, C[key]])
        L += ["", title] + [rate_line(f"{unit} {g:g}", t, i) for i, g in enumerate(t["group"])]
        det = [rows[rows[:, C[key]] == g, C["n_detected"]].mean() for g in t["group"]]
        S[key] = {"level": t["group"].tolist(), "n": t["n"].tolist(), "correct": t["correct"].tolist(),
                  "wrong": t["wrong"].tolist(), "no_solution": t["none"].tolist(),
                  "correct_lo": t["correct_lo"].tolist(), "correct_hi": t["correct_hi"].tolist(),
                  "mean_detected": [float(x) for x in det]}
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
        errorbar_rates(ax[0], t["group"], t, BLUE)
        n_txt = f"n = {t['n'].min():,}" if t["n"].min() == t["n"].max() else f"n = {t['n'].min():,}-{t['n'].max():,}"
        ax[0].set(xlabel=unit, ylabel="correct (%)", title=f"Success rate vs {unit} (95 % CI, {n_txt} per point)")
        ax[1].plot(t["group"], det, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
        ax[1].set(xlabel=unit, ylabel="dots detected per frame (mean)", title=f"Detections vs {unit}")
        fig.tight_layout()
        fig.savefig(out / f"{key}_sweep.png", **SAVE)
        plt.close(fig)

    # ---------------- success vs stars in view (baseline + the hardest noise level that has failures)
    hard = next((r for r in reversed(noise_runs) if r is not None and (r[:, C["outcome"]] != CORRECT).any()), None)
    bins = np.array([0, 10, 15, 20, 25, 30, 40, 60, 200])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    L += ["", "SUCCESS VS STARS IN VIEW"]
    for rows, color, label in [(base, BLUE, "baseline (noise 5)"),
                               (hard, ORANGE, f"noise {hard[0, C['read_noise']]:g}" if hard is not None else None)]:
        if rows is None:
            continue
        b = np.digitize(rows[:, C["n_in_view"]], bins) - 1
        t = outcome_rates(rows[:, C["outcome"]], b)
        centers = np.array([rows[b == g, C["n_in_view"]].mean() for g in t["group"]])   # real mean per bin
        errorbar_rates(ax[0], centers, t, color, label)
        L += [f"  {label}:"] + [rate_line(f"{bins[g]}-{bins[g + 1] - 1} stars", t, i) for i, g in enumerate(t["group"])]
        S.setdefault("stars_in_view", {})[label] = {"bin_lo": bins[t["group"]].tolist(), "bin_hi": (bins[t["group"] + 1] - 1).tolist(),
                                                    "n": t["n"].tolist(), "correct": t["correct"].tolist(),
                                                    "correct_lo": t["correct_lo"].tolist(), "correct_hi": t["correct_hi"].tolist()}
    ax[0].set(xlabel="catalog stars in view (mag < 5.5)", ylabel="correct (%)", title="Success vs stars in view (95 % CI)")
    ax[0].legend(frameon=False)
    ax[1].hist(base[:, C["n_in_view"]], bins=np.arange(0, base[:, C["n_in_view"]].max() + 2), color=BLUE, rwidth=0.85)
    ax[1].set(xlabel="catalog stars in view", ylabel="baseline trials", title="How many stars a random pointing sees")
    fig.tight_layout()
    fig.savefig(out / "success_vs_stars.png", **SAVE)
    plt.close(fig)

    # ---------------- failure sky map (Mollweide; RA increases to the left, like the sky seen from inside)
    def lon(ra):
        return np.radians(180.0 - ra)

    fig = plt.figure(figsize=(11, 5.6), dpi=130)
    ax = fig.add_subplot(projection="mollweide")
    hb = ax.hexbin(lon(base[:, C["ra"]]), np.radians(base[:, C["dec"]]), C=base[:, C["n_in_view"]],
                   gridsize=45, cmap=SEQ, reduce_C_function=np.mean, linewidths=0.8, edgecolors="face")
    fig.colorbar(hb, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08, label="mean catalog stars in view")
    every = [r for r in [base] + noise_runs + hot_runs if r is not None]
    fails = np.vstack([r[r[:, C["outcome"]] != CORRECT] for r in every])
    if len(fails):
        n_base_fail = int((base[:, C["outcome"]] != CORRECT).sum())
        ax.scatter(lon(fails[:, C["ra"]]), np.radians(fails[:, C["dec"]]), s=16, color=RED, edgecolor=SURFACE,
                   linewidth=0.4, label=f"not correct: {len(fails):,} of {sum(len(r) for r in every):,} trials "
                                        f"(baseline {n_base_fail:,}; the rest from the noise / hot-pixel sweeps)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), frameon=False)
    ax.set_xticks(np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]))
    ax.set_xticklabels([])                          # the built-in labels sit under the hexes: draw our own
    halo = [patheffects.withStroke(linewidth=2.5, foreground=SURFACE)]
    for x in (-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150):
        ax.text(np.radians(x), np.radians(-4), f"{(180 - x) % 360}°", ha="center", va="top", fontsize=7,
                color=INK, zorder=10, path_effects=halo)
    ax.set_xlabel("right ascension (increases to the left)")
    ax.set_title("Where the tracker fails: boresight of every non-correct trial over star density (RA, Dec)")
    fig.savefig(out / "failure_map.png", bbox_inches="tight", **SAVE)
    plt.close(fig)
    S["failures"] = {"ra": fails[:, C["ra"]].round(2).tolist(), "dec": fails[:, C["dec"]].round(2).tolist(),
                     "read_noise": fails[:, C["read_noise"]].tolist(), "hot_pixels": fails[:, C["hot_pixels"]].tolist(),
                     "n_in_view": fails[:, C["n_in_view"]].tolist()} if len(fails) else {}

    # ---------------- accuracy histograms (small multiples, not overlaid)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    for axis, x, color, name in [(ax[0], poi, BLUE, "pointing (boresight)"), (ax[1], rol, ORANGE, "roll (about boresight)")]:
        hi = np.percentile(x, 99.9)
        axis.hist(x, bins=100, range=(0, hi), color=color)
        for qq, ls in [(50, "-"), (99, "--")]:
            v = np.percentile(x, qq)
            axis.axvline(v, color=INK2, lw=1, ls=ls)
            axis.annotate(f"p{qq} {v:.1f}\"", (v, axis.get_ylim()[1] * 0.9), xytext=(4, 0), textcoords="offset points",
                          fontsize=8, color=INK2)
        axis.set(xlabel="error (arcsec)", ylabel="correct baseline trials", title=f"{name} error, n={len(x):,}")
    fig.tight_layout()
    fig.savefig(out / "accuracy_hist.png", **SAVE)
    plt.close(fig)

    report = "\n".join(L)
    print(report)
    if a.theme == "light":                    # the numbers are theme-independent: write them once
        (out / "report.txt").write_text(report + "\n")
        (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(f"\nsaved {out}/report.txt, summary.json, read_noise_sweep.png, hot_pixels_sweep.png, "
          f"success_vs_stars.png, failure_map.png, accuracy_hist.png")


if __name__ == "__main__":
    main()
