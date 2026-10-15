"""Day 3 Monte Carlo: the full CPU tracker on random skies, with per-stage timing.

    python -m experiments.montecarlo                          # 1000 trials, eps 25"
    python -m experiments.montecarlo --eps 20 25 31           # compare tolerances
    python -m experiments.montecarlo --trials 200 --eps 60
    python -m experiments.montecarlo --read-noise 15 --hot-pixels 10 --out results/mc_noise15_hot10

Each frame is rendered and detected ONCE, then identified with every eps, so the eps
values are compared on exactly the same images. Outcomes are counted separately:
  correct      solution with total attitude error < CORRECT_ARCSEC
  wrong        a solution, but further off than that (confident and wrong: the bad one)
  no solution  identify() returned None (the tracker knows it doesn't know)
"""
import argparse
import time
from pathlib import Path

import numpy as np

from tracker.camera import random_attitude, unproject
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.identify import EPS_ARCSEC, build_pair_table, identify
from tracker.render_cpu import BACKGROUND, READ_NOISE, add_hot_pixels, render_sky
from tracker.evaluate import CORRECT_ARCSEC, score
from tracker.solve import solve_attitude
from tracker.stats import wilson_interval



def rate(k, n):
    lo, hi = wilson_interval(k, n)
    return f"{k:5d}/{n}  {100 * k / n:6.2f} %  [{100 * lo:6.2f}, {100 * hi:6.2f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--eps", type=float, nargs="+", default=[EPS_ARCSEC])
    p.add_argument("--read-noise", type=float, default=READ_NOISE, help="counts per pixel (baseline 5)")
    p.add_argument("--hot-pixels", type=int, default=0, help="fake single-pixel stars per frame")
    p.add_argument("--out", default="results/montecarlo_cpu")
    a = p.parse_args()

    cat = load_catalog(5.5)
    t0 = time.perf_counter()
    table = build_pair_table(cat.r)
    t_table = time.perf_counter() - t0
    rng = np.random.default_rng(a.seed)

    E = len(a.eps)
    n_true = np.zeros(a.trials, int)
    n_det = np.zeros(a.trials, int)
    boresight = np.zeros((a.trials, 3))
    t_render, t_detect = np.zeros(a.trials), np.zeros(a.trials)
    t_ident, t_svd = np.zeros((a.trials, E)), np.zeros((a.trials, E))
    outcome = np.zeros((a.trials, E), int)               # 0 correct, 1 wrong, 2 no solution
    err = np.full((a.trials, E, 3), np.nan)              # total, pointing, roll (arcsec)
    n_match = np.zeros((a.trials, E), int)

    for t in range(a.trials):
        A = random_attitude(rng)
        boresight[t] = A[2]
        s = time.perf_counter()
        img, _, _, idx = render_sky(A, cat, rng=rng, read_noise=a.read_noise)
        if a.hot_pixels:
            add_hot_pixels(img, a.hot_pixels, rng)
        t_render[t] = time.perf_counter() - s
        s = time.perf_counter()
        u, v, f = detect(img, BACKGROUND, a.read_noise)
        t_detect[t] = time.perf_counter() - s
        n_true[t], n_det[t] = len(idx), len(u)
        for e, eps in enumerate(a.eps):
            s = time.perf_counter()
            sol = identify(unproject(u, v), table, eps_arcsec=eps)
            t_ident[t, e] = time.perf_counter() - s
            outcome[t, e], *err[t, e], n_match[t, e] = score(sol, A)
            if sol is None:
                continue
            # one SVD on the final matched set: how much of identify is the solve itself
            b_m, r_m = unproject(u[sol.det], v[sol.det]), cat.r[sol.cat]
            s = time.perf_counter()
            solve_attitude(b_m, r_m)
            t_svd[t, e] = time.perf_counter() - s

    N = a.trials
    pct = np.percentile
    L = [f"Day 3 Monte Carlo (CPU): {N} random attitudes, seed {a.seed}, catalog mag < 5.5, "
         f"pair table {len(table.angle):,} pairs built in {1000 * t_table:.0f} ms (offline, once)",
         f"conditions: read noise {a.read_noise:g} counts, {a.hot_pixels} hot pixels per frame",
         f"stars in view: mean {n_true.mean():.1f}, min {n_true.min()}, max {n_true.max()};  "
         f"detected: mean {n_det.mean():.1f}, min {n_det.min()}",
         f"correct = total error < {CORRECT_ARCSEC}\"   rates with 95 % Wilson interval", ""]
    for e, eps in enumerate(a.eps):
        ok = outcome[:, e] == 0
        L += [f"eps = {eps:g}\"",
              f"  correct      {rate(ok.sum(), N)}",
              f"  wrong        {rate((outcome[:, e] == 1).sum(), N)}",
              f"  no solution  {rate((outcome[:, e] == 2).sum(), N)}"]
        if ok.any():
            tot, poi, rol = err[ok, e].T
            L += [f"  correct frames: pointing median {np.median(poi):.2f}\" p99 {pct(poi, 99):.2f}\" | "
                  f"roll median {np.median(rol):.2f}\" p99 {pct(rol, 99):.2f}\" | "
                  f"worst total {tot.max():.1f}\" | matched stars median {np.median(n_match[ok, e]):.0f}"]
        wrong = outcome[:, e] == 1
        if wrong.any():
            L += [f"  wrong frames: smallest total error {err[wrong, e, 0].min():.0f}\""]
        L += [f"  identify time: median {1000 * np.median(t_ident[:, e]):.2f} ms, "
              f"p99 {1000 * pct(t_ident[:, e], 99):.2f} ms, max {1000 * t_ident[:, e].max():.1f} ms", ""]

    # per-stage timing for the last (default: largest) eps
    e = E - 1
    stages = [("render", t_render), ("detect", t_detect),
              (f"identify (eps {a.eps[e]:g}\", incl. solves)", t_ident[:, e])]
    total = sum(x for _, x in stages)
    L += [f"PER-STAGE TIME, one CPU core, ms/frame (eps {a.eps[e]:g}\")",
          f"  {'stage':38s} {'median':>8s} {'mean':>8s} {'p99':>8s} {'share':>7s}"]
    for name, x in stages:
        L += [f"  {name:38s} {1000 * np.median(x):8.2f} {1000 * x.mean():8.2f} {1000 * pct(x, 99):8.2f} "
              f"{100 * x.sum() / total.sum():6.1f} %"]
    solved = outcome[:, e] != 2
    L += [f"  {'  of which one SVD solve':38s} {1000 * np.median(t_svd[solved, e]):8.3f} "
          f"{1000 * t_svd[solved, e].mean():8.3f} {1000 * pct(t_svd[solved, e], 99):8.3f}",
          f"  {'total':38s} {1000 * np.median(total):8.2f} {1000 * total.mean():8.2f} {1000 * pct(total, 99):8.2f}"]

    fail = outcome[:, e] != 0
    if fail.any():
        L += ["", f"NOT-CORRECT FRAMES at eps {a.eps[e]:g}\" (trial: stars in view / detected / outcome)"]
        for t in np.flatnonzero(fail)[:20]:
            L += [f"  {t:4d}: {n_true[t]:3d} / {n_det[t]:3d} / {['correct', 'wrong', 'no solution'][outcome[t, e]]}"]

    report = "\n".join(L)
    print(report)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(report + "\n")
    cols = [boresight, n_true[:, None], n_det[:, None], 1000 * t_render[:, None], 1000 * t_detect[:, None]]
    head = ["bx", "by", "bz", "n_true", "n_detected", "t_render_ms", "t_detect_ms"]
    for e, eps in enumerate(a.eps):
        cols += [outcome[:, e, None], err[:, e], n_match[:, e, None], 1000 * t_ident[:, e, None]]
        head += [f"{k}_eps{eps:g}" for k in ("outcome", "total", "pointing", "roll", "n_matched", "t_identify_ms")]
    np.savetxt(out.with_suffix(".csv"), np.hstack(cols), delimiter=",", fmt="%.6g", header=",".join(head))
    print(f"saved {out}.txt/.csv")


if __name__ == "__main__":
    main()
