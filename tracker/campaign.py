"""The Monte Carlo campaign engine: many random skies -> one result row per trial.

Every trial is fully determined by (seed, trial number):
  attitude     random_attitude(default_rng([seed, trial, 0]))
  noise        counter-based noise with key (seed, trial)        -- tracker/noise.py
  hot pixels   draw_hot_pixels(n, default_rng([seed, trial, 1]))
so a trial gives the same row alone or inside any batch, on the CPU or the GPU, and a
crashed run resumes exactly. Conditions that share a seed see the SAME skies, so noise and
hot-pixel sweeps are paired comparisons (common random numbers).

backend "gpu" renders and detects on the GPU (tracker/gpu.py, imported only then);
"cpu" does the same arithmetic in NumPy (slower; used for tests and small runs).
Identification and the attitude solve always run on the CPU in float64.
"""
from pathlib import Path

import numpy as np

from tracker.batch import pack_frames, project_frames
from tracker.camera import pointing_from_attitude, random_attitude, unproject
from tracker.detect_cpu import detect
from tracker.evaluate import score
from tracker.identify import identify
from tracker.render_cpu import BACKGROUND, draw_hot_pixels, render

COLUMNS = ("trial", "read_noise", "hot_pixels", "ra", "dec", "roll",
           "n_in_view", "n_detected", "outcome", "total", "pointing", "roll_err", "n_matched")


def trial_attitude(seed, trial):
    return random_attitude(np.random.default_rng([seed, int(trial), 0]))


def trial_hot_pixels(seed, trial, n):
    return draw_hot_pixels(n, np.random.default_rng([seed, int(trial), 1]))


def run_batch(trials, seed, read_noise, hot_pixels, catalog, table, backend="cpu"):
    """Run trials (int array) under one condition. Returns (len(trials), len(COLUMNS)) float64."""
    trials = np.asarray(trials, dtype=np.int64)
    As = [trial_attitude(seed, t) for t in trials]
    frames, idx = project_frames(As, catalog)
    draws = [trial_hot_pixels(seed, t, hot_pixels) for t in trials] if hot_pixels else None

    if backend == "gpu":
        from tracker import gpu
        imgs = gpu.render_batch(pack_frames(frames), seed=seed, frame_ids=trials, read_noise=read_noise)
        if draws:
            gpu.add_hot_pixels_batch(imgs, draws)
        found = gpu.detect_batch(imgs, BACKGROUND, read_noise)
    elif backend == "cpu":
        found = []
        for k, (u, v, f) in enumerate(frames):
            img = render(u, v, f, read_noise=read_noise, noise_key=(seed, int(trials[k])))
            if draws:
                img.flat[draws[k][0]] += draws[k][1]
            found.append(detect(img, BACKGROUND, read_noise))
    else:
        raise ValueError(f"unknown backend {backend!r}")

    rows = np.empty((len(trials), len(COLUMNS)))
    for k, t in enumerate(trials):
        u, v, _ = found[k]
        outcome, total, pointing, roll_err, n_matched = score(identify(unproject(u, v), table), As[k])
        rows[k] = (t, read_noise, hot_pixels, *pointing_from_attitude(As[k]),
                   len(idx[k]), len(u), outcome, total, pointing, roll_err, n_matched)
    return rows


def run_condition(out_csv, n_trials, seed, read_noise, hot_pixels, catalog, table,
                  backend="cpu", batch=128, progress=None):
    """Make sure out_csv holds trials 0..n_trials-1 of this condition. Trials already in the
    file are skipped (resume); new rows are appended after every batch, so an interrupted run
    loses at most one batch. Returns the number of trials run now."""
    out_csv = Path(out_csv)
    done = set()
    if out_csv.exists() and out_csv.stat().st_size:
        done = set(np.loadtxt(out_csv, delimiter=",", skiprows=1, ndmin=2)[:, 0].astype(int).tolist())
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text(",".join(COLUMNS) + "\n")
    todo = np.array([t for t in range(n_trials) if t not in done], dtype=np.int64)
    for start in range(0, len(todo), batch):
        rows = run_batch(todo[start:start + batch], seed, read_noise, hot_pixels, catalog, table, backend)
        with open(out_csv, "a") as fh:
            np.savetxt(fh, rows, delimiter=",", fmt="%.10g")
        if progress:
            progress(min(start + batch, len(todo)), len(todo))
    return len(todo)
