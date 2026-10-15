"""Day 5 campaign: run every condition, one CSV per condition in results/campaign/.

    python -m experiments.campaign --backend gpu                 # the full campaign (Colab GPU)
    python -m experiments.campaign --backend cpu --scale 0.002   # tiny smoke run on a laptop
    python -m experiments.campaign --backend gpu --only baseline

Conditions (all share SEED, so trial k is the same sky in every condition: paired sweeps):
  baseline     read noise 5, no hot pixels                     100,000 trials
  noise_X      read noise X in NOISE_SWEEP                       10,000 trials each
  hot_N        N hot pixels per frame in HOT_SWEEP (noise 5)     10,000 trials each
Safe to interrupt and re-run: finished trials are skipped (tracker.campaign.run_condition).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from tracker.campaign import run_condition
from tracker.catalog import load_catalog
from tracker.identify import EPS_ARCSEC, build_pair_table
from tracker.render_cpu import READ_NOISE

SEED = 2026
N_BASELINE = 100_000
N_SWEEP = 10_000
NOISE_SWEEP = (2.5, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)     # 5.0 = the first 10,000 baseline trials
HOT_SWEEP = (5, 10, 20, 40, 80)


def conditions(scale=1.0):
    n_base, n_sweep = max(1, round(N_BASELINE * scale)), max(1, round(N_SWEEP * scale))
    out = [("baseline", n_base, READ_NOISE, 0)]
    out += [(f"noise_{x:g}", n_sweep, x, 0) for x in NOISE_SWEEP]
    out += [(f"hot_{n}", n_sweep, READ_NOISE, n) for n in HOT_SWEEP]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--scale", type=float, default=1.0, help="fraction of the trial counts (smoke runs)")
    p.add_argument("--only", nargs="*", help="condition names to run (default: all)")
    p.add_argument("--out", default="results/campaign")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = load_catalog(5.5)
    table = build_pair_table(cat.r)
    meta = {"seed": SEED, "eps_arcsec": EPS_ARCSEC, "backend": a.backend, "scale": a.scale,
            "python": platform.python_version(), "machine": platform.platform()}
    if a.backend == "gpu":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        meta["gpu"] = props["name"].decode()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    for name, n, noise, hot in conditions(a.scale):
        if a.only and name not in a.only:
            continue
        t0 = time.perf_counter()
        last = [t0]

        def progress(done, total):
            if time.perf_counter() - last[0] > 30 or done == total:
                last[0] = time.perf_counter()
                print(f"  {name}: {done}/{total}  ({(last[0] - t0) / max(done, 1) * 1000:.2f} ms/trial)", flush=True)

        ran = run_condition(out / f"{name}.csv", n, SEED, noise, hot, cat, table,
                            backend=a.backend, progress=progress)
        print(f"{name}: {n} trials, {ran} run now, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
