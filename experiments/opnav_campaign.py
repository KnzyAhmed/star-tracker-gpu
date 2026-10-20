"""Day 12 campaign: optical navigation over distance x Sun phase x read noise, one CSV per cell in results/opnav/.

    python -m experiments.opnav_campaign --backend cpu --workers 10          # the full campaign on the Mac (all cores)
    python -m experiments.opnav_campaign --backend gpu                       # Colab T4 (Earth + stars on the GPU)
    python -m experiments.opnav_campaign --backend cpu --workers 10 --scale 0.005   # smoke run
    python -m experiments.opnav_campaign --only d00_ph90_n5 d11_ph90_n5

Cells: 12 distances log-spaced from GEO (42,164 km) to 1,000,000 km x phase angles 0, 45, 90, 135 deg x read
noise 5 (nominal) and 10 (2x); N_CELL trials each. All cells share SEED, so trial k is the same direction to
Earth, Sun azimuth, camera offset and roll everywhere (paired sweeps). Nominal-noise cells also write a
true-attitude row per trial (att_src 1) to separate the star tracker's contribution.
Safe to interrupt and re-run: finished trials are skipped (tracker.opnav_campaign.run_condition).
"""
import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tracker.opnav import MIN_LIMB_POINTS, REFINE_ITERATIONS
from tracker.opnav_campaign import CORRECT_FRACTION, EDGE_MARGIN_PX, _init_worker, run_condition
from tracker.world import R_EQ, R_POL

SEED = 2027
GEO, FAR = 42_164.0, 1_000_000.0
DISTANCES = np.geomspace(GEO, FAR, 12)
PHASES = (0.0, 45.0, 90.0, 135.0)
NOISES = (5.0, 10.0)
N_CELL = 1000


def cells(scale=1.0):
    """-> list of (name, n_trials, distance, phase, read_noise, true_attitude)."""
    n = max(1, round(N_CELL * scale))
    return [(f"d{i:02d}_ph{int(ph)}_n{noise:g}", n, float(d), ph, noise, noise == NOISES[0])
            for noise in NOISES for ph in PHASES for i, d in enumerate(DISTANCES)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--workers", type=int, default=1, help="processes for the CPU backend")
    p.add_argument("--scale", type=float, default=1.0, help="fraction of the trial counts (smoke runs)")
    p.add_argument("--batch", type=int, default=0, help="trials per CSV append (default 100 cpu, 64 gpu)")
    p.add_argument("--only", nargs="*", help="cell names to run (default: all)")
    p.add_argument("--out", default="results/opnav")
    a = p.parse_args()
    batch = a.batch or (64 if a.backend == "gpu" else 100)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"seed": SEED, "distances_km": DISTANCES.tolist(), "phases_deg": PHASES, "read_noise": NOISES, "n_cell": N_CELL,
            "scale": a.scale, "correct_fraction": CORRECT_FRACTION, "min_limb_points": MIN_LIMB_POINTS,
            "refine_iterations": REFINE_ITERATIONS, "edge_margin_px": EDGE_MARGIN_PX, "ellipsoid_km": [R_EQ, R_POL],
            "backend": a.backend, "workers": a.workers, "python": platform.python_version(), "machine": platform.platform()}
    if a.backend == "gpu":
        import cupy as cp
        meta["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    cat = table = pool = None
    if a.workers > 1 and a.backend == "cpu":
        pool = ProcessPoolExecutor(a.workers, initializer=_init_worker)
    else:
        from tracker.catalog import load_catalog
        from tracker.identify import build_pair_table
        cat = load_catalog(5.5)
        table = build_pair_table(cat.r)
    t_all = time.perf_counter()
    try:
        for name, n, distance, phase, noise, true_att in cells(a.scale):
            if a.only and name not in a.only:
                continue
            t0 = time.perf_counter()
            last = [t0]

            def progress(done, total):
                if time.perf_counter() - last[0] > 30 or done == total:
                    last[0] = time.perf_counter()
                    print(f"  {name}: {done}/{total}  ({(last[0] - t0) / max(done, 1) * 1000:.0f} ms/trial)", flush=True)

            ran = run_condition(out / f"{name}.csv", n, SEED, distance, phase, noise, cat, table, backend=a.backend,
                                batch=batch, workers=a.workers, true_attitude=true_att, progress=progress, pool=pool)
            print(f"{name}: {distance:,.0f} km, phase {phase:g}, noise {noise:g}: {n} trials, {ran} run now, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"total {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
