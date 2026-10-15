"""Day 6 benchmark: CPU vs GPU, per stage and end to end (transfers included).

    python -m experiments.benchmark --device cpu      # on the Mac: 1 core and all cores
    python -m experiments.benchmark --device gpu      # on Colab: T4 (+ that machine's CPU for identify)
Writes results/benchmark/<device>_<host>.json; experiments/benchmark_report.py merges them.

Fairness rules (project plan): the CPU baseline is vectorised NumPy (no per-pixel loops), uses its
fastest noise (rng.normal, not the GPU-matching counter noise), and runs on ALL cores with one
process per core. Everything is warmed up before timing; GPU timers stop only after a synchronise.
Frames are the same skies on both devices (campaign seeding).

Stages of one frame:
  prepare   attitude -> project the 2,847 catalog stars -> star list (CPU, both pipelines)
  render    star list -> 1024x1024 noisy image
  detect    image -> star centroids
  identify  centroids -> names -> attitude (vote, check, SVD, verify; CPU float64, both pipelines)
"""
import argparse
import json
import os
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from tracker.camera import project, unproject
from tracker.campaign import trial_attitude
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.identify import build_pair_table, identify
from tracker.render_cpu import BACKGROUND, READ_NOISE, render, star_flux

STAGES = ("prepare", "render", "detect", "identify")
SEED = 7
_W = {}                       # per-process state for pool workers


def host_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cores": os.cpu_count()}
    if platform.system() == "Darwin":
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        perf = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"],
                              capture_output=True, text=True).stdout.split()
        if perf:
            info["cores_detail"] = f"{perf[0]} performance + {perf[1]} efficiency" if len(perf) > 1 else perf[0]
        power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        info["power_source"] = "AC" if "AC Power" in power else "battery"
    else:
        try:
            names = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
            info["cpu"] = names[0] if names else platform.processor()
        except OSError:
            info["cpu"] = platform.processor()
    return info


ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                 "VECLIB_MAXIMUM_THREADS")}


def _pool(workers, seed):
    """One process per core. Thread-pool env vars must be set BEFORE the workers start (spawn copies
    the environment), otherwise each worker's BLAS could start its own threads and oversubscribe."""
    os.environ.update(ONE_THREAD)
    return get_context("spawn").Pool(workers, initializer=_init, initargs=(seed,))


def _init(seed):
    cat = load_catalog(5.5)
    _W.update(cat=cat, table=build_pair_table(cat.r), seed=seed)


def _frame(trial, full):
    """One frame on the CPU, fastest honest way. Returns the number of detections."""
    cat, seed = _W["cat"], _W["seed"]
    A = trial_attitude(seed, trial)
    u, v, vis = project(A, cat.r)
    k = np.flatnonzero(vis)
    img = render(u[k], v[k], star_flux(cat.mag[k]), rng=np.random.default_rng([seed, int(trial), 2]))
    uu, vv, _ = detect(img, BACKGROUND, READ_NOISE)
    if full:
        identify(unproject(uu, vv), _W["table"])
    return len(uu)


def _frame_task(args):
    return _frame(*args)


def _identify_task(uv):
    return identify(unproject(*uv), _W["table"]) is not None


def cpu_stage_times(n_frames=200, seed=SEED):
    """Single core, each stage timed separately for n_frames frames. Returns {stage: [seconds]}."""
    _init(seed)
    cat, table = _W["cat"], _W["table"]
    _frame(10**6, True)                                   # warm-up
    t = {s: [] for s in STAGES}
    for trial in range(n_frames):
        t0 = time.perf_counter()
        A = trial_attitude(seed, trial)
        u, v, vis = project(A, cat.r)
        k = np.flatnonzero(vis)
        uu_, vv_, ff_ = u[k], v[k], star_flux(cat.mag[k])
        t1 = time.perf_counter()
        img = render(uu_, vv_, ff_, rng=np.random.default_rng([seed, trial, 2]))
        t2 = time.perf_counter()
        du, dv, _ = detect(img, BACKGROUND, READ_NOISE)
        t3 = time.perf_counter()
        identify(unproject(du, dv), table)
        t4 = time.perf_counter()
        for s, dt in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            t[s].append(dt)
    return t


def cpu_throughput(n_frames=2000, workers=None, full=True, seed=SEED):
    """All-core throughput: `workers` processes, each running whole frames. full=False stops after
    detect (the GPU's part of the work). Pool start-up and a warm-up pass are excluded."""
    workers = workers or os.cpu_count()
    with _pool(workers, seed) as pool:
        pool.map(_frame_task, [(10**6 + i, full) for i in range(2 * workers)])         # warm-up
        t0 = time.perf_counter()
        pool.map(_frame_task, [(i, full) for i in range(n_frames)], chunksize=max(1, n_frames // (8 * workers)))
        dt = time.perf_counter() - t0
    return {"frames": n_frames, "workers": workers, "full": full, "seconds": dt, "ms_per_frame": 1000 * dt / n_frames}


def gpu_benchmark(n_batches=20, batch=128, seed=SEED, identify_workers=None):
    """GPU pipeline per batch: prepare (CPU) -> pack + upload -> render -> detect -> download star
    lists -> identify (CPU, all cores). Kernel times from CUDA events; wall times after synchronise."""
    import cupy as cp
    from tracker import gpu
    from tracker.batch import pack_frames, project_frames

    _init(seed)
    cat = _W["cat"]
    identify_workers = identify_workers or os.cpu_count()
    ev = [cp.cuda.Event() for _ in range(3)]
    rows = []
    with _pool(identify_workers, seed) as pool:
        for b in range(-2, n_batches):                           # two warm-up batches (kernel compile, caches)
            trials = np.arange(max(b, 0) * batch, max(b, 0) * batch + batch)
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            frames, _ = project_frames([trial_attitude(seed, t) for t in trials], cat)
            t1 = time.perf_counter()
            sb = pack_frames(frames)
            t2 = time.perf_counter()
            ev[0].record()
            imgs = gpu.render_batch(sb, seed=seed, frame_ids=trials)       # includes star-list upload
            ev[1].record()
            found = gpu.detect_batch(imgs, BACKGROUND, READ_NOISE)         # includes download + host sort
            ev[2].record()
            ev[2].synchronize()
            t3 = time.perf_counter()
            ok = pool.map(_identify_task, [(u, v) for u, v, _ in found], chunksize=max(1, batch // (4 * identify_workers)))
            t4 = time.perf_counter()
            if b >= 0:
                rows.append({"prepare": t1 - t0, "pack": t2 - t1,
                             "render_kernel": cp.cuda.get_elapsed_time(ev[0], ev[1]) / 1000,
                             "detect_and_download": cp.cuda.get_elapsed_time(ev[1], ev[2]) / 1000,
                             "gpu_wall": t3 - t2, "identify_pool": t4 - t3, "total": t4 - t0,
                             "solved": int(sum(ok))})
            del imgs
    per_frame = {k: 1000 * float(np.median([r[k] for r in rows])) / batch for k in rows[0] if k != "solved"}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,memory.total,driver_version,clocks.max.sm",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"batch": batch, "batches": n_batches, "identify_workers": identify_workers,
            "ms_per_frame_median": per_frame, "solved": int(sum(r["solved"] for r in rows)),
            "frames": n_batches * batch, "nvidia_smi": smi}


def summarise_stage_times(t):
    return {s: {"median_ms": 1000 * float(np.median(v)), "mean_ms": 1000 * float(np.mean(v)),
                "p99_ms": 1000 * float(np.percentile(v, 99))} for s, v in t.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "gpu"), required=True)
    p.add_argument("--frames", type=int, default=2000, help="frames for CPU throughput runs")
    p.add_argument("--stage-frames", type=int, default=300, help="frames for single-core stage timing")
    p.add_argument("--batches", type=int, default=20, help="GPU batches of 128")
    p.add_argument("--out", default="results/benchmark")
    a = p.parse_args()

    res = {"host": host_info(), "device": a.device, "seed": SEED}
    print(json.dumps(res["host"]))
    res["single_core_stages"] = summarise_stage_times(cpu_stage_times(a.stage_frames))
    for s, v in res["single_core_stages"].items():
        print(f"  1 core  {s:9s} median {v['median_ms']:7.3f} ms   p99 {v['p99_ms']:7.3f} ms")
    if a.device == "cpu":
        for full in (False, True):
            for workers in (1, os.cpu_count()):
                r = cpu_throughput(a.frames if workers > 1 else a.frames // 4, workers, full)
                res.setdefault("throughput", []).append(r)
                print(f"  {'full pipeline' if full else 'render+detect':13s} {workers:2d} processes: "
                      f"{r['ms_per_frame']:7.3f} ms/frame ({1000 / r['ms_per_frame']:6.0f} frames/s)")
    else:
        res["gpu"] = gpu_benchmark(a.batches)
        res["host"]["gpu"] = res["gpu"]["nvidia_smi"]
        for k, v in res["gpu"]["ms_per_frame_median"].items():
            print(f"  GPU pipeline {k:20s} {v:7.3f} ms/frame")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.device}_{platform.node().split('.')[0] or 'host'}.json"
    (out / name).write_text(json.dumps(res, indent=1))
    print(f"saved {out / name}")


if __name__ == "__main__":
    main()
