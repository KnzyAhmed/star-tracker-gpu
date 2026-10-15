"""Day 4 gate numbers on a GPU machine: how close CPU and GPU are, and a first timing.

    python -m experiments.gpu_check          # needs CuPy + an NVIDIA GPU (e.g. Colab T4)

Not the Day 6 benchmark (that one compares against a multi-core CPU baseline); this only
reports agreement and a rough per-frame GPU time for a 128-frame batch.
"""
import subprocess
import time

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

from tracker import gpu
from tracker.batch import pack_frames, project_frames
from tracker.camera import random_attitude
from tracker.catalog import load_catalog
from tracker.detect_cpu import detect
from tracker.render_cpu import BACKGROUND, READ_NOISE, render

BATCH = 128


def main():
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    cat = load_catalog(5.5)
    rng = np.random.default_rng(0)

    # --- agreement on 16 frames
    frames, _ = project_frames([random_attitude(rng) for _ in range(16)], cat)
    batch = pack_frames(frames)
    ids = np.arange(len(frames))
    for label, kw_gpu, kw_cpu in [("noise off", dict(read_noise=0.0), lambda k: dict(read_noise=0.0)),
                                  ("counter noise", dict(seed=3, frame_ids=ids), lambda k: dict(noise_key=(3, k)))]:
        g = cp.asnumpy(gpu.render_batch(batch, **kw_gpu)).astype(np.float64)
        c = np.stack([render(u, v, f, **kw_cpu(k)) for k, (u, v, f) in enumerate(frames)]).astype(np.float64)
        d = np.abs(g - c)
        print(f"render {label:13s}: max |GPU-CPU| {d.max():.2e} counts, max relative {(d / np.abs(c)).max():.2e}, "
              f"pixels differing at all {np.mean(d > 0) * 100:.2f} %")
    g_imgs = gpu.render_batch(batch, seed=3, frame_ids=ids)
    found = gpu.detect_batch(g_imgs, BACKGROUND, READ_NOISE)
    n_same = dmax = 0
    for k, (u, v, f) in enumerate(frames):
        uc, vc, fc = detect(render(u, v, f, noise_key=(3, k)), BACKGROUND, READ_NOISE)
        ug, vg, fg = found[k]
        n_same += len(ug) == len(uc)
        if len(uc):
            dist = cKDTree(np.column_stack([uc, vc])).query(np.column_stack([ug, vg]))[0]
            dmax = max(dmax, dist.max())
    print(f"detect: same star count in {n_same}/{len(frames)} frames, max centroid difference {dmax:.2e} px")

    # --- rough timing: one 128-frame batch, after a warm-up (compiles the kernels)
    frames, _ = project_frames([random_attitude(rng) for _ in range(BATCH)], cat)
    batch = pack_frames(frames)
    ids = np.arange(BATCH)
    gpu.detect_batch(gpu.render_batch(batch, seed=1, frame_ids=ids), BACKGROUND, READ_NOISE)   # warm-up
    cp.cuda.Device().synchronize()
    start, mid, end = cp.cuda.Event(), cp.cuda.Event(), cp.cuda.Event()
    t_render, t_total = [], []
    for rep in range(5):
        t0 = time.perf_counter()
        start.record()
        imgs = gpu.render_batch(batch, seed=1, frame_ids=ids)
        mid.record()
        gpu.detect_batch(imgs, BACKGROUND, READ_NOISE)            # includes copying star lists back
        end.record()
        end.synchronize()
        t_total.append(time.perf_counter() - t0)
        t_render.append(cp.cuda.get_elapsed_time(start, mid) / 1000)
    r, t = np.median(t_render), np.median(t_total)
    print(f"timing, {BATCH}-frame batch (median of 5): render {1000 * r / BATCH:.3f} ms/frame, "
          f"render+detect+star-list download {1000 * t / BATCH:.3f} ms/frame "
          f"(CPU single core, Day 3: render 8.3 + detect 5.1 ms/frame)")


if __name__ == "__main__":
    main()
