# Day 6 benchmark: CPU vs GPU

- CPU: **Apple M1 Pro (8 performance + 2 efficiency, AC power)**, NumPy, one process per core, fastest CPU noise (`rng.normal`).
- GPU: **Tesla T4, 70.00 W, 15360 MiB, 580.82.07, 1590 MHz** (name, power limit, memory, driver, max SM clock), CuPy RawKernels, batches of 128, 2 warm-up batches, timers stopped after synchronise.
- Colab host CPU: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 vCPU (Colab).
- Same skies on both (campaign seeding). Medians, ms per frame (lower is better).

| configuration | render + detect | full pipeline | frames/s (full) | note |
|---|---|---|---|---|
| Apple M1 Pro, 1 core | 11.968 | 13.351 | 75 | measured |
| Apple M1 Pro, 10 processes | 1.854 | 1.999 | 500 | measured |
| Colab Xeon 2.0 GHz, 1 core | 29.029 | 32.085 | 31 | measured (sum of stage medians) |
| Tesla T4 + Colab 2 vCPU for identify | 0.323 | 3.028 | 330 | measured, transfers included |
| Tesla T4 + Apple M1 Pro 10 cores for identify | 0.323 | 0.673 | 1,485 | PROJECTION (not run) |

**Speed-ups for the image stages (render + detect, transfers included):**
- T4 vs Apple M1 Pro 1 core: **37×**
- T4 vs Apple M1 Pro all 10 cores: **5.7×**
- T4 vs its own Colab CPU (1 core): 90×

**GPU pipeline breakdown (ms/frame):** prepare 0.138, pack 0.004, render_kernel 0.242, detect_and_download 0.075, gpu_wall 0.319, identify_pool 2.535, total 3.028

Identification is 84 % of the GPU pipeline's time on Colab's 2 vCPUs: after the GPU removes the image work, the CPU-side identify is the bottleneck (Amdahl's law).
The projection row assumes identify scales over the Mac's cores like the full CPU pipeline did (6.7× on 10 processes); it was not run as one system.
