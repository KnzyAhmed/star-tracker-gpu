# BUILD_LOG.md

Dated record of what was built, what was decided (and rejected), measured numbers, and problems.

## Before 2026-10-11: Days 0–1 (built by the owner; summary from git history)
- Day 0: project skeleton, `data/bsc5.csv` (Yale BSC, 9,096 stars).
- Day 1a: camera model (`tracker/camera.py`).
- Day 1b: CPU renderer, first rendered sky (Orion verified).
- State on 2026-10-11 before Day 2 work: `pytest -q` → **25 passed**.

## 2026-10-11: Day 2, picture back to answer

### `tracker/detect_cpu.py`: `detect`
- **Built.** Threshold (`> background + k·read_noise`), then a 3×3 local-max test with a raster-order
  tie-break (`>` against the 4 earlier neighbours, `>=` against the 4 later ones), then a 5×5
  background-subtracted weighted-mean centroid. Output sorted brightest first.
- **Decisions.**
  - The local-max test runs only on pixels that passed the threshold, not on the full image (8 full-image
    shifted comparisons would be wasted work). Fully vectorised, no per-pixel Python loops.
  - Windows are padded with the background value. Edge stars are handled without special cases.
  - Negative weights are kept (no clipping). Clipping biases centroids toward the window centre.
  - Stars whose window flux ≤ 0 are dropped (avoids divide-by-zero; practically never happens above threshold).
  - Output sorted by flux. This is not in the spec, but Day 3 identification wants brightest stars first, and
    Day 4 GPU output (atomicAdd order is arbitrary) needs a canonical order to compare against.
- **Rejected.** A 7×7 window: noise-free bias drops from 0.100 to 0.017 px, but the noise error for faint
  stars roughly doubles (Σx² 50 → 196). Clipping negative weights. Full-image local-max.
- **Measured.**
  - Noise-free pixel locking of the 5×5 window: bias up to −0.100 px per axis at 0.5 px offset. The test tolerance
    of 0.12 px was set from this calculation, not by trial and error.
  - Empty noisy frames: 0.2 false detections per frame (20 seeds). Theory: 2.9e-7 × 1,048,576 ≈ 0.3.
  - Orion (RA 83.8, Dec −1.0, seed 1): 55 true stars, 54 detected, 53 matched within 1.5 px,
    **median centroid error 0.101 px**.
  - Speed: 4.9 ms per 1024×1024 frame (laptop CPU, single call, averaged over 50).
- **Tests.** `tests/test_detect.py`: 22 new tests. Suite: **47 passed**.

### `tracker/solve.py`: `solve_attitude` (2026-10-14)
- **Built.** SVD solution of Wahba's problem: `B = Σ wᵢ bᵢ rᵢᵀ`, `A = U diag(1,1,det U det V) Vᵀ`, all float64.
- **Decisions.** `B` is formed as one matrix product `(w b)ᵀ r` (no Python loop over stars). `w=None` → equal weights.
  No input normalisation or validation; callers pass unit vectors (`unproject`, `catalog.r`).
- **Rejected.** Linear least squares on A's 9 entries (not a rotation). TRIAD (uses only 2 stars).
  q-method/QUEST (same optimum; SVD is simpler to explain and more robust).
- **Measured.** Mirror-image input really triggers `det U det V = −1` (checked, so the test is not vacuous).
  30 random-direction stars with 10″ per-axis noise: median total error 3.6″ (50 trials). Theory: 3.4″.
- **Tests.** `tests/test_solve.py`: 7 new tests (exact recovery, 2 stars, rotation under noise, no reflection,
  zero weight drops an outlier, weight scale invariance, noise averaging). Suite: **54 passed**.

### `tracker/solve.py`: `attitude_error` (2026-10-14)
- **Built.** `(total, pointing, roll)` in arcsec from dA = A_est A_trueᵀ. Total = atan2 form of the rotation angle;
  pointing = atan2(|z×z'|, z·z'); roll = twist about camera z, atan2(dA₁₀−dA₀₁, dA₀₀+dA₁₁) (exact, = 2 atan2(q_z, q_w)).
- **Decisions.** All three are returned as magnitudes ≥ 0 (sign of roll dropped; can add a signed variant if Day 5 needs bias plots).
  The roll definition is swing-twist, so it is exact at any size, not just a small-angle approximation.
- **Rejected.** Clipped arccos of (trace−1)/2: measured 0 instead of 1 mas, 6 % off at 10 mas.
  SciPy `as_rotvec` (works, but hides the maths the owner must explain).
- **Tests.** 7 new tests (no NaN at identity, pure roll, pure pointing, symmetry, 1 mas precision, 180° cases,
  quadrature total² = pointing² + roll²). Suite: **61 passed**.

### Day 2 gate (2026-10-14): `experiments/day2_gate.py`, `test_orion_true_identities_gives_a_few_arcsec`
- **Built.** Experiment script: N random attitudes → render (noise on) → detect → nearest true star within 1.5 px →
  SVD solve with true identities → error. Writes `results/day2_gate.{txt,csv,png}` (results/ is gitignored).
  Also a single-field integration test (Orion, seed 1): 53 matched stars, **pointing 1.08″, roll 4.22″, total 4.36″**.
- **Measured (1,000 trials, seed 2, 13.7 ms/frame for render+detect+solve on the laptop CPU, single process):**
  - Stars in view mean 27.2 (min 9, max 87). Matched 26,803 / 27,211 = 98.5 %. 287 unmatched detections (~0.3/frame,
    consistent with the 5σ noise false-alarm rate measured earlier).
  - Centroid error: median **0.103 px**, mean 0.119, p90 0.209, p99 0.383, p99.9 0.862 px. Median 7.3″.
  - By magnitude: 0.05–0.08 px for mag < 4 (bias-limited), 0.095 at mag 4–5, 0.133 at mag 5–5.5 (noise-limited).
  - Pair-angle |error|: median 6.1″, p99 30.9″, p99.9 60.2″, max 95.5″.
  - True-identity attitude: solved 1000/1000. Pointing median **1.67″** (p99 4.8″); roll median **6.80″** (p99 27.9″).
- **GATE: PASS.** Median centroid error 0.103 < 0.15 px; true-identity pointing ≈ 1.7″ ("a few arcsec").
- **For Day 3.** eps should come from the pair-angle distribution above. Proposal: ~60″ (p99.9), to be confirmed on Day 3.
- **Tests.** Suite: **62 passed**.

## 2026-10-14: Day 3, name the dots

Note on eps (from review): the 1 % miss rate at eps = 31″ applies to a photo's ~350 pairs (≈ 3–4 lost pairs/frame),
not the catalog's 266k. Each star has ~26 pairs voting for it, so the trade-off is close → decide by measurement.

### `tracker/identify.py`: `build_pair_table`, `angle_between`, `PairTable`
- **Built.** k-d tree `query_pairs` with chord radius 2 sin(θ/2), then atan2 angles, stable sort by angle.
- **Measured.** 266,153 pairs for 2,847 stars (mag < 5.5), built in 8 ms. Exact sensor corner-to-corner angle is 28.00°,
  so the spec's 28.3° covers it.
- **Rejected.** Full N×N angle matrix (O(N²) memory). Cutting at 20° (FOV edge-to-edge): it would miss corner pairs.
- **Tests.** `tests/test_identify.py`: 6 tests (sorted, angles correct, i<j unique, complete vs brute force,
  covers diagonal, real-catalog size). Suite: **68 passed**.

### `tracker/identify.py`: `vote`, `ARCSEC`
- **Built.** All detected pairs → atan2 angles → `searchsorted` window [θ−eps, θ+eps] → flat expansion of table rows →
  4 votes per row (a→p, a→q, c→p, c→q) → one `bincount` over (N × n_cat) → argmax per dot.
- **Problem 1: a noise-free test failed after a change.** Cause: β Mon A/B (catalog idx 750/751, HR 2356/2357), 9.6″ apart, tie exactly
  on votes. My test assumed exact identity, which is impossible for doubles closer than eps (36 such catalog pairs < 60″).
  Fixed **my new (uncommitted) test** to accept the true star or one within eps of it. This is a correction of a wrong expectation,
  not a weakening: either choice changes the attitude by < eps.
- **Problem 2: the fake-star test's margin.** I had asserted "fake < half of weakest real" without measuring. Measured: 14 vs 21.
  Changed the test to assert the property that matters (fake < every real star), with the measured rate in the docstring.
- **Decision: plain voting, no per-pair dedupe** (300 random fields, 7″ noise, eps 60″):

| variant | median ms | p99 ms | max ms | misnamed stars |
|---|---|---|---|---|
| plain, all stars | 2.12 | 9.9 | 19.6 | 0.11 % |
| plain, brightest 20 | 1.36 | 1.8 | 2.0 | 0.16 % |
| dedupe, all stars | 30.4 | 310 | 768 | 0 % |
| dedupe, brightest 20 | 18.7 | 22.9 | 23.5 | 0 % |

  Dedupe is ~15× slower; verification will handle the rare misnamed star. A brightest-20 cap would bound the worst case at 2 ms,
  but it is not needed yet (spec: every pair votes). Kept as an option if Day 3 timing shows a problem.
- **Tests.** 5 vote tests. Suite: **73 passed**.

### `tracker/identify.py`: `identify`, `consistent_names`, `verify`, `Solution`; `PairTable` gains `r`, `tree`
- **Built.** vote → pairwise consistency check on the names → SVD → verify (k-d tree, ≥ 4 within eps) → refine → verify.
  `PairTable` now also keeps the catalog vectors and the k-d tree it was built with (the tree was already being built there).
- **Decision: the consistency check (not in the spec) is essential.** Measured, eps 60″, real rendered frames:
  2 hot pixels per frame: with check 500/500 correct, without 0/500 (all "no solution"). Clean frames: 300/300 vs 218/300.
  Cause: every fake/blended dot gets *some* top-voted name; one wrong name in the SVD shifts the attitude past eps.
- **Rejected.** Weighting the solve by vote count (a wrong name still has non-zero weight). Repeatedly dropping the worst residual
  (more iterations; the pair check removes the outliers before the first solve).
- **Tests.** 6 identify tests (noise-free, 7″ noise, 3 fake dots, random dots → None ×20, too few stars, Orion image end to end
  with Betelgeuse/Rigel named). Suite: **79 passed**.

### `tracker/stats.py`: `wilson_interval`
- **Built.** Wilson score interval (z = 1.96), vectorised. Needed because every rate is reported with a 95 % error bar.
- **Rejected.** Normal-approximation (Wald) interval: zero width at k = 0 or k = n, and leaves [0, 1].
- **Tests.** `tests/test_stats.py`: 5 tests (textbook value 81/263, k = n, k = 0, shrinks with n, simulated coverage ≈ 95 %).
  Suite: **84 passed**.

### `experiments/montecarlo.py`: Day 3 gate (2026-10-14)
- **Built.** 1,000 random attitudes (seed 3). Each frame is rendered and detected once, then identified at eps 31″, 45″, 60″
  (same images for all three). Outcomes: correct (total < 300″), wrong, no solution, each with a 95 % Wilson interval.
  Per-stage timing. One CSV row per trial in `results/montecarlo_cpu.csv` (gitignored).
- **Measured.** Stars in view: mean 26.8, min 7, max 83.

| eps | correct | wrong | no solution | pointing median / p99 | roll median / p99 | identify median / p99 |
|---|---|---|---|---|---|---|
| 31″ | 1000/1000 [99.62, 100] % | 0 [0, 0.38] % | 0 [0, 0.38] % | 1.65″ / 4.53″ | 6.83″ / 26.5″ | 1.59 / 9.4 ms |
| 45″ | 1000/1000 [99.62, 100] % | 0 | 0 | 1.68″ / 4.84″ | 6.96″ / 28.8″ | 1.80 / 13.0 ms |
| 60″ | 1000/1000 [99.62, 100] % | 0 | 0 | 1.69″ / 4.91″ | 7.01″ / 28.9″ | 2.19 / 16.2 ms |

  Worst correct frame: 41.6″ total, so the 300″ threshold sits in a wide gap (no wrong frames to bound it from above).
- **Per-stage time, one CPU core, eps 60″:** render 8.29 ms (50.4 %), detect 5.05 ms (30.8 %), identify 2.19 ms median /
  3.11 mean (18.8 %, of which the SVD is 0.021 ms). Total 15.7 ms median, 30.0 ms p99.
  Pair table: 55 ms here vs 8 ms earlier (first call, cold caches); it is built once.
- **GATE: PASS.** Success 100 % (≥ 99.62 % at 95 %), well above 90 %. Render + detect = **81 %** of the time.
- **eps not decided.** On clean images all three are perfect, so this run cannot separate them. Smaller eps is slightly
  faster and slightly more accurate (fewer slightly-off stars accepted in verify). The difference should only appear
  under stress (more noise, hot pixels), which is a Day 5 experiment, or a quick stress run now. Owner to decide.

### eps stress test (2026-10-14): `add_hot_pixels`, `--read-noise` / `--hot-pixels` in montecarlo.py
- **Built.** `tracker/render_cpu.add_hot_pixels(image, n, rng, lo=100, hi=2000)`: n distinct pixels get +U(lo, hi) counts
  (test added first; suite **85 passed**). montecarlo.py gained `--read-noise` (renderer AND detector threshold) and `--hot-pixels`.
  Hot pixels are only drawn when n > 0, so the baseline run's random stream is unchanged.
- **Measured: correct rate, 1,000 frames per condition, same images for every eps** (wrong = 0 in EVERY run below,
  upper bound 0.38 %):

| condition | 15″ | 20″ | 25″ | 31″ | 45″ | 60″ |
|---|---|---|---|---|---|---|
| noise 5 (baseline) | 100 | 100 | 100 | 100 | 100 | 100 |
| noise 10 | | | | 100 | 99.9 | 99.9 |
| noise 15 | | | | 98.1 | 96.6 | 92.3 |
| noise 20 | 81.4 | 85.5 | **86.0** | 84.7 | 80.8 | 71.4 |
| noise 5 + 20 hot pixels | 100 | 100 | 100 | 99.9 | 97.5 | 89.1 |
| noise 15 + 10 hot pixels | 88.7 | **90.6** | 90.3 | 86.3 | 75.8 | 61.2 |

  95 % intervals ±~2 points at 85–90 %. E.g. noise 15 + hot 10: 20″ [88.6, 92.3], 25″ [88.3, 92.0], 31″ [84.0, 88.3], 60″ [58.1, 64.2].
  Accuracy of correct frames is the same for all eps within noise (e.g. noise 15+hot 10 pointing median 3.25–3.35″).
- **My earlier prediction was wrong.** I expected a wide eps to be safer. Under stress, the wide eps fails most.
- **Diagnosis** (300 frames each): failures happen in the VOTE. Noise 5 + 20 hot, eps 60″: only 83.9 % of real dots get the right
  top vote (98.9 % at 31″); in the failed frames only 8 % do. Mechanism: true-star votes ≈ (real dots − 1), independent of eps;
  coincidence votes ∝ eps × total dots. Few real stars (noise hides faint ones) or many fakes → the luckiest wrong star wins.
- "Too tight" begins below 20″: at noise 20, 15″ drops to 81.4 % (true pairs fall outside the window as centroid errors grow).
- **Verification held everywhere:** 0 wrong answers in 15,000 identify calls across all stressed runs. Failures are always "no solution".
- **Recommendation: eps = 25″** (ties 20″ within error bars; ≥ 20″ is the flat top; 25″ keeps more margin from the 15″ cliff and
  verifies slightly more stars). Owner to decide; nothing in the code defaults to an eps yet.
- Idea for later (not built): scale eps with the noise level, since the best eps is "a few × the pair-angle error".

### eps decided (2026-10-14): **25″** (owner approved)
- `tracker/identify.EPS_ARCSEC = 25.0`, now the default of `identify()` and of `montecarlo.py --eps`.
- The identify tests now run at the default 25″ instead of 60″ (stricter; all still pass). The vote tests stay at 60″,
  where they document voting behaviour. Suite: **86 passed**.

## 2026-10-14: Day 4, GPU

### `tracker/noise.py`: `hash32`, `pixel_normals`, `MAX_SIGMA`
- **Built.** lowbias32 integer hash (uint32, wraps mod 2³²) → key from (seed + golden ratio, frame) → two 24-bit uniforms per pixel
  → Box-Muller (cos branch), all float32. Pixel counter = v·1024 + u.
- **Decisions.** 24-bit uniforms (exact float32 conversion on both machines). u1 offset by +1 (never log 0; caps |z| at 5.77).
  Golden-ratio offset on the seed, because hash32(0) = 0. Checked: without it, pixel (0,0) of seed 0 / frame 0 is fixed at −4.84σ.
- **Measured.** Statistics OK; false alarms 0.25/frame (theory 0.30). CPU speed 16.0 ms/frame vs `rng.normal` 6.9 ms, so the CPU
  benchmark (Day 6) keeps `rng.normal`.
- **Rejected.** Philox (heavier to hand-write identically in NumPy and CUDA). Using both Box-Muller outputs (it would need pixel pairs, which
  complicates the one-thread-per-pixel mapping).
- **Open question for the owner (blocks render.cu):** the spec caps a frame's star list at 64. Measured: 13/1000 frames have > 64 stars
  in view (max 83). See the chat for options.
- **Tests.** `tests/test_noise.py`: 5 tests. Suite: **91 passed**.

### Decision: > 64 stars per frame (2026-10-14)
- Asked the owner; no answer given ("continue"), so I took my recommended option: **render.cu processes the star list in chunks
  of 64** through shared memory. This keeps the spec's 64-star tile and supports any count (measured max 83). Easy to revisit.

### `tracker/render_cpu.py`: `star_windows`, `render(noise_key=...)`
- **Built.** Window bounds factored out (int32, inclusive, clipped); the same arrays are given to the GPU. `noise_key=(seed, frame)`
  adds `read_noise * pixel_normals(seed, frame)`. `rng` and `noise_key` together → ValueError.
- **Measured.** Bit-identical output to the previous renderer on 30 random skies; day2_gate (50 trials) unchanged in character.
- **Tests.** 2 new render tests. Suite: **93 passed**.

### `tracker/batch.py`: `StarBatch`, `pack_frames`, `project_frames`
- **Built.** CSR packing of per-frame star lists: offsets + per-star du, dv, amp (float32) and box (int32 = CPU windows).
- **Problem caught before it happened.** float32 u near 1000 is off by 1.2e-5 px, which changes the edge pixels by ~1e-4 relative and would break
  CPU/GPU equality. Fix: send positions relative to the box corner (error 1.9e-7 px). Amplitude flux/(2πσ²) is precomputed in float64.
- **Tests.** `tests/test_batch.py`: 5 tests. Suite: **98 passed**.

## 2026-10-15: Day 4 continued, GPU kernels and the gate

### `tracker/kernels/render.cu`, `gpu.render_batch`
- **Built.** One thread per pixel per frame, block 16×16, grid 64×64×F. Star list staged in shared memory in chunks of 64.
  Box test with the host-computed integer boxes; dx from box-relative positions; counter-based noise identical to `tracker/noise.py`.
  Output indexed with a 64-bit offset.
- **Problem.** `size_t` is undefined without system headers (caught by the new clang check) → `long long`.

### `tracker/kernels/detect.cu`, `gpu.detect_batch`, `detect_cpu` threshold made explicitly float32
- **Built.** 20×20 shared tile (16 + 2-px halo), OFF_SENSOR = −FLT_MAX for off-sensor halo pixels, raster tie-break, 5×5 centroid,
  atomicAdd slots, output = peak (exact ints) + offsets + flux. Host sorts by (−flux, v, u).
- **Deviation from spec: capacity 256, not 64.** Frames reach 83 stars (more with hot pixels). Overflow raises RuntimeError rather than
  dropping a thread-timing-dependent subset.
- **Problem: a threshold test that tested nothing, twice.** (1) read_noise 5.1 gives a threshold float32 represents exactly → switched to 5.13,
  which rounds up. (2) My premise was wrong: I assumed the CPU compares in float64. NumPy 2 (2.5.4) compares a float32 array with a Python float
  *in float32*, but with an np.float64 it uses float64. So `detect` depended on its input *types*. Fixed by `threshold = np.float32(...)`
  (no change for current callers) and a CPU test for both input types. The GPU uses the same float32 value.

### Tooling: `tools/cuda_shim.h`, `tests/test_kernels_compile.py`
- clang (Apple 21) syntax/type-checks the `.cu` files on the Mac, with `-Wall -Werror` and a tiny shim of CUDA keywords. It skips where clang can't
  target CUDA (e.g. Colab). It caught the `size_t` error before any GPU run.

### Running on a GPU (Colab)
- The first attempt ran in a Chrome profile signed in to someone else's Google account. That notebook was moved to the bin and its runtime deleted
  at the owner's request. (Its single run also showed 103 passed on a T4; not counted.)
- On the owner's account (knzyahmed@aucegypt.edu), Drive is **full**, so new notebooks can't be created. Used Colab's built-in example notebook in
  **playground mode** (nothing saved to Drive), uploaded a `git archive` zip via the Files panel (the repo is private), ran, then deleted the runtime.
- `notebooks/colab_run.ipynb` added so the owner can repeat this.

### Day 4 gate (2026-10-15), Tesla T4, 70 W power limit, 15,360 MiB, driver 580.82.07, CuPy 14.0.1
- `pytest -q tests/`: **109 passed, 3 skipped** (the 3 skips are the clang checks). All 10 GPU tests passed.
- `experiments/gpu_check.py` (commit a557d68):
  - render noise off: max |GPU−CPU| **4.88e-4 counts** (= 1 float32 ulp in 4096–8192), max relative 5.15e-7, 0.06 % of pixels differ.
  - render counter noise: max |GPU−CPU| 4.88e-4 counts; max relative 1.87e-3 (only near-zero pixels); 10.3 % of pixels differ by an ulp (log/cos).
  - detect: same star count in **16/16** frames, max centroid difference **3.2e-7 px**.
  - rough timing, 128-frame batch, median of 5 after warm-up: render **0.243 ms/frame**; render + detect + star-list download
    **0.318 ms/frame** (CPU single core: 8.3 + 5.1 = 13.4 ms). Not the Day 6 benchmark (no multi-core CPU baseline yet).
- **GATE: PASS.** Noise off → GPU image equals CPU to float32 tolerance; same stars detected.

### Decision (2026-10-15): add a public demo website as Day 7
- Owner's idea, discussed first. Options: (A) replay of precomputed skies, (B) the real Python tracker running live in the browser
  via Pyodide, (C) a Streamlit-style app with a server. **Chose B** for the live part; the Monte Carlo/benchmark charts are stored
  results on a separate tab. GitHub Pages, public. Scheduled after Days 5–6 so the results tab has real numbers. Spec added to the project plan.
- Rejected: C (needs a running server). GPU live on the web (no GPUs on free static hosting; GPU numbers come from the Colab runs).
