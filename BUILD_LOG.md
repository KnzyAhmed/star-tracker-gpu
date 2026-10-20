# BUILD_LOG.md

Dated record of what was built, what was decided (and rejected), measured numbers, and problems.

## Before 2026-10-11: Days 0–1 (summary from git history)
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
  SciPy `as_rotvec` (works, but hides the maths I need to be able to explain).
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
  under stress (more noise, hot pixels), which is a Day 5 experiment, or a quick stress run now. I decided to run the stress test first.

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
  verifies slightly more stars). Decision pending; nothing in the code defaults to an eps yet.
- Idea for later (not built): scale eps with the noise level, since the best eps is "a few × the pair-angle error".

### eps decided (2026-10-14): **25″**
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
- **Open question (blocks render.cu):** the spec caps a frame's star list at 64. Measured: 13/1000 frames have > 64 stars
  in view (max 83). See the chat for options.
- **Tests.** `tests/test_noise.py`: 5 tests. Suite: **91 passed**.

### Decision: > 64 stars per frame (2026-10-14)
- I went with the most flexible option: **render.cu processes the star list in chunks
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
- A first attempt accidentally ran under the wrong Google account; that notebook was deleted and its run not counted
  (it also showed 103 passed on a T4).
- My university Drive is **full**, so new notebooks can't be created there. I used Colab's built-in example notebook in
  **playground mode** (nothing saved to Drive), uploaded a `git archive` zip via the Files panel (the repo was private then), ran, then deleted the runtime.
- `notebooks/colab_run.ipynb` added so the GPU runs can be repeated.

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
- Options considered: (A) replay of precomputed skies, (B) the real Python tracker running live in the browser
  via Pyodide, (C) a Streamlit-style app with a server. **Chose B** for the live part; the Monte Carlo/benchmark charts are stored
  results on a separate tab. GitHub Pages, public. Scheduled after Days 5–6 so the results tab has real numbers. Spec added to the project plan.
- Rejected: C (needs a running server). GPU live on the web (no GPUs on free static hosting; GPU numbers come from the Colab runs).

## 2026-10-15: Day 5, campaign

- Colab GPU runs now use my personal Google account, whose Drive is not full, so a normal notebook can be saved.

### `tracker/camera.py`: `pointing_from_attitude` (new; existing functions unchanged)
- **Built.** Boresight → RA (mod 360), Dec (atan2), roll from the camera x-axis against attitude_from_pointing(ra, dec, 0).
- **Problem.** `(-1e-17) % 360 == 360.0` in float64 → mapped to 0 (test added).
- **Tests.** 3 new (round trip from angles, rebuild of 200 random attitudes + a pole case, RA never 360). Suite: **105 passed**.

### `tracker/evaluate.py`: `score`, `CORRECT/WRONG/NO_SOLUTION`, `CORRECT_ARCSEC = 300`
- **Built.** One shared definition of a trial's outcome (moved out of montecarlo.py so the campaign can't use a different rule).
- **Check.** montecarlo.py refactored to use it: 100-trial run gives identical outcomes/errors before and after.
- **Tests.** `tests/test_evaluate.py`: 3 tests (codes, no solution → NaN errors, threshold split ±1″). Suite: **108 passed**.

### Hot pixels on the GPU: `render_cpu.draw_hot_pixels`, `gpu.add_hot_pixels_batch`
- Host draws (flat indices, float32 values) once; CPU and GPU both add them → identical images. `add_hot_pixels` uses the same draw
  order as before, so Day 3 stress results still reproduce. Tests: 1 CPU, 1 GPU.

### `tracker/campaign.py` + `experiments/campaign.py`
- **Built.** `trial_attitude`, `trial_hot_pixels`, `run_batch` (cpu|gpu backend; identify + score always on CPU float64),
  `run_condition` (CSV per condition, append per batch, resume by skipping finished trials). `experiments/campaign.py` defines the
  conditions: baseline 100k, noise 2.5–20 ×10k, hot pixels 5–80 ×10k (seed 2026, eps 25″), writes `results/campaign/*.csv` + meta.json.
- **Decisions.** Per-trial seeding (resume-safe, batch-independent, CPU == GPU). Common random numbers across conditions (paired sweeps).
  CPU backend uses the counter noise (slow but identical to GPU) because it is for tests and smoke runs, not the benchmark.
- **Measured.** CPU smoke run (scale 0.002: 200 baseline + 12×20): 12.6 s, ~25 ms/trial; all files written.
- **Tests.** `tests/test_campaign.py` 5 (row contents, batch independence, shared skies, hot pixels add 8–10 detections, resume);
  `test_gpu.py` +1 (CPU rows == GPU rows). Suite (Mac): **114 passed**.

### `tracker/stats.outcome_rates` + `experiments/analyze_campaign.py`
- **Built.** Per-group counts of correct/wrong/no solution with Wilson intervals (tested, 2 tests). Analysis script → `analysis/report.txt`,
  `summary.json` (for the Day 7 website), figures: read-noise sweep, hot-pixel sweep (each: success with 95 % CI + detections,
  two panels, no dual axis), success vs stars in view, Mollweide failure map over star density, pointing/roll histograms.
  Colours from the dataviz reference palette (blue/orange series, red = failures).
- **Problems.** (1) matplotlib rejects a −1e−14 error bar at exactly 100 % → clipped at 0. (2) Overlapping "n=" labels and a legend
  over the colour-bar label → moved n into the title, legend below.
- **Sanity check.** The smoke run showed noise 20 at 20/20 correct (Day 3: 86 %). Detections did fall (27.5 → 11.6), so noise was applied;
  200 trials gave **172/200 = 86.0 % [80.5, 90.1]**, 0 wrong, the same as Day 3. The first 20 were just all solvable.
- Suite (Mac): **116 passed**.

### Day 5 campaign run (2026-10-15), Colab Tesla T4 (70 W), Python 3.13.15, private account
- Mac: smoke + 1 % pilot (2,200 frames on the GPU: 3 ms/trial end to end). GPU test suite on Colab: **125 passed, 3 skipped**
  (skips = Mac-only clang checks; includes CPU rows == GPU rows and GPU hot pixels == CPU).
- Full campaign: **220,000 frames in 19 min 18 s** (baseline 100k in 335 s ≈ 3.4 ms/trial; hot_80 was slowest at 38 ms/trial because
  ~107 dots per frame make identification O(N²) work). 220,013 CSV lines = 220,000 rows + 13 headers. 21 MB.
- Results downloaded (campaign_results.zip, 10.1 MB) → `results/campaign/` (git-ignored). The analysis re-run
  on the Mac gives a **byte-identical report**. Report, summary.json and figures copied to the tracked `docs/results/day5/`.
- **Baseline:** 100,000/100,000 correct, CI [99.996, 100] %. Pointing p50/p90/p99/p99.9 = 1.58/3.06/4.61/6.09″; roll 6.45/16.60/28.17/38.67″.
  Stars in view mean 27.3 (min 7, max 85); the 88 baseline frames with < 10 stars were all solved.
- **Noise sweep** (correct %): 2.5 100 · 5 100 · 7.5 100 · 10 99.99 · 12.5 99.59 [99.44, 99.70] · 15 98.06 [97.77, 98.31] ·
  17.5 93.28 [92.77, 93.75] · 20 86.35 [85.66, 87.01]. (Day 3's 1,000-trial noise-20 value was 86.0.)
- **Hot pixels** (correct %): 5 100 · 10 99.99 · 20 99.83 [99.73, 99.89] · 40 98.11 [97.82, 98.36] · 80 80.54 [79.75, 81.30].
- **Wrong answers: 0 / 220,000** (95 % upper bound 0.0017 %).
- **Failure geography** (noise 20): fail 2.5 % at |galactic latitude| < 15° (mean 38.9 stars) vs 29.0 % at > 60° (18.2 stars).
- **Figure fixes after viewing:** last star-count bin plotted at 130 (bin midpoint) → plot at the bin's mean; failure-map hex gaps and RA labels hidden
  under the hexes → edge-coloured hexes + labels drawn on top with a halo; legend states "baseline 0".
- Colab cleanup: runtime deleted; the notebook remains in my Drive. A generative-AI privacy notice popped up
  once (mis-click); it was cancelled, nothing accepted. `notebooks/colab_run.ipynb` gained the campaign + download cells.

### Correction (2026-10-15): Day 5 figures were NOT in commit 91f819f
- `.gitignore` had `results/`, which ignores a `results` folder at ANY depth, so `docs/results/` was silently skipped by `git add -A`.
  Commit 91f819f's message (and my chat summary) wrongly said the figures were committed. Fixed by changing the rule to `/results/`
  (top-level raw data only) and committing `docs/results/` now. Lesson: after committing new artefacts, check `git ls-files`.

## 2026-10-15: Day 6, benchmark + README

- **Built.** `experiments/benchmark.py` (CPU: 1-core stage medians + all-core process pool with thread env vars set before spawn;
  GPU: batches of 128, 2 warm-up batches, CUDA events + synchronise, transfers included, identify on a process pool),
  `experiments/benchmark_report.py` (merged table + figure, projection labelled), `experiments/demo_figure.py` (README image),
  `README.md`. Tests: `tests/test_benchmark.py` (3 smoke tests). Suite (Mac): **119 passed**, + 12 GPU-only tests = 131.
- **Problem: thread env var set too late.** `OMP_NUM_THREADS=1` was set inside the workers after NumPy loaded (no effect) → now set before spawning.
- **GPU run** (Colab T4 70 W, Xeon 2.0 GHz × 2 vCPU, private account): median ms/frame prepare 0.138, pack 0.004, render kernel 0.242,
  detect + download 0.075, identify (2 processes) 2.535, total **3.028**; 2,560/2,560 solved. Colab single core: render 21.0, detect 8.0,
  identify 2.8. JSON saved on the Mac by reading it off the page (no second download needed).
- **CPU draft run on battery** (M1 Pro, 10 cores): 1 core render + detect 15.03, full 16.98 ms; 10 processes 2.33 / 2.72 ms.
  T4 image stages: 47× vs 1 core, **7.2× vs 10 cores**. End to end the T4 host (3.03) is slower than the 10-core Mac (2.72): Amdahl.
  FINAL CPU numbers to be re-measured on AC power.
- **Problem: `docs/results/` was git-ignored** (see the correction above). Fixed `.gitignore`; Day 5 artefacts committed in 9d5100c.
- **README fact-check.** Corrected my own draft: 131 tests (not 126), render + detect = 87 % of a 1-core frame (not 89 %).

### Day 6 final CPU benchmark (2026-10-15, AC power, Low Power Mode off): Apple M1 Pro 8P + 2E, 32 GB
- 1 core stage medians (500 frames): prepare 0.215, render 7.225, detect 4.922, identify 1.421 ms (render + detect = 88 %).
- Throughput (4,000 frames, 10 processes): render + detect **1.854 ms/frame**, full pipeline **1.999 ms/frame (500 frames/s)**;
  1 process (1,000 frames): 11.968 / 13.351 ms.
- Battery vs AC: battery was ~20 % slower (10-process render + detect 2.33 → 1.85 ms). The draft numbers are replaced everywhere.
- **Final comparison:** T4 image stages 0.323 ms/frame incl. transfers → **37× vs 1 core, 5.7× vs all 10 cores**. End to end the
  10-core Mac (2.00) beats the T4 + 2-vCPU host (3.03) by 1.5×: identify is 84 % of the GPU host's time (Amdahl). Projection with identify
  on the Mac's 10 cores: 0.67 ms/frame (not run as one system; labelled as a projection).
- Artefacts: `docs/results/day6/{benchmark.md, benchmark.png, cpu_m1pro.json, gpu_colab_t4.json}`; README updated.

## 2026-10-15: Day 7, public demo website

### Publishing prep: history rewritten
- Backup: `~/star-tracker-gpu-backup/before-rewrite-2026-10-15.bundle` (verified) + a copy of the private project notes file.
- `git filter-branch` over all 36 commits: dropped the private notes file, replaced its two mentions with "project plan", fixed one commit
  message; force-pushed with `--force-with-lease` against the known old head (a369630 → 0a9aacf). Verified 0 matches in every commit's files and messages.
  The notes file stays local, excluded via `.git/info/exclude` (not `.gitignore`, which is public).

### Website (`web/`, `tools/build_site.py`, `.github/workflows/pages.yml`)
- **Built.** `web/bridge.py` (Python entry points: setup, run_frame, random_pointing, sky_stars; 6 tests), `tools/build_site.py`
  (→ `_site/`: pages + tracker code minus gpu.py/fetch_catalog.py + catalog + stored results + manifest; 2 tests, one runs the bundle in an isolated
  interpreter), `web/worker.js` (Pyodide 0.27.7 pinned: NumPy 2.0.2, SciPy 1.14.1), `web/app.js` (globe, image + overlays, steps),
  `web/results.js` (stored results; every number read from JSON), `web/index.html`, `web/style.css`, Pages workflow (tests → build → deploy).
- **Deviation.** Folder `web/` instead of the planned `site/`: `site` is a standard-library module name and would shadow it on `sys.path`.
- **Measured in Chrome (M1 Pro).** Orion solved, 1.5″ pointing; render 207, detect 46, identify 83 ms (343 ms total). Random sparse field: 20 dots,
  solved, 0.5″, 132 ms. Results tab: tiles and tables match `docs/results/day5/report.txt`.
- **Problems found while testing in the browser.** (1) `display: grid` overrode the `hidden` attribute, so the layout showed while loading →
  `[hidden] { display: none !important }`. (2) Slider labels wrapped. (3) My "native is ~3× faster" text was wrong (it's ~25×) → state the measured
  native time instead. (4) "Lost in space" clicked during a solve was silently dropped → now queued. (5) Blank white blocks under shorter figures →
  `align-items: start`. (6) "220,000" was typed into results.js → now derived from the JSON. (7) The no-solution test assumed noise 200 hides Orion;
  it doesn't (its bright stars survive), so it uses 2000.
- Suite (Mac): **127 passed**.

### Deployed (2026-10-15): https://knzyahmed.github.io/star-tracker-gpu/
- I made the repo public and set Pages → Source: GitHub Actions. The first run (238c79c, before that) passed the tests on GitHub's Ubuntu
  runner but failed at configure-pages (Pages not yet enabled) as expected. A README badge commit (fa65902) re-triggered it: tests ✓, publish ✓.
- Live check in Chrome: first load ~35 s (Pyodide + NumPy + SciPy from the CDN), Orion solved 1.5″ (280 ms); a globe drag to Lepus
  (RA 74.5°, Dec −17.9°, 29 stars) solved 1.4″ in 135 ms. All bundle files return HTTP 200.

### Website redesign (2026-10-15, from my brief + 8 inspiration images)
- **Brief.** Globe and camera image as the main focus; the step-by-step must not take horizontal space; "lost in space" mood,
  deep blues → purples → violets. Inspiration: full-bleed dark skies, no card boxes, thin lines, small wide-tracked uppercase labels,
  a large light serif title, numbered sections (Space Advisor, NASA Eyes, SpaceX concepts, a Webb nebula).
- **Built.** New `web/style.css` (tokens: #060816 → #0b1030 → #1b1f5e → #4b2a8a → #7c4dff/#b39dff; Cormorant Garamond display +
  Inter UI), drawn starfield canvas + faint SVG orbit arcs behind the page, nebula gradients; two equal 1:1 visuals side by side
  (globe 640 px canvas with glow and boresight cross); the solve as a 4-node horizontal timeline under them (vertical on phones);
  verdict badge on the image; thin sliders and outlined buttons; results tiles as a ruled strip; numbered "How it works".
- **Dark charts.** `analyze_campaign.py --theme dark` and `benchmark_report.py --theme dark` render transparent PNGs with light ink
  and the dark-stepped palette into `docs/results/day5/dark/` and `day6/dark/`; the site uses those, the README keeps the light ones.
- **Checked in Chrome.** Desktop 1300 px: all three tabs. Phone 390 px: an overflow to 399 px was traced (JS: elements with right > 390)
  to the nowrap comparison table in the solve section stretching the one-column grid → `min-width: 0`, scrollable `.answer`,
  first column may wrap. Now exactly 390 px wide.

### Website, second pass (2026-10-15, after my review): blue tones, text boxes, 3-D sphere, Starlink/Space Advisor layout
- **Brief.** Blue tones instead of violet; text should not sit on bare stars (text-box effect); the globe should look 3-D and sit on
  one side with the space image behind it; the pointing panel underneath it. References: the Starlink concept and Space Advisor.
- **Built.** Palette #04070f → #081225 → #12264a → #1f4f8f → #4aa3ff/#9fd0ff. `.box` = dark glass panel (rgba(5,11,26,.74) +
  blur) on every text block, captions, controls, timeline, tables and figures. Background = a Webb nebula image
  (`web/bg.webp`, 640 px source) converted to a blue duotone with Pillow (grayscale → colorize navy/steel/ice → darkened, 39 KB),
  blurred 6 px behind the starfield; credit in the footer. Globe: ZOOM 1 (true orthographic hemisphere), body gradient lit from the
  upper left, grid and stars fading toward the limb, limb-darkening overlay, specular highlight, glowing FOV box, atmosphere glow.
  Layout: left column = title box → camera image; right column = sphere → "pointing and conditions" panel (2-column sliders);
  solve timeline full width. Charts re-rendered with the blue dark palette.
- **Problems.** (1) Grid rows aligned the two columns and left a hole under the title box → two independent flex columns
  (`display: contents` on phones to reorder: title, globe, panel, image, solve). (2) Phone overflow to 405 px: the desktop
  `align-items: start` leaked into the phone flex column, so `.solve` sized to its content → `align-items: stretch` in the media query.
  Both found by measuring element rects with JS in a 390-px iframe. Final: 390 px, all boxes 358 px.

# Phase 2: optical navigation

## 2026-10-18: docs rewritten in the first person
- Rewrote the "owner" wording in BUILD_LOG.md and LEARNING.md into the first person as the phase rules require, and removed the
  personal account details (e-mail addresses, a browser-connection name) that had crept into this public log.

## 2026-10-18: Day 8, geometry + Earth renderer (CPU reference)

### `tracker/world.py`
- **Built.** WGS84 radii with source, `EARTH_LEVEL = 3000` counts (assumption documented), `sample_position` (log-uniform distance,
  uniform direction), `sample_sun`, `earth_in_view` (overlap test of the disc's box with the sensor; raises inside Earth).
- **Correction to the plan.** The GEO sanity value is 8.70°, not 8.69° (arcsin(6378.137/42164) = 8.7005°); 444.4 px. Lunar: 0.951°, 48.2 px.
- **Tests.** `tests/test_world.py`: 10 (constants, log-uniform quartiles, uniform directions, sanity values, off-axis, partial, inside-Earth error).

### `tracker/earth_render.py`: `render_earth`, `occluded`, `render_scene`
- **Built.** Bounding box from `earth_in_view` (stretched 1/cos² off-axis, +6 px halo), 4 × 4 sub-rays per pixel via `unproject`,
  ray–ellipsoid in scaled space, true ellipsoid normals, Lambertian, average, `gaussian_filter` with SIGMA_PSF. `shade=False` draws a
  flat disc for geometry tests. Mask = any sub-ray hit. Occlusion by the same ray test.
- **Measured.** Edge of the flat disc: 50 % crossing 0.12 px inside the geometric limb at 4 × 4 (edge quantisation 1/8 px), within
  0.08 px at 8 × 8. Oblate Earth from the equator: 1.5 px shorter than wide at GEO, as predicted. Full-phase shading at r = 0.5 R: 0.899
  (finite-distance formula), centre = EARTH_LEVEL within 1 %. Flux ratios at GEO vs an independent surface integral: within 1 %;
  half/full = 0.26 (1/π = 0.318 only in the orthographic limit; lunar distance gives 1/π within 1.2 %).
  Render time on the M1 Pro: GEO 1.5 s, 100,000 km 0.3 s, lunar 0.04 s per frame.
- **Problems.** (1) Four of my first test expectations were orthographic-camera results (sqrt(1 − (r/R)²) shading, Lambert phase law)
  and a too-tight 0.1 px edge tolerance at 4 × 4; the renderer was right each time, so I fixed the expectations and documented the finite-distance
  maths. (2) The occlusion test had no star behind a 48 px Earth; it now puts Earth in front of Betelgeuse from GEO.
- **Design note for Day 9.** The plan's "outermost crossing of 50 % of the line's peak" would bias the limb inward at low phase angles
  (Lambertian fade-out); use a low fixed threshold or the gradient instead.
- **Tests.** `tests/test_earth_render.py`: 13. Suite: **151 passed** (22 s of it is this file: GEO renders at 4 × 4 and 8 × 8).

### Day 8 gate: `experiments/show_earth.py` → `docs/results/day8/earth_grid.png`, `earth_geo_phase90.png`
- 3 distances × 3 phases with the real star field, occlusion and noise. Lit fractions 1.00 / 0.49 / 0.09–0.15; stars in view 9 (GEO, most
  hidden or outside), 17, 18. Images look right: sharp limb, soft terminator, thin crescents, hidden stars. **Gate: images ready for review.**
- **Review checks before Day 9 (2026-10-18).** (1) Stars behind the night side: added a test that the star list is identical at phase 0
  and 180 and that the fully dark disc holds only noise (max < background + 6σ, 8 px inside the limb); Betelgeuse stays hidden.
  Occlusion was already pure geometry, so this passed first time. (2) Raw-data edge test: at phase 90 the lit limb goes 10 % → 90 % within
  4 px, the terminator takes > 40 px. Day 9's limb finder will only search the lit side.

## 2026-10-18: Day 9, limb detection

### `detect(mask=)`, `limb.exclusion_mask`
- **Built.** Optional ignore-mask in `detect` (old tests unchanged); `exclusion_mask` = largest bright region dilated by 6 px.
- **Measured.** 100,000 km, phase 90, with stars and noise: without the mask the limb makes > 20 false stars; with it, all detections are
  real stars except one 5σ noise false alarm 175 px away (0.3 expected per frame).

### `tracker/limb.py`: `segment_earth`, `find_limb`
- **Decision: edge rule chosen by measurement.** Noise-free sphere, three distances, five phases (table in LEARNING.md): the plan's
  "50 % of the line's peak" is biased on a Lambertian limb; a fixed low threshold is +2–3 px outward (PSF tail); max gradient is noisy.
  Chose 50 % of the local plateau read 6 px inside the coarse edge (tuned over 2/3/4/6/8 px): within ±0.16 px at phases ≥ 45°.
- **Deviation from the plan.** Tolerance 0.15 px (not 0.1) for the noise-free limb test at GEO and 100,000 km: the 4 × 4 render edge
  quantisation alone is 0.125 px. Lunar distance: 0.3 px.
- **Known biases kept, not hidden:** phase 0 (fading limb) −0.5 to −1.1 px; thin crescents (lunar, 135°) +0.7 px. A test pins the
  phase-0 bias range so a later fix is visible as a test change.
- **Measured on noisy frames with stars** (`experiments/show_limb.py`): GEO phase 90: 179 points, median +0.03 px, p95 |err| 0.06 px.
  100,000 km, 5° off-axis, roll 30°, oblique Sun: 179 points, −0.03 / 0.09 px. Lunar, phase 135°: 180 points, +0.73 / 0.96 px.
- **Problems.** Two test expectations were mine, not the code's: 300 points asked of 180 lines; "mask bigger than the true disc" when
  only the lit half is bright; one noise false alarm out of 17 detections failed a 95 % rule → "at most one".
- **Tests.** `tests/test_limb.py`: 9. Suite: **163 passed**.
- **Day 9 gate:** `docs/results/day9/limb_grid.png` (limb points on three noisy frames + radial errors) ready for review.

## 2026-10-18: Day 10, horizon fit → position

### `tracker/opnav.py`: `limb_directions`, `fit_horizon_sphere`, `fit_horizon_ellipsoid`, `position_error`, `position_from_image`, `attitude_second_head`
- **Built.** Linear least squares for n in d·n = 1, p = −R n/√(n·n − 1); ellipsoid by scaling space; MAD outlier rejection (k = 4,
  ≤ 5 rounds); `position_error` → (total, signed range, cross-range) km; `position_from_image` = find_limb → directions → fit, None when
  < 20 points or the fit is inconsistent (|n| ≤ 1); configuration (b) second head: A_est = Mᵀ A_head_est (10″/60″ in the test).
- **Measured.** Exact recovery (1e-9) for sphere and ellipsoid at GEO/100k/lunar and on a 60° arc. Sphere model on an oblate Earth,
  full limb from the equator: +71 km at GEO, +645 km at lunar (+0.17 %); from the pole +3.2 km (hand: +3.4). The plan's "66 km at GEO for a
  random direction" is consistent (it depends on the viewing latitude). Noise 0.3 px: error ratio 20 vs 180 points ≈ 3 (1/√n).
  End to end, GEO phase 90, noisy, true attitude: range +1.0 km (0.002 %), cross 0.7 km.
- **Problems.** Two expectations of mine: a 10 µm difference failed rtol with no atol; "zero bias from the pole" is only true from infinity.
- **Tests.** `tests/test_opnav.py`: 14. Suite: **177 passed**.

### Limb bias → position error (`experiments/day10_bias.py` → `docs/results/day10/bias_table.md`)
- Real fit on the Day 9 frames (sphere Earth, stars, noise; also noise-free twins), TRUE attitude, sphere model. Noise changes nothing
  at the 1-km level: these are biases, not noise.
- GEO phase 0 (full circle, uniform −0.71 px): range **+65.5 km (+0.155 %)**, cross 0; hand estimate −Δr/r = +67 km. Rule confirmed.
- Lunar phase 135 (60° arc, +0.47..+0.96 px, biggest in the middle): range **+22,355 km (+5.8 %)**, cross 489 km (0.13 %); the hand
  estimate says −5,917 km (−1.5 %): wrong sign, ¼ the size. Explained by curvature: the uneven bias adds 0.49 px to a 6.46 px sagitta →
  apparent radius 45.2 px instead of 48.2 → 6.5 % farther. Lunar phase 90 (97° arc, +0.03..+0.19 px): +1,507 km (+0.39 %), same mechanism.
- 100,000 km off-axis (median bias −0.03 px but uneven −0.13..+0.03): +87 km (+0.09 %), cross 5 km.
- **The lunar range error exceeds 1 %, so I stopped here and proposed a fix (below) instead of applying it.**
- **Proposed fix (awaiting approval):** replace the 50 %-of-plateau crossing by fitting, per scan line, the *predicted* blurred edge
  profile: Lambertian shading along the line (known from the geometry and the Sun direction) × disc coverage, convolved with the
  Gaussian PSF (σ = 1.2 px), with the edge position and a brightness scale as the two free parameters. This removes the phase-dependent
  bias (fading limb at phase 0, thin crescents) at its source, keeps the same scan lines, and can be tested against the noise-free renders
  to 0.05 px. Cost: a small nonlinear fit per line (~180 per frame, each a 1-D search over ~0.1 px steps or a Gauss–Newton on 2 parameters).

### Fix approved and implemented (2026-10-18): predicted-profile edge fit, `limb.refine_limb`, two passes in `position_from_image`
- **Built.** `scan_lines` shared by both finders; `refine_limb`: per line, predicted shape = Lambertian (n·s)+ of the sphere at p₀ along
  the line (0 beyond its limb), convolved with the σ = 1.2 px Gaussian; model = A · shape(r − δ); A in closed form, δ on a 0.05 px grid
  (±4 px) + parabolic refinement; fit window 14 px inside / 8 px outside the predicted edge; lines with peak < 20σ skipped.
  `position_from_image(refine=True, iterations=2)`: 50 % rule → fit → (refine → fit) × 2. `refine=False` keeps the old rule.
- **Decision: 2 iterations.** Measured on the noise-free lunar crescent: 1 pass −0.13 px / +0.66 % range; 2 passes −0.05 px / −0.06 %;
  3 passes −0.04 px / −0.01 % (no real gain). Render supersampling 4/8/16 gives the same residual, so it is the 1-D model's limit, not a render artefact.
- **Before/after** (`experiments/day10_bias.py`, both methods, `docs/results/day10/bias_table.md`): GEO phase 0 +65.5 → +1.8 km;
  100,000 km off-axis +87 → +13 km; lunar 90 +1,507 → +348 km (+0.09 %); lunar crescent +22,355 km (+5.8 %) → −220 km (−0.06 %) noise-free,
  +122 km (+0.03 %) noisy; cross-range 489 → 11 km. All below the 1 % bar; the worst case is now lunar phase 90 at 0.09 %.
- **Tests.** `tests/test_limb.py` +7 (bias removed at 4 phase/distance cases, even along the arc, with noise, end-to-end < 0.3 % and
  better than the old rule). Suite: **184 passed**.

## 2026-10-18: Day 11, Earth on the GPU

- **Built.** `earth_render.earth_bbox` + `subpixel_offsets` factored out (CPU renderer unchanged, same tests pass); `kernels/earth.cu`
  (one thread per bbox pixel, 4 × 4 sub-rays, double-precision ray–ellipsoid + true normal + Lambertian, float32 out, uint8 mask);
  `gpu.render_earth_batch` (per-frame launch over the host bbox, cupyx `gaussian_filter` blur, bool masks). `tools/cuda_shim.h` learned
  `sqrt(double)`. `experiments/gpu_check.py --earth` for agreement + timing.
- **Decision: double precision for the Earth geometry** (stars stay float32): the limb is a hit/miss cliff and |p| ~ 10⁵ km makes the
  float32 discriminant cancel; one flipped sub-ray = brightness/16 in a pixel. Doubles cost milliseconds on the T4.
- **Gate on the T4 (Colab, private account):** full suite **196 passed, 4 skipped** (clang checks) after one fix; GPU vs CPU Earth:
  max |Δ| 7.3e-4 counts (GEO, 100k), 2.4e-4 (lunar), masks identical, limb points 180/180 within 4.4e-7 px. Timing: 128 Earth-pointing
  frames GEO–lunar (median 153,585 km) in 0.52 s = **4.07 ms/frame** incl. blur; CPU GEO frame 3.2 s (Colab Xeon) / 1.5 s (M1 Pro).
- **Problems.** (1) `test_gpu.py` lacked an import; invisible on the Mac (skips at importorskip) → fixed, plus an AST undefined-name check
  before zipping. (2) `test_opnav` exactness tolerance 1e-6 km failed on x86 by 1.5e-6 km (lstsq rounding) → 1e-4 km (10 cm).
  (3) Colab's session-files warning dialog swallowed a typed cell once; Escape does not close it, OK does.
- Suite (Mac): **185 passed**, 1 skipped; with CUDA: 196 passed.

## 2026-10-20: Day 12, the optical-navigation campaign (part 1: engine, thresholds, estimate)

- **Thresholds, fixed before any campaign trial ran.** Three outcomes per trial, scored on the TOTAL position error
  against the true distance D:
  - **no solution**: the second head's star tracker returns nothing, or fewer than 20 limb points survive
    (`opnav.MIN_LIMB_POINTS`), or the horizon fit raises (|n| ≤ 1). The reason is stored per row.
  - **correct**: a position was returned and |p_est − p_true| < **1 % of D** (`opnav_campaign.CORRECT_FRACTION`).
  - **wrong**: a position was returned and the error is ≥ 1 % of D.
  Why 1 %: the worst honest error I had measured before the campaign was 0.17 % of range (lunar distance, Day 10
  profile fit on an ellipsoid Earth), so 1 % is about 6× above it, and a fit that locks onto the wrong edge
  (the terminator, or a star) is off by tens of %, so 1 % is well below that. A relative threshold is the right one
  because every error source here (limb bias in px, attitude error in arcsec) scales with D.
- **Precision line for `earth.cu`.** I measured what single precision does by running the CPU reference with the
  ray–ellipsoid intersection cast to float32 (same arithmetic as the kernel): 30 % of Earth pixels change at GEO
  (max 48 counts, one flipped sub-ray = 3000/16 ≈ 190 counts × PSF), 7 % at 100,000 km, 0.3 % on the lunar crescent;
  individual limb points move by at most 0.05 px (median 0.000), and the range error changes by < 0.02 percentage
  points (GEO +0.0001 → +0.0007 %, lunar crescent +0.025 → +0.005 %). So float32 would have been accurate enough;
  I kept **double** because the Day 11 gate is "GPU image equals CPU image", which float32 cannot pass at a hit/miss
  edge with |p| ~ 10⁵ km (a 1e-7 relative error in the discriminant flips sub-rays), and because an exact CPU reference
  is what lets the CPU campaign rows stand in for GPU rows. **T4 cost of double, measured** (`gpu_check --earth`, 128 Earth-pointing frames GEO to lunar, median of 3): kernel
  only 2.65 ms/frame in double vs 0.27 ms in float32 (`-DREAL_T=float`), i.e. **~10× slower**; with the PSF blur 3.03 vs
  0.67 ms. On the T4 itself float32 changed 30 % of Earth pixels at GEO (max 27 counts), flipped 1 mask pixel in
  three frames, moved limb points by at most 0.04 px and the range error by < 0.03 percentage points. So the double
  decision buys CPU-equals-GPU exactness (4.4e-7 px on the limb) for 2.4 ms per frame, which is nothing next to the
  ~300 ms CPU limb fit per frame; if the fit ever moved to the GPU, float32 would be the right choice there.
- **Built.** `tracker/opnav_campaign.py`: `trial_geometry(seed, trial, distance, phase)` (direction to Earth uniform,
  Sun azimuth uniform at the requested phase angle, camera offset uniform within the sensor minus 10 px so the whole
  disc is in the frame, roll uniform; the draws do not depend on the cell, so trial k is the same sky in every cell),
  `score_position`, `solve_trial` (ellipsoid fit, then the sphere model on the same points), `render_trials`
  (CPU or GPU: stars with occlusion + Earth + counter noise; the second head's star frame with its own noise key),
  `solve_rows`, `run_batch`, `run_condition` (resume; `workers` processes on the CPU backend). Rows carry the
  position, Sun, pixel radius, offset, the head tracker's attitude error, the reason, limb count, outcome, total /
  range / cross-range error and the sphere-model error. Nominal-noise cells get a second row per trial solved with the
  TRUE attitude (`att_src` 1) so the star tracker's share of the error can be separated. 9 tests
  (`tests/test_opnav_campaign.py`), plus a GPU-only CPU-equals-GPU row test in `tests/test_gpu.py`.
- **Speed-up that made the campaign feasible.** `refine_limb` took 0.5 s per pass at GEO: a Python loop over 161
  shifts × 180 lines, and the predicted profile sampled at 0.05 px from the disc centre out (9,000 samples per line).
  Now the shift search is one vectorised least-squares over all shifts (`limb._fit_shift`, exactness test against the
  plain loop) and the fine profile is built only around the model edge on the same lattice. Results are bit-identical
  (the Day 10 bias table regenerates unchanged); full pipeline 1,050 → 368 ms at GEO, 528 → 246 ms at lunar distance.
- **Smoke run** (5 trials per cell, 10 processes, 90 s): 480 trials, 2 "too few limb points", 0 tracker failures;
  tracker pointing error median 7.8" (the head's roll error becomes the Earth camera's pointing error, as expected
  for a 90° mount), cross-range median 7 km with the tracker vs 1 km with the true attitude. Range error median
  0.005 % at GEO rising to 0.09 % at 420,000 km; beyond ~560,000 km (disc radius < 33 px) half the trials fail the
  1 % threshold and at 1,000,000 km (18 px) the median is 2.3 %: the limb finder's fixed windows (plateau read
  6 px inside the edge, fit window 14 px inside) are wider than the disc. That is a finding the campaign will
  report, not something to tune away before running.
- **Estimate.** 12 distances × 4 phases × 2 noise levels × 1,000 = **96,000 trials**, 144,000 solves (the nominal-noise
  cells are solved twice, tracker and true attitude), 144,000 rows. Measured CPU cost ≈ 0.96 core-seconds per trial
  (render + head tracker + two solves) → ≈ 26 core-hours → **≈ 3 h on the M1 Pro's 10 cores** on mains power.
  On Colab the GPU part is small (≈ 8 min of Earth + star rendering) but the limb fit is CPU work on a 2-vCPU host
  about 2× slower than the M1: ≈ 14 h, beyond a session. Decision (to be approved): the campaign runs on the Mac with
  the CPU backend; Colab runs the GPU checks (CPU-equals-GPU rows, float32 cost and timing) and one full GEO cell on
  the GPU backend as the record that the GPU path produces the same rows.
