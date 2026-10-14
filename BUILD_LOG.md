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
