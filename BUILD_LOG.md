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
