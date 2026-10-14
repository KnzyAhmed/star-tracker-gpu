# LEARNING.md

One entry per function or concept: what it does, IN / OUT, a worked example by hand,
why it is done this way, and one interview question.

Entries start at Day 2. Day 0–1 (camera model, catalog, renderer) are not yet covered here.

---

## Day 2: picture back to answer

### `detect(image, background, read_noise, k_sigma=5)`: finding stars (`tracker/detect_cpu.py`)

**What it does.** It scans the image for pixels that are clearly brighter than the noise and
are the brightest pixel in their 3×3 neighbourhood. Each such pixel is one star. It then
measures where the star's centre really is, to a fraction of a pixel.

**IN / OUT.**
- IN: `image` (1024, 1024) float, counts, indexed `image[v, u]`; `background` (counts/px, here 20);
  `read_noise` (counts, here 5); `k_sigma` (dimensionless, default 5).
- OUT: `u, v, flux`, each of shape (N,), float64. `u` = column and `v` = row in pixels, with
  sub-pixel precision. `flux` = background-subtracted sum of the 5×5 window in counts.
  Sorted brightest first.

**Worked example (centroid).** A star's peak is at pixel (u=300, v=400). After subtracting the
background, the 5×5 window has 100 in the centre, 50 just to the right (u=301), 20 just
to the left (u=299), and 0 everywhere else.
- flux = 100 + 50 + 20 = 170
- u = 300 + (50·(+1) + 20·(−1) + 100·0) / 170 = 300 + 30/170 = **300.176**
- v = 400 + 0/170 = **400.000**

The star is pulled toward the brighter right-hand neighbour by about 0.18 px.

**Worked example (tie-break).** Two neighbouring pixels both read exactly 1000:
(v=500, u=500) and (v=500, u=501).
- Pixel u=500: its right neighbour (1000) comes *after* it in raster order, so it only needs
  1000 ≥ 1000, which is true. **Candidate.**
- Pixel u=501: its left neighbour (1000) comes *before* it, so it needs 1000 > 1000, which is false. **Rejected.**

Result: one star, centroid u = 500.5. If both tests used `>`, the star would be lost. If both used
`≥`, it would be counted twice.

**Threshold arithmetic.** Threshold = 20 + 5·5 = 45 counts. A star's peak pixel is about
flux / (2π σ²) = flux / 9.05. A magnitude 5.5 star has flux 60000·10^(−2.2) = 379 counts, so its peak is about
42 counts above background. That is 8.4σ, comfortably detected. Pure noise exceeds 5σ with probability
2.9·10⁻⁷ per pixel, so about 0.3 false stars per 1-Mpx frame. Measured: 0.2.

**Why this way.**
- *Threshold first, then local-max only on survivors.* Only a few hundred of the 1 M pixels pass,
  so the eight neighbour comparisons are cheap. The obvious wrong approach is a Python loop over
  every pixel, which is ~1000× slower and makes the CPU baseline unfair.
- *Local maximum.* Without it, every bright pixel of a blob would count as a separate star.
- *Pad windows with `background`* so off-sensor pixels contribute exactly zero weight. Edge stars
  don't crash, and the image doesn't wrap around.
- *Negative weights are kept.* Noise below background gives negative weights. Clipping them to 0
  would add positive weight everywhere in the window and pull every centroid toward the window centre.
- *5×5 window and pixel locking.* The window is centred on the *peak pixel*, not the true centre.
  So when the star sits 0.5 px off a pixel centre, the window cuts off more of the blob on one side,
  and the centroid is pulled toward the pixel centre by up to **0.10 px**. That is with no noise at all (see the table below).
  A 7×7 window cuts the bias to 0.017 px, but Σx² over the window grows from 50 to 196. That roughly doubles
  the noise error for faint stars, which dominate the star count. The spec keeps 5×5.

| true offset (px) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| 5×5 bias (px) | −0.019 | −0.038 | −0.057 | −0.078 | −0.100 |
| 7×7 bias (px) | −0.003 | −0.006 | −0.009 | −0.013 | −0.017 |

**Interview question.** *"Your centroid errors are not centred on zero: they cluster toward
integer pixels. Why, and how would you fix it?"*
Answer: this is pixel locking. A finite window centred on the peak pixel truncates the PSF
asymmetrically, so the weighted mean is biased toward the window centre. Fixes: a larger window
(more noise), iterating the window around the latest centroid, fitting a Gaussian PSF (slower,
near-optimal), or a lookup-table correction calibrated on simulated stars. Defocusing the optics so σ
is larger relative to the pixels also helps, which is why real trackers defocus.

---

### `solve_attitude(b, r, w=None)`: Wahba's problem by SVD (`tracker/solve.py`)

**What it does.** You know where some stars appear in the camera (`b_i`) and where they are on
the sky (`r_i`). It finds the one rotation `A` that best turns every `r_i` into its `b_i`.
That rotation *is* the spacecraft attitude.

**IN / OUT.**
- IN: `b` (N, 3) camera-frame unit vectors (from `unproject` of detected centroids);
  `r` (N, 3) sky-frame unit vectors (catalog), row i of each = same star; `w` (N,) weights ≥ 0
  (None → all 1). Dimensionless. Needs N ≥ 2 non-parallel stars.
- OUT: `A` (3, 3) float64, proper rotation (Aᵀ A = I, det A = +1), sky → camera: `b ≈ A @ r`.

**The maths in four lines.** Minimise J(A) = ½ Σ wᵢ |bᵢ − A rᵢ|². Because |bᵢ|=|A rᵢ|=1, this equals
Σ wᵢ − Σ wᵢ bᵢᵀ A rᵢ = const − trace(A Bᵀ), with **B = Σ wᵢ bᵢ rᵢᵀ** (the "attitude profile matrix").
So maximise trace(A Bᵀ). Write B = U S Vᵀ. The trace is largest when A = U Vᵀ ("rotate V onto U").
If det(U Vᵀ) = −1, that is a reflection, so flip the axis with the smallest singular value:
**A = U diag(1, 1, det U · det V) Vᵀ**.

**Worked example by hand.** Camera rolled 90° about its boresight: the sky x-axis appears as camera y, and the sky y-axis appears as camera −x.
- r₁ = (1,0,0) → b₁ = (0,1,0); r₂ = (0,1,0) → b₂ = (−1,0,0).
- B = b₁r₁ᵀ + b₂r₂ᵀ = [[0,−1,0],[1,0,0],[0,0,0]].
- B is already a rotation times diag(1,1,0): U = B + e₃e₃ᵀ = [[0,−1,0],[1,0,0],[0,0,1]], S = diag(1,1,0), V = I.
  det U = 1, det V = 1 → d = 1.
- A = U · I · I = [[0,−1,0],[1,0,0],[0,0,1]]. Check: A r₁ = (0,1,0) = b₁ ✓, A r₂ = (−1,0,0) = b₂ ✓.
  Note the third singular value was 0 (two stars fill only a plane), but the solution is still unique.

**Why this way.**
- *Obvious wrong approach 1: least squares for the 9 entries of A* (solve A from `b = A r` with `lstsq`). It gives
  a matrix that is not a rotation (it shears/scales to soak up noise), and you then have to "fix" it.
- *Obvious wrong approach 2: TRIAD with the two brightest stars.* It is exact for 2 stars, but it throws away the other ~25.
  SVD uses every star, weighted, in one shot and is optimal in the least-squares sense.
- *Why `d = det U det V`:* if the data looks mirrored (bad matches, or noise with near-coplanar stars), `U Vᵀ` has det −1.
  A camera can't see a mirror-image sky, so we take the best *proper* rotation instead. The test
  `test_never_returns_a_reflection` builds exactly this case and confirms d = −1 is hit.
- Alternatives that give the same answer: Davenport's q-method (eigenvector of a 4×4 matrix, gives a quaternion) and QUEST
  (a fast iterative version of q-method used in flight software). SVD is the most robust; 3×3 SVD costs microseconds.

**Accuracy rule of thumb.** For N stars spread over all directions with σ per-axis noise each, each rotation axis is
known to σ·√(3/(2N)). With N = 30 and σ = 10″ that gives 2.24″ per axis, so the median total error is ≈ 1.54 × 2.24 = 3.4″. Measured: 3.6″.
In a narrow-FOV camera the stars are NOT spread over all directions: they all sit near the boresight, so
**roll (about the boresight) is much worse than pointing.** We will see that on the Day 2 gate.

**Interview question.** *"Why does Wahba's SVD solution need the det(U)det(V) term, and when does it matter?"*
Answer: the unconstrained maximiser of trace(A Bᵀ) over orthogonal matrices is U Vᵀ, which can have det −1 (a reflection).
The term restricts the answer to proper rotations by flipping the sign on the weakest singular direction. It matters when the
measurements are inconsistent or nearly degenerate (few stars, near-coplanar vectors, a wrong match). With good data,
det U det V = +1 and the term does nothing.
