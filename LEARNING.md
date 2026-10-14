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

---

### `attitude_error(A_est, A_true)`: how wrong is the answer? (`tracker/solve.py`)

**What it does.** It compares the estimated attitude with the truth and reports three angles in arcseconds:
- **total**: the size of the one rotation that turns the true attitude into the estimate;
- **pointing**: how far the boresight (where the camera looks) is off;
- **roll**: how far the image is twisted about the boresight.

**IN / OUT.** IN: two (3, 3) rotation matrices, sky → camera. OUT: `(total, pointing, roll)`, three floats ≥ 0, arcsec.

**The three formulas.** Let dA = A_est · A_trueᵀ (true camera axes → estimated camera axes).
- total θ: for any rotation, trace = 1 + 2 cos θ and the antisymmetric part dA − dAᵀ holds 2 sin θ · axis.
  So θ = atan2(|vee(dA − dAᵀ)|, trace − 1). (The two arguments are 2 sin θ and 2 cos θ; the common factor 2 cancels inside atan2.)
- pointing = atan2(|z_est × z_true|, z_est · z_true), with z = row 2 of A.
- roll = |atan2(dA₁₀ − dA₀₁, dA₀₀ + dA₁₁)|. For a pure z-rotation by φ, these are 2 sin φ and 2 cos φ.
  For a general dA, it is still exactly the "twist" angle 2·atan2(q_z, q_w) of dA's quaternion
  (expand the matrix in quaternion terms: dA₁₀ − dA₀₁ = 4 q_w q_z and dA₀₀ + dA₁₁ = 2(q_w² − q_z²)).

**Worked example by hand.** True: boresight at (RA 83.8°, Dec −5.4°), roll 30°. Estimate: same boresight, roll 30° + 20″.
dA is then a pure rotation about camera z by φ = 20″ = 9.696·10⁻⁵ rad:
- dA₁₀ − dA₀₁ = 2 sin φ = 1.9393·10⁻⁴; dA₀₀ + dA₁₁ = 2 cos φ = 2 − 9.4·10⁻⁹.
- roll = atan2(1.9393·10⁻⁴, 2) = 9.696·10⁻⁵ rad = **20.0″**. The boresights are identical, so pointing = **0**. total = **20.0″**.
- Check: total² = pointing² + roll² → 400 = 0 + 400 ✓.

**Why atan2, not arccos (measured):**

| true error | clipped arccos((tr−1)/2) | atan2 form |
|---|---|---|
| 1 mas | **0.0000 mas** | 1.0000 mas |
| 10 mas | 10.647 mas (6 % off) | 10.000 mas |
| 100 mas | 100.069 mas | 100.000 mas |

cos θ ≈ 1 − θ²/2. At θ = 5·10⁻⁹ rad, θ²/2 = 10⁻¹⁷, which is below float64's resolution near 1 (1.1·10⁻¹⁶),
so the information is gone before arccos even runs. Without the clip, rounding can give 1.0000000000000002 and arccos returns **NaN**.
sin θ ≈ θ has no such problem. Our errors are arcseconds (5·10⁻⁶ rad), where arccos would still lose about 3 digits.

**Why split pointing and roll.** A narrow-FOV tracker measures pointing much better than roll. All stars sit within about 10° of the
boresight, so a small twist about the boresight moves each star by only (its distance from the centre) × angle.
A single "total" number hides this. Every spec sheet quotes the two separately (e.g. "5″ cross-boresight, 40″ about boresight").

**Interview question.** *"Your star tracker's roll error is ~10× its pointing error. Is that a bug?"*
Answer: no, it's geometry. A pointing error shifts every star by about F·δ pixels. A roll error δ shifts a star at radius ρ from the
centre by only ρ·δ, and ρ is at most ~512 px, compared with F ≈ 2904 px. So roll sensitivity is roughly ρ_rms/F ≈ 1/10 of pointing,
and its error is ~10× bigger. Fixes: a wider FOV, or a second tracker head mounted at an angle (its boresight
measures the first one's roll).

---

### Day 2 gate: what the numbers say (`experiments/day2_gate.py`)

**What it does.** It runs 1,000 random attitudes through render → detect → match to truth → solve, using the TRUE
star identities (cheating on identification on purpose). This separates "how good is the measurement" from
"did we name the stars right" (Day 3).

**Results (seed 2).** 27.2 stars in view per frame on average (9–87). 98.5 % of them were detected and matched.
| quantity | median | p99 |
|---|---|---|
| centroid error | 0.103 px (7.3″) | 0.383 px |
| pair-angle error | 6.1″ | 30.9″ (p99.9 60.2″, max 95.5″) |
| pointing error | 1.67″ | 4.8″ |
| roll error | 6.80″ | 27.9″ |

**Three things to understand.**
1. *Bright stars are NOT much better than faint ones* (median 0.05–0.08 px for mag 0–4, 0.13 px at mag 5).
   With thousands of counts, photon/read noise is tiny, so what's left is **pixel-locking bias** of the 5×5 window
   (up to 0.10 px, see the `detect` entry). Bright stars are bias-limited and faint stars are noise-limited.
2. *Roll is ~4× worse than pointing.* This is the geometry explained in the `attitude_error` entry. Sanity check:
   7.3″ per star / √27 ≈ 1.4″, close to the measured 1.67″ pointing.
3. *Pair-angle error sets Day 3's eps.* Identification compares measured star-pair angles with catalog pair angles.
   The tolerance must cover nearly all real pairs, so we take it from this distribution rather than from the centroid
   error directly. A pair's angle error combines two stars' errors (and bias along the line joining them). Candidate:
   eps ≈ p99.9 ≈ 60″. Too tight loses true matches; too loose lets in more false catalog pairs.

**Interview question.** *"Why test the solver with true identities before writing the identification step?"*
Answer: to isolate error sources. If the full pipeline fails later, I already know the measurement chain (render,
detect, centroid, unproject, solve) is good to ~2″ pointing. So any failure is an identification problem. It also gives
the measured noise level needed to set the matching tolerance, rather than guessing it.

---

## Day 3: name the dots

### `build_pair_table(r, max_sep_deg=28.3)`: the offline pair catalog (`tracker/identify.py`)

**What it does.** It lists every pair of catalog stars that could both appear in one photo (closer than the FOV diagonal),
along with the angle between them, sorted by that angle. It is built once at startup and used for every image.

**IN / OUT.** IN: `r` (N, 3) sky unit vectors (N = 2,847). OUT: `PairTable(angle, i, j)`, each (M,) with M = 266,153.
`angle` is in radians, ascending; `i < j` are catalog indices.

**Worked example.** Three stars with separations A–B 5°, B–C 9°, A–C 12°. The sorted table is:
`[(5°, A, B), (9°, B, C), (12°, A, C)]`. Now a photo shows two dots 9.01° apart, and eps = 0.05°.
Binary search for 8.96° lands at index 1; binary search for 9.06° lands at index 2. So the candidates are rows 1..1 = **(B, C)**.
That took two `log₂(M)` searches (≈ 18 steps each for M = 266k) instead of scanning 266k rows.

**Why this way.**
- *Sorted by angle* → any "pairs within eps of θ" query is two binary searches (`np.searchsorted`), O(log M).
- *k-d tree to build it.* For unit vectors, angle < θ ⇔ chord |rᵢ − rⱼ| < 2 sin(θ/2), so a 3-D radius search finds the
  pairs directly. Time: 8 ms. The obvious wrong approach is the full N×N angle matrix: 8.1 M entries. That is fine at N = 2,847,
  but it is O(N²) memory and doesn't scale to fainter catalogs.
- *Cutoff = 28.3°.* Two stars farther apart than the sensor diagonal can never both be in the photo, so those pairs are
  pure clutter. The exact corner-to-corner angle is 28.0°; 28.3° (√2 × 20°) is a slightly generous bound.
- *How crowded is the table?* The number of pairs per arcsec grows like sin θ:

| separation | pairs per arcsec | inside ±31″ | inside ±60″ |
|---|---|---|---|
| 2° | 0.34 | 21 | 41 |
| 10° | 1.71 | 105 | 204 |
| 20° | 3.36 | 208 | 403 |

So a single measured angle is ambiguous: it matches hundreds of catalog pairs. Identification works only because
of **voting**: many pairs agreeing on the same star (next entry).

**Interview question.** *"Why store pairs, not single stars, and why not triangles?"*
Answer: a single star's brightness is a poor ID (magnitudes are noisy and many stars look alike). The angle between two
stars doesn't depend on the attitude, so it can be compared directly with the catalog. Triangles (three angles) are far more
distinctive per lookup, but the table is much bigger: catalog triples within the FOV number in the tens of millions. Pairs plus
voting combine many weak clues instead of a few strong ones. It is simple, and it is robust to a missing or fake star.

---

### `vote(b, table, n_cat, eps_arcsec)`: geometric voting (`tracker/identify.py`)

**What it does.** For every pair of detected dots, it measures their angle and finds every catalog pair with
(almost) the same angle. Each of those catalog pairs says "these two dots might be stars p and q", which gives one vote each
to p and q for both dots. After all pairs have voted, each dot takes the catalog star with the most votes.

**IN / OUT.** IN: `b` (N, 3) camera unit vectors; `table` (PairTable); `n_cat` = 2,847; `eps_arcsec` (e.g. 60).
OUT: `best` (N,) catalog index per dot (−1 if no votes), `votes` (N,) how many votes the winner got.

**Worked example by hand.** Three dots D1, D2, D3, measured angles D1–D2 = 5.00°, D1–D3 = 12.00°, D2–D3 = 9.00°.
The table (with eps) gives: 5.00° → {(A,B), (X,Y)}; 12.00° → {(A,C)}; 9.00° → {(B,C), (X,Z)}.
- Pair D1–D2 votes A,B,X,Y for D1 and for D2. D1–D3 votes A,C for D1 and D3. D2–D3 votes B,C,X,Z for D2 and D3.
- D1 tallies: **A 2**, B 1, C 1, X 1, Y 1 → **A**. D2: A 1, **B 2**, C 1, X 2, Y 1, Z 1 → tie B/X (real fields have ~27 dots, so ties
  this small don't survive). D3: A 1, B 1, **C 2**, X 1, Z 1 → **C**.
- The right star appears in *every* pair its dot is in. A wrong star needs repeated coincidences.

**Rough numbers.** With 27 dots, the true star gets ~26 votes. A random wrong star gets about
26 × (chance that one of its own ~170 neighbours lands inside the ±eps window) ≈ 26 × 0.14 ≈ 4 votes. The best wrong star out of 2,847
gets roughly 10–14. A fake dot (hot pixel) measured 14 against ≥ 21 for every real star.

**Why this way, and what was measured.**
- *Vectorised expansion.* Each detected pair matches `hi − lo` table rows. `np.repeat` + `cumsum` turns the list of
  ranges into one flat index array, and `np.bincount` tallies all votes at once. No Python loop over pairs.
- *Tried and rejected: counting each detected pair at most once per catalog star.* It removes coincidental
  double votes: misnamed stars drop from 0.11 % to 0 %, and the fake star stays below every real star in 100 % of frames instead of 98.5 %.
  But it needs a sort (`np.unique`) of up to millions of votes: **30 ms median, 768 ms worst** vs **2.1 ms / 20 ms** for plain.
  The verify step catches the rare misnamed star much more cheaply.
- *Close doubles are unresolvable.* β Mon A and B are 9.6″ apart (0.13 px). Both render as one dot, and pair angles can't tell them
  apart: they tie. The catalog has 36 pairs closer than 60″. Picking either moves the attitude by less than eps, so
  the tests accept "right star or its unresolvable partner".

**Interview question.** *"A hot pixel appears in the image. What happens in your identification?"*
Answer: it forms pairs with every real star, but its angles match the catalog only by coincidence. So it gets about as many votes
as a random wrong star (measured 14 vs ≥ 21 for real stars), and the real stars are still named correctly because their votes come from
real pairs. Its best guess is wrong, but after solving, the catalog star it was "named" as doesn't re-project onto it, so verification
drops it. Voting degrades gracefully, where a scheme that relies on one specific triangle of stars could be completely broken by one bad dot.

---

### `identify(b, table, eps_arcsec, min_match=4)`: lost in space, end to end (`tracker/identify.py`)

**What it does.** Dots in, attitude and star names out, or an honest "no solution" (`None`). Five steps:
1. **vote**: each dot gets its top-voted catalog star (previous entry).
2. **consistency check** (`consistent_names`): for every two named dots, does the catalog angle between their *names*
   match the angle *measured* between the dots (within eps)? Keep dots that agree with at least half as many others as the best one does.
3. **solve**: SVD on the consistent dots only.
4. **verify** (`verify`): rotate *every* dot into the sky with that attitude (r = Aᵀb, row form `b @ A`) and look
   for a catalog star within eps (k-d tree). Fewer than 4 matched → `None`.
5. **refine**: re-solve with all matched dots, then verify again.

**IN / OUT.** IN: `b` (N, 3) camera unit vectors; `table` (PairTable, which also holds the catalog vectors and k-d tree); `eps_arcsec`.
OUT: `Solution(A (3,3), det (K,) indices into b, cat (K,) catalog indices)` or `None`.

**Worked example (consistency check).** Four dots named A, B, C, X. Measured angles vs catalog angles between the names:
A–B 5.00° vs 5.00° ✓, A–C 12.00° vs 12.01° ✓, B–C 9.00° vs 9.00° ✓, X–A 7.3° vs 15.2° ✗, X–B 3.1° vs 21.0° ✗, X–C 6.6° vs 9.9° ✗.
Agreement counts: A 2, B 2, C 2, X 0. Threshold = max(2, 2/2) = 2 → keep A, B, C, drop X.
X was misnamed (maybe a hot pixel). Verification later checks whether it matches anything once the attitude is known.

**Why this way (measured).** Identification with vs without step 2, on real rendered frames:

| frames | with check | without check |
|---|---|---|
| 500, each with 2 hot pixels | **500 correct** | 0 correct, 500 no solution |
| 300, clean | **300 correct** | 218 correct, 82 no solution |

Without the check, one misnamed dot (a hot pixel, a noise false alarm, a blended pair) goes into the SVD with full weight and
drags the attitude off by arcminutes. Then even the correct stars fail the 60″ verify. The check costs one N² angle comparison.
- *Why verify at all?* Voting always produces *some* answer, even for a picture of nothing. Verify asks an independent question:
  "if this attitude is right, do the stars land where the catalog says?" Random dots give `None` in 20/20 tests.
- *Why ≥ 4 matches?* 3 points can always be fitted reasonably well by *some* rotation, so they make a weak check. The chance that a wrong attitude
  puts 4 dots within 60″ of catalog stars by luck is tiny (catalog density 0.069 stars/deg², a 60″ disc is 8.7·10⁻⁴ deg²,
  so it's ~6·10⁻⁵ per dot, to the 4th power).
- *Why refine?* The first solve uses only the consistent dots. After verify, more dots (including dimmer ones the vote missed) are
  confirmed, so solving again with all of them is more accurate.

**Interview question.** *"What's the difference between a 'no solution' and a 'wrong' answer, and which is worse?"*
Answer: "no solution" means the tracker knows it doesn't know. The spacecraft keeps propagating attitude with its gyros and
tries the next frame. A wrong answer is reported with confidence and fed to the controller, which may then slew the spacecraft
to the wrong attitude. Wrong is far worse, so the pipeline is designed to convert doubt into "no solution". That's the job of verification,
and the Monte Carlo counts the two separately.

---

### `wilson_interval(k, n)`: error bars on a success rate (`tracker/stats.py`)

**What it does.** "We succeeded 997 times out of 1,000" is a measurement with uncertainty. This gives the range the true
success rate probably lies in (95 % confidence).

**IN / OUT.** IN: `k` successes, `n` trials (ints or arrays). OUT: `(lo, hi)`, rates in [0, 1].

**Worked example by hand.** k = 1000, n = 1000, z = 1.96 (z² = 3.84):
- p = 1; denom = 1 + 3.84/1000 = 1.00384.
- centre = (1 + 3.84/2000) / 1.00384 = 1.00192 / 1.00384 = 0.99809.
- half = 1.96 / 1.00384 × √(0 + 3.84/4,000,000) = 1.9525 × 0.00098 = 0.00191.
- Interval = **[0.9962, 1.0000]**. So 1000/1000 means "at least 99.6 %", not "100 %".

**Why this way.** The obvious formula p ± 1.96·√(p(1−p)/n) gives [1, 1] for 1000/1000 (zero width, which is absurd), and it can go
below 0 or above 1 for rare events. Rare events (wrong answers, failures) are exactly what a validation campaign is about.
Wilson stays sensible at the edges, and the coverage test confirms it covers the truth ~95 % of the time.
Rule of thumb: 0 failures in n trials → the failure rate is below about 3/n (the "rule of three").

**Interview question.** *"You ran 1,000 trials and saw zero wrong answers. What do you claim?"*
Answer: not "zero wrong answers". I claim the wrong-answer rate is below ~0.4 % (Wilson upper bound 3.8/n, or the rule of three, 3/n), at 95 % confidence.
To claim < 0.01 % you need ~30,000+ clean trials. That's one reason Day 5 runs 100,000 frames, and why the GPU matters.

---

### Day 3 gate: the Monte Carlo and what it can (and can't) tell us (`experiments/montecarlo.py`)

**Result.** 1,000 random skies: **1000/1000 correct** at eps 31″, 45″ and 60″. With the error bar, that means "≥ 99.62 % at 95 % confidence".
Zero wrong and zero "no solution". Pointing is 1.7″ and roll 7″ (medians), the same as the Day 2 true-identity numbers.
So identification costs no accuracy: it finds the same stars we would have used anyway.

**Timing (one CPU core, ms/frame):** render 8.3 (50 %), detect 5.1 (31 %), identify 2.2 (19 %). The SVD is only 0.02 ms.
The two image stages touch all 1,048,576 pixels; identify touches ~27 stars. That is why Day 4 moves render and detect to the GPU:
the work is per-pixel and independent, which is what GPUs are built for.

**Why the eps question is still open.** A test only separates two options if at least one of them fails sometimes.
Here, nothing failed, so all three tie. The differences we expect (a looser eps lets in more coincidences, a tighter one drops real stars
when noise grows) only show up in harder conditions: more read noise, hot pixels, fewer stars. Clean images can't reveal them.
The small differences that did show are in the expected direction: smaller eps is faster (fewer candidates per lookup)
and very slightly more accurate (verify accepts fewer slightly-off stars).

**Interview question.** *"All your trials passed. Is the system done?"*
Answer: it means the failure rate is below ~0.4 % for *these* conditions. To find where it breaks, I push on it: more noise, fake stars,
sparse sky regions, many more trials. A validation campaign is about finding the edges of the envelope, not about confirming the easy cases.
That's Day 5.
