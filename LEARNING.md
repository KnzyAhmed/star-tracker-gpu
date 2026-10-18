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

---

### Stress test: choosing eps from data (and why my first guess was wrong)

**What was done.** The same Monte Carlo, made harder: read noise ×2, ×3, ×4 (fewer faint stars detected, worse centroids) and
hot pixels (fake dots). Six eps values were tried on identical images.

**Result.** On easy images, every eps works. Under stress, **wide eps collapses**: at noise 15 + 10 hot pixels, 60″ gets 61 % correct and 20–25″ get 90–91 %.
Below 20″, a different failure starts: at noise 20, 15″ drops to 81 % against 86 % at 25″. The best region is **20–25″**. Recommendation: 25″.
In every run, wrong answers were **0**. Every failure was an honest "no solution", so verification did its job.

**Why wide eps fails (signal vs noise in the vote).**
- *Signal:* a true star gets ≈ 1 vote from each other *real* star → (real dots − 1). That doesn't depend on eps.
- *Noise:* each detected pair matches ~(pairs per arcsec) × 2·eps catalog pairs by coincidence. So coincidence votes grow **in proportion to eps**,
  and with the number of dots, fakes included.
- Worked numbers: 15 real stars + 20 hot pixels = 35 dots. A true star's signal is ~14 votes. At eps 60″, a random wrong star expects ~34 × 0.14 ≈ 5 votes,
  and the luckiest of 2,847 wrong stars reaches ~14–15, which ties or beats the truth. At eps 25″, the expectation drops to ~2, and the luckiest gets ~8: the truth wins.
- *Too narrow:* if eps is smaller than the real pair-angle errors, true pairs fall outside the window and the *signal* shrinks.
  That's the 15″ drop at noise 20.

**What I got wrong, and the lesson.** I argued "missing a true pair is worse than letting in a coincidence". That reasoning looked only at
the signal side. The real trade-off is signal *against* noise, and on our images the noise side dominates. The argument sounded fine; the
measurement disagreed. That's why the decision waited for data.

**Interview question.** *"How did you choose your matching tolerance?"*
Answer: not from the centroid error alone. I swept it on identical simulated images under nominal and stressed conditions and counted correct, wrong
and no-solution outcomes with error bars. The optimum is where the vote's signal (independent of eps) best beats the coincidence noise (∝ eps),
while staying above the measurement error. Here that's 20–25″, about 3–4× the median pair-angle error. Wider tolerances looked
"safer" but cut the stressed success rate from 90 % to 61 %.

---

## Day 4: GPU

### How we do GPU work without an NVIDIA GPU
The Mac can't run CUDA. So: (1) every GPU step has a NumPy twin doing the *same arithmetic*, tested on the Mac;
(2) the CUDA kernels live in `tracker/kernels/*.cu` and are driven only from `tracker/gpu.py` (CuPy);
(3) GPU tests begin with `pytest.importorskip("cupy")`, so on the Mac they are skipped, not failed;
(4) `notebooks/colab_run.ipynb` runs them on a free Colab NVIDIA GPU.

### `pixel_normals(seed, frame)` and `hash32(x)`: counter-based noise (`tracker/noise.py`)

**What it does.** It gives every pixel of every frame its own Gaussian noise value, computed only from (seed, frame, pixel).
The same value comes out on the CPU and on the GPU, in any order.

**IN / OUT.** IN: `seed`, `frame` (ints ≥ 0, fit in 32 bits). OUT: (1024, 1024) float32 standard normals, indexed [v, u].
Multiply by `READ_NOISE` to get counts.

**Why not `rng.normal`?** A normal RNG is a *sequence*: number k comes after numbers 0..k−1. A million GPU threads running at once
can't each "take the next number". A counter-based RNG is a function, value = f(key, counter), so thread k just computes f(key, k).
This is the same idea as Philox/Threefry, which CuPy and JAX use internally.

**The recipe.**
1. key = hash32(hash32(seed + 0x9E3779B9) + frame); h1 = hash32(key + 2·pixel); h2 = hash32(key + 2·pixel + 1).
2. Keep the top 24 bits: u1 = (h1 >> 8) + 1 over 2²⁴, in (0, 1]; u2 = (h2 >> 8) / 2²⁴, in [0, 1). 24 bits because float32 has a 24-bit mantissa,
   so the integer-to-float conversion is *exact* on both machines.
3. Box-Muller: z = √(−2 ln u1) · cos(2π u2).

**Worked example (the bit tricks), by hand.** Say h1 = 0x00000000 (the worst case). Then h1 >> 8 = 0, plus 1 → u1 = 1/16,777,216 = 5.96·10⁻⁸.
ln u1 = −16.64, so √(−2 · −16.64) = √33.27 = **5.77**. That is the largest |z| this generator can ever produce (`MAX_SIGMA`).
Without the "+1", u1 = 0 → ln 0 = −∞ → z = ∞, and one pixel would be infinite. The cap at 5.77σ cuts off a tail of probability ~8·10⁻⁹. That is
harmless for us: our threshold is 5σ.

**Two traps we hit / avoided.**
- *hash32(0) = 0.* Without the golden-ratio offset, seed 0, frame 0, pixel 0 would always get u1 = 2⁻²⁴ and come out at −4.84σ.
  The test checks this.
- *Speed on the CPU.* 16 ms/frame vs 6.9 ms for `rng.normal`. So the **CPU benchmark keeps `rng.normal`** (the fastest honest CPU
  method), and counter-based noise is only used for CPU-vs-GPU equality checks. Otherwise the GPU speed-up would be inflated.

**Measured.** Mean, std, 1σ and 3σ tail fractions match a standard normal. Frames, seeds and neighbouring pixels are uncorrelated (|r| < 0.005).
`detect` on noise-only frames: 0.25 false stars/frame (theory 0.30).

**Interview question.** *"How do you generate random numbers on a GPU reproducibly?"*
Answer: with a counter-based generator. Each thread hashes (seed, stream id, its own index), so the result doesn't depend on thread scheduling
or block size, and a CPU reference can reproduce it bit for bit. A sequential generator would need a separate state per thread
(e.g. cuRAND states with skip-ahead), and those results are tied to the launch configuration.

---

### `star_windows(u, v)` and `render(..., noise_key=(seed, frame))` (`tracker/render_cpu.py`)

**What changed.** (1) The ±5 px box each star is drawn in is now computed by one function, `star_windows`. Its integer bounds are
sent to the GPU as well, so CPU and GPU draw *exactly* the same pixels. (2) `render` can use the counter-based noise
(`noise_key=(seed, frame)`) instead of `rng`. That image is what the GPU must reproduce.

**IN / OUT.** `star_windows`: IN u, v (N,) pixels; OUT four (N,) int32 arrays u_min, u_max, v_min, v_max (inclusive, clipped to 0..1023).

**Worked example.** A star at u = 0.2: floor(0.2 − 5) = floor(−4.8) = −5 → clipped to **0**; ceil(0.2 + 5) = ceil(5.2) = **6**. So columns 0..6.
A star at u = 1023.0: floor(1018.0) = **1018**, ceil(1028.0) = 1028 → clipped to **1023**.

**Why compute the box on the host?** If the GPU computed floor(u − 5) itself in float32, a star whose u − 5 lands within float32
rounding of an integer could get a box one pixel different. At the box edge (5 px out), a bright star still contributes ~4.5 counts,
so that would be a visible CPU/GPU mismatch. Integers computed once can't disagree.

**Check.** The refactored renderer is **bit-identical** to the previous one on 30 random skies.

---

### `pack_frames(frames)` / `project_frames(As, catalog)`: batching for the GPU (`tracker/batch.py`)

**What it does.** The GPU renders 128 frames in one launch, and each frame has a different number of stars. `pack_frames` concatenates all the
star lists into flat arrays and records where each frame starts (`offsets`). `project_frames` produces those per-frame lists from attitudes,
the same way `render_sky` does.

**IN / OUT.** IN: list of F tuples (u, v, flux). OUT: `StarBatch` with `offsets` (F+1,) int32, and per star (S total) `du, dv, amp` float32
and `box` (S, 4) int32.

**Worked example.** Frames with 2, 0 and 1 stars → offsets = [0, 2, 2, 3]. The GPU block for frame 1 reads offsets[1] = 2 and offsets[2] = 2,
so it has stars 2..1, which is none. Frame 2 has stars 2..2. This is the "CSR" (compressed sparse row) layout.

**Why relative positions (`du = u − u_min`)?** float32 has 24 bits of precision. Near 1000 the step between representable numbers is
6·10⁻⁵ px, so 1000.7 is stored 1.2·10⁻⁵ px off. Near 5.7 the step is 1000× smaller (measured error 1.9·10⁻⁷ px). The GPU computes
dx = (x − u_min) − du, where every number involved is small. The obvious approach, sending u in float32 and computing x − u on the GPU,
shifts the blob slightly, and the brightness at the box edge then disagrees with the CPU by ~10⁻⁴ relative.

**Interview question.** *"Your GPU and CPU results differ in the 4th digit. Where do you look first?"*
Answer: at large numbers stored in float32 and then subtracted (catastrophic cancellation). Positions, timestamps and global indices are the usual
suspects. The fix is to subtract on the host in float64 and send small offsets, or keep that one quantity in float64.

---

### `render.cu` / `gpu.render_batch`: one thread per pixel (`tracker/kernels/render.cu`, `tracker/gpu.py`)

**What it does.** It draws 128 frames at once. Thread (x, y) of block z owns pixel [z, y, x]. It starts from the background, adds every star whose box
contains it, then adds the counter-based noise. The result stays on the GPU as a CuPy array.

**IN / OUT.** IN: a `StarBatch` (offsets, du, dv, amp, box), `seed`, `frame_ids`. OUT: (F, 1024, 1024) float32 CuPy array.
Launch: block 16×16 = 256 threads, grid 64 × 64 × F. That's 1,048,576 threads per frame, 134 M for a 128-frame batch.

**Worked example: who draws what.** A star at u = 300.3 has box columns 295..306. The thread for column x = 297 computes
dx = (297 − 295) − du = 2 − 5.3 = −3.3 px, so it adds amp · exp(−(3.3² + dy²)/(2·1.2²)). The thread for x = 310 is outside the box and adds nothing.
This is exactly the CPU loop, turned inside out: the CPU loops over stars and touches their pixels, while the GPU gives each pixel a thread that loops over stars.

**Why shared memory, in chunks of 64.** All 256 threads in a block need the *same* star list. One cooperative load into shared memory
(an on-chip cache shared by the block) replaces 256 separate reads of each star. Chunks of 64 keep that tile small and still handle any star count:
the test uses 150 stars, which takes 3 chunks.
**The trap:** `__syncthreads()` is a barrier every thread of the block must reach. So the "is my pixel on the sensor?" early exit happens only *after* the loop.
Returning earlier could leave the other threads waiting at the barrier forever.

**Measured on a Tesla T4 (Colab, 70 W).** GPU vs CPU, noise off: max difference **4.9·10⁻⁴ counts**, which is one float32 rounding step for values
4,096–8,192; 0.06 % of pixels differ at all. Render takes **0.243 ms/frame** against 8.3 ms on one CPU core.

**Interview question.** *"Why one thread per pixel rather than one thread per star?"*
Answer: one thread per star would need many threads writing into the same pixels where blobs overlap. That means atomic adds, which are slow,
and their order isn't fixed, so float sums wouldn't be reproducible. One thread per pixel only *reads* the stars and writes its own pixel once:
no races, and a deterministic order (stars in list order, like the CPU). The cost is that each thread checks every star's box, which is cheap for ≤ ~100 stars.

### `detect.cu` / `gpu.detect_batch`: threshold, local max, centroid on the GPU

**What it does.** Same contract as `detect_cpu.detect`, for a whole batch, without the images leaving the GPU.
Each 16×16 block copies a **20×20 tile** (its pixels plus a 2-px halo) into shared memory. Each thread then tests its pixel:
threshold, 3×3 local max with the same tie-break, and the 5×5 centroid. A found star claims an output slot with **`atomicAdd`** on its frame's counter.

**IN / OUT.** IN: (F, 1024, 1024) CuPy images, background, read_noise. OUT: a list of F `(u, v, flux)` NumPy tuples, brightest first.

**Worked example: why a 2-px halo.** A thread at the left edge of its tile (threadIdx.x = 0) needs the 5×5 window, which reaches columns −2..+2.
The two columns to its left belong to the *neighbouring* block, so the tile stores them too: 16 + 2 + 2 = 20 wide.
On the sensor edge, those halo pixels are set to OFF_SENSOR (−FLT_MAX). That value loses every local-max comparison and counts as zero weight, which is
exactly what the CPU's −inf and background padding do.

**Three design points.**
- *atomicAdd makes the order random*, so the host sorts by flux (ties by position). Tests compare by position, never by order.
- *Overflow fails loudly.* The counter keeps counting past capacity (256, not the spec's 64). If more stars arrive than slots, *which* ones got a slot would
  depend on thread timing, so `detect_batch` raises instead of silently returning an arbitrary subset.
- *Same threshold, bit for bit.* NumPy 2 compares a float32 image with a Python-float threshold *in float32*, but an `np.float64` threshold
  would switch it to float64. `detect_cpu` now makes this explicit (`np.float32(...)`), and the GPU gets that same float32 value.

**Measured (T4).** Same star count in 16/16 frames; max centroid difference **3.2·10⁻⁷ px**.
Render + detect + star-list download: **0.318 ms/frame**, against 13.4 ms per frame for both stages on one CPU core.
(A first look only: the fair benchmark against all CPU cores is Day 6.)

**Interview question.** *"Your GPU detector finds the same stars as the CPU but in a different order each run. Is that a bug?"*
Answer: no. Slots are claimed with atomicAdd, and which thread gets there first depends on scheduling. The *set* is deterministic; the order isn't.
So the host sorts into a canonical order before anything downstream uses the list. The real bug to guard against is overflow: then the *set* would depend on timing too.
That's why it raises an error.

---

## Day 5: the campaign

### `pointing_from_attitude(A)`: attitude back to (RA, Dec, roll) (`tracker/camera.py`)

**What it does.** It's the inverse of `attitude_from_pointing`: given the 3×3 attitude, it says where the camera looks (RA, Dec of the boresight)
and how it is rolled. Every campaign row stores this, and the failure sky map plots it.

**IN / OUT.** IN: A (3, 3), sky → camera. OUT: `(ra_deg in [0, 360), dec_deg in [−90, 90], roll_deg in (−180, 180])`.

**Worked example.** Boresight z = A[2] = (0, 0.7071, 0.7071). RA = atan2(0.7071, 0) = **90°**; Dec = atan2(0.7071, √(0² + 0.7071²)) = **45°**.
For roll: build A0 = attitude_from_pointing(90, 45, 0). If the camera x-axis is A[0] = 0.866·A0[0] + 0.5·A0[1], then
roll = atan2(0.5, 0.866) = **30°**.

**Why this way.** atan2 for every angle (Dec too, not arcsin), so values at the poles and near ±90° stay accurate. Roll is measured
*relative to the function we're inverting*, so the round trip is exact by construction, even at the poles where "east" is arbitrary.
Edge case found: `(−1e−17) % 360` returns exactly 360.0 in floating point, so it's mapped to 0.

**Interview question.** *"Why store RA/Dec/roll instead of the 3×3 matrix or a quaternion?"*
Answer: for analysis. RA/Dec go straight onto a sky map, and roll is a physical angle you can bin. For computation the matrix (or a quaternion)
is better: no singularities, and composing rotations is just multiplication. RA/Dec/roll are Euler-like angles, which become ill-defined at the poles.

---

### `tracker/campaign.py`: `run_batch`, `run_condition`, and `experiments/campaign.py`

**What it does.** It runs thousands of trials under one *condition* (noise level, number of hot pixels) and writes one CSV row per trial:
`trial, read_noise, hot_pixels, ra, dec, roll, n_in_view, n_detected, outcome, total, pointing, roll_err, n_matched`.
`run_condition` appends a batch at a time and skips trials already in the file, so an interrupted Colab run just resumes.

**IN / OUT.** `run_batch(trials, seed, read_noise, hot_pixels, catalog, table, backend)` → (len(trials), 13) float64.
Errors are in arcsec (NaN for "no solution"); outcome 0/1/2 = correct/wrong/no solution.

**Key idea 1: a trial is a pure function of (seed, trial number).** Attitude from `default_rng([seed, trial, 0])`, hot pixels from
`default_rng([seed, trial, 1])`, noise from the counter-based hash of (seed, trial). Worked example: trial 4 run alone gives the *same row* as
trial 4 run as the second item of the batch [3, 4]. The test checks exactly that. With a single shared generator, trial 4's sky would depend on
how many random numbers trial 3 used, so resuming or reordering would silently change results.

**Key idea 2: common random numbers.** Every condition uses the same seed, so trial k is the same sky at noise 5 and at noise 20. Comparing
the conditions then measures the *effect of noise*, not luck in which skies were drawn. That is why `noise_5` is not run separately: it's the first
10,000 baseline trials.

**The plan** (`experiments/campaign.py`, seed 2026): baseline 100,000 trials; noise sweep 2.5–20 (7 levels × 10,000); hot pixels 5–80
(5 levels × 10,000). Total 220,000 frames.

**Interview question.** *"Your noise sweep shows 98 % at noise 15 and 97 % at noise 17.5, but the error bars overlap. Can you claim noise 17.5 is worse?"*
Answer: with independent samples, not on that comparison alone. Here the skies are *paired* (common random numbers): I can count the trials that
pass at 15 but fail at 17.5 (and vice versa) and test that difference directly (McNemar's test). That is far more sensitive than comparing two
overlapping intervals. Pairing removes the sky-to-sky variation from the comparison.

---

### Day 5 results: what 220,000 simulated skies say (`docs/results/day5/`)

**Headline.** Baseline: **100,000 / 100,000 correct**, so the success rate is ≥ 99.996 % at 95 % confidence. **Zero wrong answers in all 220,000 trials**,
under every noise and hot-pixel setting: the wrong-answer rate is < 0.0017 % (95 %). Every failure was an honest "no solution".
Accuracy (correct baseline trials): pointing **1.58″** median / 4.6″ p99, roll **6.5″** median / 28″ p99.

**Robustness** (10,000 trials per point, % correct): read noise 10 → 99.99; 12.5 → 99.59; 15 → 98.06; 17.5 → 93.28; 20 → 86.35.
Hot pixels 10 → 99.99; 20 → 99.83; 40 → 98.11; 80 → 80.54.

**Three things to understand.**
1. *Why 100,000 trials?* With 0 failures in n trials, you can only claim the failure rate is below ≈ 3/n (rule of three). 1,000 trials → < 0.3 %;
   100,000 → < 0.003 %. Rare-event claims need many trials. That is the whole reason for the GPU: 220,000 frames took 19 min on a T4.
2. *The tracker fails where the sky is empty.* At noise 20, failures are 2.5 % within 15° of the Milky Way plane (39 stars in view on average), but 29 %
   within 30° of the galactic poles (18 stars). Success vs stars in view: 0 % with < 10 stars, 41 % with 10–14, 100 % with 40+.
   The vote needs enough *real* stars to beat the coincidences (the signal-vs-noise argument from the eps stress test), and noise hides the faint ones first.
3. *It fails safe.* As noise rises, failures grow, but they are all "no solution", never a wrong attitude. The consistency check plus the ≥ 4-star verify
   turn doubt into silence. For a spacecraft, that is the property that matters most: a wrong attitude can drive a bad manoeuvre, while "no solution" just
   means "use the gyros and try the next frame".

**Interview question.** *"How would you make your tracker work in the sparse regions near the galactic poles?"*
Answer: give the vote more real stars. Use a fainter catalog limit with a more sensitive camera (more stars per field); widen the field of view; use a
lower detection threshold with a better centroider (more faint stars detected, at the cost of more false dots); or use more distinctive patterns
(triangles) so fewer stars are needed. Operationally, a spacecraft can also avoid pointing the tracker at the poles, or use two tracker heads.

---

## Day 6: benchmark and README

### `experiments/benchmark.py`: a fair CPU vs GPU comparison

**What it does.** It times the same skies through both pipelines. The CPU side runs on 1 core, stage by stage, then on all cores with one process
per core. The GPU side runs batches of 128: prepare → pack + upload → render → detect + download → identify on CPU cores.

**The fairness rules, and why each matters.**
- *Best CPU code, all cores.* A GPU vs single-core comparison inflates the speed-up by the core count. Here: 37× vs 1 core, but only **5.7× vs 10 cores**.
- *Plug in.* On battery the M1 Pro ran ~20 % slower (10-core render + detect 2.33 vs 1.85 ms), which would have flattered the GPU.
- *Fastest CPU noise.* The GPU-matching counter noise is 2.3× slower on a CPU, so the baseline uses `rng.normal`.
- *Warm up first.* The first GPU launch compiles kernels (NVRTC) and allocates memory. Two warm-up batches are discarded.
- *Synchronise before stopping the clock.* GPU calls return immediately, and the work happens later. Without a synchronise you time the *launch*, not the work.
  CUDA events (`start.record() … end.synchronize()`) time the kernels themselves.
- *Include transfers.* Upload of star lists and download of detections are counted. (Images never move, which is the design point.)
- *Name the hardware.* Apple M1 Pro (8P + 2E cores); Tesla T4 70 W, driver 580.82.07; Colab host Xeon 2.0 GHz with 2 vCPUs.

**Worked example: Amdahl's law with our numbers.** One M1 Pro core spends 12.1 ms on render + detect and 1.6 ms on everything else (13.8 ms total).
Make render + detect infinitely fast and the frame still takes 1.6 ms: at best 13.8 / 1.6 ≈ **8×**, however good the GPU is.
On Colab the "everything else" (identify) runs on 2 slow vCPUs at 2.5 ms/frame. So the T4 pipeline (3.0 ms) is *slower* end to end than the
10-core Mac (2.0 ms), even though its image stages are 5.7× faster.

**Interview question.** *"Your GPU kernel is 37× faster. Why is your end-to-end speed-up so much smaller?"*
Answer: 37× is against one CPU core. The fair comparison (all cores) is 5.7×, and the end-to-end gain is capped by Amdahl's law: identification
stayed on the CPU, and on the 2-vCPU GPU host it became 84 % of the time. Next steps: run identify on more cores (or on the GPU), and overlap it
with the GPU's next batch (pipelining), so the CPU and GPU work at the same time instead of taking turns.

---

## Day 7: the live demo website

### How the real Python tracker runs in a web page (`web/`, `tools/build_site.py`)

**What it does.** The page runs the project's actual `tracker/*.py` modules, not a JavaScript re-write, inside the visitor's browser. Pyodide is
CPython compiled to WebAssembly, and it ships NumPy and SciPy. A Web Worker (a background thread) loads Pyodide, fetches the tracker files and the
catalog listed in `manifest.json` into Pyodide's virtual file system, and calls `web/bridge.py`. The page itself only draws.

**IN / OUT** (`bridge.run_frame(ra, dec, roll, read_noise, hot_pixels, seed)`). IN: the hidden true pointing (degrees) and the conditions.
OUT: the image as 1,048,576 bytes (asinh-stretched grey levels), every detected dot (u, v, flux, votes, verified, name), the true star positions,
per-stage times, the estimate, the errors and the outcome. The tracker itself only ever sees the image.

**Worked example (measured in Chrome on the M1 Pro).** Orion: render 207 ms, detect 46 ms, identify 83 ms → 343 ms total, against ~14 ms for the
same code natively. WebAssembly NumPy is ~25× slower here, and still interactive. A sparse field (20 dots) took 132 ms.

**Why this way.**
- *Same code, not a port.* A JavaScript re-implementation could drift from the tested Python. Here the demo can't disagree with the tests.
  `test_build_site` even runs the bundle in a clean Python process to prove it is self-contained.
- *Web Worker.* The ~0.3 s computation runs off the main thread, so dragging the globe never freezes.
- *Globe maths mirrors `attitude_from_pointing`.* So the 20° box on the globe and the camera image show the same sky the same way round.
  Dragging moves the boresight along the camera's own x/y axes, which works at any roll and over the poles.
- *No GPU on the page.* Browsers can't run our CUDA kernels; the Results tab shows the stored T4 numbers and says so.

**Interview question.** *"Your demo runs in the browser. How do you know it does the same thing as your tested code?"*
Answer: it *is* the tested code. The build script copies the same `tracker/*.py` files into the site, and Pyodide runs them unchanged. A test builds the
bundle and runs it in an isolated Python process, and the page's Python entry point (`bridge.py`) has its own unit tests.
