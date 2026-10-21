# Day 12: optical-navigation Monte Carlo campaign

96,000 trials (12 distances log-spaced from GEO to 1,000,000 km × Sun phase 0/45/90/135° × read noise 5 and 10, 1,000 per cell),
plus 48,000 true-attitude solves of the nominal-noise frames. Run on an Apple M1 Pro (10 processes, CPU backend) in 3 h 40 min;
seed 2027. Raw CSVs (144,000 rows) are in `results/opnav/` (not committed); `report.txt` and `summary.json` hold every number
in the figures. Thresholds (fixed before the run): correct = total position error < 1 % of the true distance; no solution =
tracker failure / fewer than 20 limb points / fit failure.

Figures: `outcomes_vs_distance.png`, `range_vs_distance.png`, `attitude_contribution.png`, `sphere_vs_ellipsoid.png`, `noise.png`
(`dark/` has the website versions).

## CPU rows = GPU rows (Colab Tesla T4, GPU backend, one full GEO phase-90 cell, 1,000 trials, 2,000 rows)

| column group | Mac (CPU) vs T4 (GPU) |
|---|---|
| trial, distance, phase, noise, attitude source, p, s, pixel radius, offset, reason, limb count, outcome | identical |
| attitude error (arcsec) | max difference 4.7e-5 |
| position (total / range / cross-range, km) | max difference 0.074 / 0.139 / 0.012 |
| sphere-model refit (km) | max difference 0.61 |
| median \|range error\| | 0.0011592 % on both |

The GPU Earth image equals the CPU one to float32 rounding (7e-4 counts); the two profile-fit passes amplify that to tens of
metres on a 42,164 km range (3 parts in 10⁶). All 1,000 outcomes agree. GPU precision check on the same T4: kernel in double
2.65 ms/frame vs float32 0.27 ms/frame; float32 moves limb points by ≤ 0.04 px.
