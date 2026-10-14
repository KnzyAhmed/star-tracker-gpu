"""Lost-in-space star identification: which catalog star is each detected dot?

Offline: a table of every catalog pair closer than the FOV diagonal, sorted by angle.
Per image: geometric voting on star-pair angles, a consistency check on the names,
then solve, then VERIFY by re-projecting the catalog.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from tracker.solve import solve_attitude

PAIR_MAX_DEG = 28.3      # FOV diagonal, sqrt(2) * 20 deg (exact corner-to-corner is 28.0)


@dataclass
class PairTable:
    angle: np.ndarray    # (M,) separation in radians, sorted ascending
    i: np.ndarray        # (M,) catalog index of the first star  (i < j)
    j: np.ndarray        # (M,) catalog index of the second star
    r: np.ndarray        # (N, 3) the catalog unit vectors the table was built from
    tree: cKDTree        # k-d tree over r, for "nearest catalog star" in verify


def angle_between(a, b):
    """Row-wise angle (radians) between unit vectors a and b, both (M, 3). atan2 form."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1))


def build_pair_table(r, max_sep_deg=PAIR_MAX_DEG):
    """All catalog pairs closer than max_sep_deg, sorted by angle. r: (N, 3) sky unit vectors.

    A k-d tree finds the pairs: for unit vectors, angle < theta <=> chord |r_i - r_j| < 2 sin(theta/2).
    """
    chord = 2 * np.sin(np.radians(max_sep_deg) / 2)
    tree = cKDTree(r)
    ij = tree.query_pairs(chord, output_type="ndarray")            # (M, 2), i < j
    angle = angle_between(r[ij[:, 0]], r[ij[:, 1]])
    order = np.argsort(angle, kind="stable")
    return PairTable(angle=angle[order], i=ij[order, 0], j=ij[order, 1], r=r, tree=tree)


ARCSEC = np.pi / (180 * 3600)    # one arcsecond in radians
EPS_ARCSEC = 25.0                # matching tolerance, chosen by the Day 3 stress test (BUILD_LOG)


def vote(b, table, n_cat, eps_arcsec):
    """Geometric voting: guess the catalog star behind each detected star.

    b: (N, 3) camera-frame unit vectors of the detected stars. table: PairTable.
    n_cat: number of catalog stars. eps_arcsec: angle tolerance.
    Returns (best, votes), both (N,) int: best catalog index per detected star (-1 if no
    votes at all) and how many votes it got.

    Every detected pair (a, c) with angle theta looks up all catalog pairs (p, q) with
    |angle - theta| <= eps. The angle can't tell which end is which, so each such row
    votes for a->p, a->q, c->p, c->q. The true star collects a vote from almost every
    pair it is in; wrong stars only collect scattered coincidences.
    """
    n = len(b)
    if n < 2:
        return np.full(n, -1), np.zeros(n, dtype=int)
    a, c = np.triu_indices(n, k=1)                                  # all detected pairs
    theta = angle_between(b[a], b[c])
    eps = eps_arcsec * ARCSEC
    lo = np.searchsorted(table.angle, theta - eps, side="left")
    hi = np.searchsorted(table.angle, theta + eps, side="right")
    # expand each detected pair into its table rows lo..hi-1, without a Python loop
    count = hi - lo
    pair = np.repeat(np.arange(len(a)), count)                      # which detected pair
    start = np.cumsum(count) - count                                 # first slot of each pair
    row = lo[pair] + np.arange(count.sum()) - start[pair]            # table row
    p, q = table.i[row], table.j[row]
    a, c = a[pair], c[pair]
    keys = np.concatenate([a * n_cat + p, a * n_cat + q, c * n_cat + p, c * n_cat + q])
    tally = np.bincount(keys, minlength=n * n_cat).reshape(n, n_cat)
    best = tally.argmax(axis=1)
    votes = tally[np.arange(n), best]
    best[votes == 0] = -1
    return best, votes


@dataclass
class Solution:
    A: np.ndarray        # (3, 3) attitude, sky -> camera
    det: np.ndarray      # (K,) indices into b of the verified dots
    cat: np.ndarray      # (K,) catalog index of each verified dot


def consistent_names(b, names, eps_arcsec):
    """Which dots' names agree with each other? Returns a (N,) bool mask.

    For every two named dots, compare their measured angle with the catalog angle between
    their names. A correctly named dot agrees with (almost) every other correct one; a
    misnamed dot agrees with almost none. Keep dots that agree with at least half as many
    others as the best-agreeing dot does, and with at least 2.
    """
    n = len(b)
    a, c = np.triu_indices(n, k=1)
    ok = np.abs(angle_between(b[a], b[c]) - angle_between(names[a], names[c])) <= eps_arcsec * ARCSEC
    agree = np.bincount(a[ok], minlength=n) + np.bincount(c[ok], minlength=n)
    return agree >= max(2, agree.max() / 2)


def identify(b, table, eps_arcsec=EPS_ARCSEC, min_match=4):
    """Lost in space: detected dots -> attitude and star names, or None ("no solution").

    b: (N, 3) camera unit vectors, brightest first. table: PairTable.
      1. vote: each dot gets its top-voted catalog star.
      2. check: keep only dots whose names are geometrically consistent with each other.
      3. solve: SVD on the consistent dots.
      4. verify: rotate EVERY dot into the sky with that attitude; a dot is matched if a
         catalog star lies within eps. Fewer than min_match -> None.
      5. refine: re-solve with all matched dots and verify once more.
    """
    named = vote(b, table, len(table.r), eps_arcsec)[0]
    use = named >= 0
    if use.sum() < 3:
        return None
    idx = np.flatnonzero(use)
    keep = idx[consistent_names(b[idx], table.r[named[idx]], eps_arcsec)]
    if len(keep) < 3:
        return None
    A = solve_attitude(b[keep], table.r[named[keep]])
    det, cat = verify(b, A, table, eps_arcsec)
    if len(det) < min_match:
        return None
    A = solve_attitude(b[det], table.r[cat])             # refine with every matched dot
    det, cat = verify(b, A, table, eps_arcsec)
    if len(det) < min_match:
        return None
    return Solution(A=A, det=det, cat=cat)


def verify(b, A, table, eps_arcsec):
    """Rotate every dot into the sky (r = A^T b, row form b @ A) and look for a catalog star
    within eps. Returns (det, cat): indices of the matched dots and their catalog stars."""
    chord = 2 * np.sin(eps_arcsec * ARCSEC / 2)
    dist, nearest = table.tree.query(b @ A, distance_upper_bound=chord)
    det = np.flatnonzero(np.isfinite(dist))              # unmatched dots come back as inf
    return det, nearest[det]
