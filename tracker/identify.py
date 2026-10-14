"""Lost-in-space star identification: which catalog star is each detected dot?

Offline: a table of every catalog pair closer than the FOV diagonal, sorted by angle.
Per image: geometric voting on star-pair angles, then solve, then verify.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

PAIR_MAX_DEG = 28.3      # FOV diagonal, sqrt(2) * 20 deg (exact corner-to-corner is 28.0)


@dataclass
class PairTable:
    angle: np.ndarray    # (M,) separation in radians, sorted ascending
    i: np.ndarray        # (M,) catalog index of the first star  (i < j)
    j: np.ndarray        # (M,) catalog index of the second star


def angle_between(a, b):
    """Row-wise angle (radians) between unit vectors a and b, both (M, 3). atan2 form."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1))


def build_pair_table(r, max_sep_deg=PAIR_MAX_DEG):
    """All catalog pairs closer than max_sep_deg, sorted by angle. r: (N, 3) sky unit vectors.

    A k-d tree finds the pairs: for unit vectors, angle < theta <=> chord |r_i - r_j| < 2 sin(theta/2).
    """
    chord = 2 * np.sin(np.radians(max_sep_deg) / 2)
    ij = cKDTree(r).query_pairs(chord, output_type="ndarray")      # (M, 2), i < j
    angle = angle_between(r[ij[:, 0]], r[ij[:, 1]])
    order = np.argsort(angle, kind="stable")
    return PairTable(angle=angle[order], i=ij[order, 0], j=ij[order, 1])


ARCSEC = np.pi / (180 * 3600)    # one arcsecond in radians


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
