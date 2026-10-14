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
