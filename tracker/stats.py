"""Statistics for reporting Monte Carlo results honestly."""
import numpy as np


def wilson_interval(k, n, z=1.96):
    """95 % (z = 1.96) Wilson score interval for a rate of k successes in n trials.

    Returns (lo, hi). Works on arrays. Unlike the textbook p +- z sqrt(p(1-p)/n), it never
    leaves [0, 1] and does not collapse to zero width when k = 0 or k = n.
        centre = (p + z^2/2n) / (1 + z^2/n)
        half   = z / (1 + z^2/n) * sqrt(p(1-p)/n + z^2/4n^2)
    """
    k = np.asarray(k, dtype=np.float64)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)
