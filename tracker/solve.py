"""Attitude from matched star pairs: Wahba's problem solved by SVD.

Given unit vectors b_i (camera frame, from the image) and r_i (sky frame, from the
catalog) for the same stars, find the rotation A (sky -> camera) that minimises
    J(A) = 1/2 * sum_i w_i |b_i - A r_i|^2.

Everything here is float64: the GPU hands over star lists, the solve runs on the CPU.
"""
import numpy as np


def solve_attitude(b, r, w=None):
    """Best rotation A with b_i ~ A @ r_i (Wahba's problem, SVD method).

    b: (N, 3) camera-frame unit vectors. r: (N, 3) sky-frame unit vectors, same order.
    w: (N,) non-negative weights, or None for equal weights. Needs N >= 2 non-parallel.
    Returns A, a (3, 3) proper rotation (det +1) mapping SKY -> CAMERA.

        B = sum_i w_i b_i r_i^T = U S V^T
        A = U diag(1, 1, det U det V) V^T
    """
    b = np.asarray(b, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    w = np.ones(len(b)) if w is None else np.asarray(w, dtype=np.float64)
    B = (w[:, None] * b).T @ r                    # (3, 3) = sum of w_i * outer(b_i, r_i)
    U, S, Vt = np.linalg.svd(B)
    d = np.linalg.det(U) * np.linalg.det(Vt)      # +1 or -1: forbids a reflection
    return U @ np.diag([1.0, 1.0, d]) @ Vt
