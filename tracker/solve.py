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


RAD_TO_ARCSEC = 180 / np.pi * 3600


def attitude_error(A_est, A_true):
    """How far apart two attitudes are, in arcseconds: (total, pointing, roll), all >= 0.

    dA = A_est @ A_true.T is the small rotation that carries the TRUE camera axes onto
    the ESTIMATED ones, written in the camera frame.
      total:    rotation angle of dA.
      pointing: angle between the two boresights A_est[2] and A_true[2].
      roll:     twist of dA about the camera z-axis (boresight).
    For small errors, total^2 = pointing^2 + roll^2.

    Every angle uses atan2(sin, cos), never arccos: near 0, cos is ~1 - theta^2/2, and
    float64 cannot see theta below ~1e-8 rad (2 mas) that way; rounding can also push
    the arccos argument past 1 and return NaN.
    """
    dA = np.asarray(A_est, dtype=np.float64) @ np.asarray(A_true, dtype=np.float64).T
    # rotation angle: 2 sin(theta) = |vee(dA - dA^T)|,  2 cos(theta) = trace(dA) - 1
    vee = np.array([dA[2, 1] - dA[1, 2], dA[0, 2] - dA[2, 0], dA[1, 0] - dA[0, 1]])
    total = np.arctan2(np.linalg.norm(vee), np.trace(dA) - 1)
    # boresight angle: |z1 x z2| = sin, z1 . z2 = cos
    z_est, z_true = A_est[2], A_true[2]
    pointing = np.arctan2(np.linalg.norm(np.cross(z_est, z_true)), z_est @ z_true)
    # twist about z (exact for any dA, = 2 atan2(q_z, q_w) of its quaternion)
    roll = abs(np.arctan2(dA[1, 0] - dA[0, 1], dA[0, 0] + dA[1, 1]))
    return total * RAD_TO_ARCSEC, pointing * RAD_TO_ARCSEC, roll * RAD_TO_ARCSEC
