"""Camera model: sky directions, attitudes, and the pinhole projection.

Conventions (the tests enforce every one of these):
  * Sky frame: x -> RA 0h on the equator, y -> RA 6h on the equator, z -> north pole.
  * Attitude A is a 3x3 rotation mapping SKY -> CAMERA:  b = A @ r.
    Its rows are the camera axes written in sky coordinates, so A[2] is the boresight.
  * The camera looks along its own +z. Image u (column) grows with camera x,
    image v (row) grows with camera y. Pixel centres sit at integers 0..1023.
  * At roll 0: north is up (smaller v) and east is left (smaller u),
    which is how the sky looks to someone standing under it.
"""
import numpy as np
from scipy.spatial.transform import Rotation

N_PIX = 1024                     # sensor is N_PIX x N_PIX
FOV_DEG = 20.0                   # full field of view, edge to edge
F_PIX = (N_PIX / 2) / np.tan(np.radians(FOV_DEG / 2))   # focal length in pixels
C_PIX = (N_PIX - 1) / 2                                  # image centre, in pixels


def radec_to_unit(ra_deg, dec_deg):
    """Arrays of RA and Dec in degrees -> (N, 3) array of unit vectors.
    r = [cos(dec) cos(ra), cos(dec) sin(ra), sin(dec)]"""
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack((x, y, z))


def random_attitude(rng):
    """A rotation matrix drawn UNIFORMLY over all orientations (Shoemake's method).
    Returns A (sky -> camera), the transpose of SciPy's from_quat(q).as_matrix()."""
    u1, u2, u3 = rng.random(3)
    q = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3),
                  np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    R = Rotation.from_quat(q).as_matrix()
    return R.T

def attitude_from_pointing(ra_deg, dec_deg, roll_deg):
    """Rotation matrix whose boresight points at (RA, Dec), with a roll about it.

    Hint: build the boresight z from (RA, Dec). Then find the local 'east' and
    'north' directions at that point on the sky (a cross product with the pole
    gives east). Choose camera x and y from those so the image is north-up,
    east-left at roll 0, and check that x cross y = z. Then rotate x and y by roll.
    """
    z = radec_to_unit(ra_deg, dec_deg).squeeze()
    east = np.cross([0, 0, 1], z)
    if np.linalg.norm(east) < 1e-8:
        # At the pole, east is undefined. Pick an arbitrary east direction.
        east = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(z, east)
    # Camera axes at roll 0: x = -east, y = -north, z = boresight
    x = -east
    y = -north
    # Rotate x and y about z by roll_deg
    roll_rad = np.radians(roll_deg)
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    x_rot = c * x + s * y
    y_rot = -s * x + c * y
    A = np.vstack((x_rot, y_rot, z))
    return A


def project(A, r):
    """Sky unit vectors r (N, 3) -> pixel coordinates (u, v) and a visibility mask.

    Visible means: in front of the camera (b_z > 0) and landing inside the sensor,
    i.e. -0.5 <= u < N_PIX - 0.5 and the same for v.
    """
    b = r @ A.T
    with np.errstate(divide='ignore', invalid='ignore'):
        u = F_PIX * (b[:, 0] / b[:, 2]) + C_PIX
        v = F_PIX * (b[:, 1] / b[:, 2]) + C_PIX
    visible = (b[:, 2] > 0) & (u >= -0.5) & (u < N_PIX - 0.5) & (v >= -0.5) & (v < N_PIX - 0.5)
    return u, v, visible


def unproject(u, v):
    """Pixel coordinates -> (N, 3) unit vectors in the CAMERA frame."""
    x = (u - C_PIX) / F_PIX
    y = (v - C_PIX) / F_PIX
    z = np.ones_like(x)
    b = np.column_stack((x, y, z))
    b /= np.linalg.norm(b, axis=1)[:, None]
    return b