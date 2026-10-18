"""World geometry for optical navigation: where the spacecraft, Earth and the Sun are.

Frame: the sky frame of camera.py (J2000 equatorial), Earth's centre at the origin, km.
Earth's rotation axis is the sky z axis (true by definition of the equatorial frame; precession ignored).
Every physical constant and modelling assumption of Phase 2 lives here, with its source.
"""
import numpy as np

from tracker.camera import C_PIX, F_PIX, N_PIX

# WGS 84 ellipsoid (NIMA TR8350.2): semi-major axis a and semi-minor axis b = a (1 - 1/298.257223563)
R_EQ = 6378.137          # km, equatorial radius
R_POL = 6356.752         # km, polar radius (6356.7523...)
# Brightness of a fully lit Earth pixel (normal facing the Sun), in sensor counts. An assumption of the
# single-exposure model: far above the 5-sigma star threshold (background 20 + 5*5 = 45 counts) and below
# the peak of a magnitude-0 star (F0 / (2 pi sigma^2) ~ 6,600). Real systems often use a separate short
# exposure for the Earth; see the README limitations.
EARTH_LEVEL = 3000.0
# Sanity values from the plan: |p| = 42,164 km (GEO) -> rho 8.69 deg, ~444 px; 384,400 km -> 0.95 deg, ~48 px.


def _unit_sphere(rng):
    """One direction uniformly distributed over the sphere (Marsaglia: uniform z, uniform azimuth)."""
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    s = np.sqrt(1.0 - z * z)
    return np.array([s * np.cos(phi), s * np.sin(phi), z])


def sample_position(rng, r_min, r_max):
    """Spacecraft position p (km): direction uniform on the sphere, distance log-uniform in [r_min, r_max].
    Log-uniform gives every factor-of-two shell the same number of trials, so the campaign covers GEO to
    lunar distance evenly in the quantity that matters (apparent size scales like 1/distance)."""
    r = np.exp(rng.uniform(np.log(r_min), np.log(r_max)))
    return r * _unit_sphere(rng)


def sample_sun(rng):
    """Sun direction s: unit vector from Earth to the Sun, uniform over the sphere (every season and
    hour, no ecliptic constraint; the Sun is at infinity, so sunlight is parallel)."""
    return _unit_sphere(rng)


def earth_in_view(p, A, R=R_EQ, n_pix=N_PIX):
    """Where Earth appears for a camera at p (km) with attitude A (sky -> camera).

    Returns (visible, (u, v), rho, pixel_radius):
      visible       True if the disc overlaps the sensor (also partially); False otherwise
      (u, v)        pixel position of Earth's centre (may be off-sensor)
      rho           apparent angular radius, arcsin(R / |p|), radians
      pixel_radius  F_PIX * tan(rho): the disc radius in pixels when Earth is on-axis
    Earth is treated as a sphere of radius R here (fine for "is it in the frame"); the renderer and
    the horizon fit use the ellipsoid. Valid for rho well below 90 deg (GEO and beyond).
    """
    p = np.asarray(p, dtype=np.float64)
    dist = np.linalg.norm(p)
    if dist <= R:
        raise ValueError(f"position is inside Earth (|p| = {dist:.1f} km <= R = {R} km)")
    rho = np.arcsin(R / dist)
    pixel_radius = F_PIX * np.tan(rho)
    b = A @ (-p / dist)                                  # Earth's centre in the camera frame
    if b[2] <= 0:                                        # behind the camera
        return False, (np.nan, np.nan), rho, pixel_radius
    u = F_PIX * b[0] / b[2] + C_PIX
    v = F_PIX * b[1] / b[2] + C_PIX
    # overlap test: the disc's axis-aligned box against the sensor box
    margin = pixel_radius
    visible = bool(-0.5 - margin <= u <= n_pix - 0.5 + margin and -0.5 - margin <= v <= n_pix - 0.5 + margin)
    return visible, (float(u), float(v)), float(rho), float(pixel_radius)
