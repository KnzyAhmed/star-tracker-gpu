"""Pack many frames' star lists into flat arrays for the GPU (pure NumPy, no CuPy here).

Frame f owns stars offsets[f] .. offsets[f+1]-1 of the flat arrays ("CSR" layout): one
upload for a whole batch, and a GPU block for frame f finds its stars with two reads.
"""
from dataclasses import dataclass

import numpy as np

from tracker.camera import project
from tracker.render_cpu import SIGMA_PSF, star_flux, star_windows


@dataclass
class StarBatch:
    offsets: np.ndarray    # (F+1,) int32
    du: np.ndarray         # (S,) float32 column of the star MINUS box u_min (0 .. ~11)
    dv: np.ndarray         # (S,) float32 row of the star MINUS box v_min
    amp: np.ndarray        # (S,) float32 peak counts = flux / (2 pi sigma^2)
    box: np.ndarray        # (S, 4) int32 u_min, u_max, v_min, v_max (inclusive, = CPU windows)

    @property
    def n_frames(self):
        return len(self.offsets) - 1


def pack_frames(frames, sigma_psf=SIGMA_PSF):
    """frames: list of (u, v, flux) arrays, one tuple per frame. Returns a StarBatch."""
    counts = [len(f[0]) for f in frames]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    u = np.concatenate([np.asarray(f[0], dtype=np.float64) for f in frames])
    v = np.concatenate([np.asarray(f[1], dtype=np.float64) for f in frames])
    flux = np.concatenate([np.asarray(f[2], dtype=np.float64) for f in frames])
    amp = flux / (2 * np.pi * sigma_psf**2)
    box = np.column_stack(star_windows(u, v)).astype(np.int32).reshape(-1, 4)
    # Positions are sent RELATIVE to the box corner: float32 resolves 1000.7 only to ~3e-5 px,
    # but 5.7 to ~2e-7 px. The GPU then computes dx = (x - u_min) - du with small numbers.
    du = (u - box[:, 0]).astype(np.float32)
    dv = (v - box[:, 2]).astype(np.float32)
    return StarBatch(offsets=offsets, du=du, dv=dv, amp=amp.astype(np.float32), box=box)


def project_frames(As, catalog):
    """Attitudes -> (frames, idx): per frame the visible stars' (u, v, flux), and their catalog
    indices (the truth), exactly as render_sky would draw them."""
    frames, idx = [], []
    for A in As:
        u, v, visible = project(A, catalog.r)
        k = np.flatnonzero(visible)
        frames.append((u[k], v[k], star_flux(catalog.mag[k])))
        idx.append(k)
    return frames, idx
