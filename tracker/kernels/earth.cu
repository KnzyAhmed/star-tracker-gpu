// earth.cu: the sunlit Earth (an ellipsoid) seen by the pinhole camera, one thread per pixel of the
// host-computed bounding box, `ss` x `ss` sub-pixel rays per thread. Mirrors tracker/earth_render.render_earth:
//   ray = unproject(u + du, v + dv) rotated into the world frame (d = b @ A, i.e. d_j = sum_i b_i A[i][j])
//   scale space by S = diag(1/a, 1/a, 1/c): the ellipsoid is the unit sphere; solve the ray-sphere quadratic
//   hit point x = p + t d; normal n ~ (x/a^2, x/a^2, z/c^2); Lambertian max(0, n.s); average the sub-rays
// The PSF blur is applied afterwards by the host (cupyx gaussian_filter), exactly as the CPU does with scipy.
//
// Precision: the geometry runs in DOUBLE. The limb is a hit/miss discontinuity and |p| is ~1e4..1e6 km, so the
// float32 quadratic would flip a few edge sub-rays per frame relative to the CPU; each flip is a brightness/ss^2
// jump in one pixel. Doubles make the GPU image agree with the CPU to float32 rounding, which is the Day 11 gate.
// Output pixels are float32. (The Tesla T4 does ~250 GFLOP/s in double: 13 M rays per GEO frame take milliseconds.)

// REAL_T selects the geometry precision: double (default, the shipped kernel) or float, compiled with
// -DREAL_T=float by gpu.render_earth_batch(precision="float32") for the Day 12 cost/accuracy comparison.
#ifndef REAL_T
#define REAL_T double
#endif
typedef REAL_T real;

extern "C" __global__ void earth_kernel(
    const double* __restrict__ p,        // (3,) spacecraft position, km, world frame
    const double* __restrict__ A,        // (9,) attitude, row-major, sky -> camera
    const double* __restrict__ s,        // (3,) unit vector Earth -> Sun
    const double a, const double c,      // ellipsoid semi-axes, km
    const double f_pix, const double c_pix,
    const int u0, const int v0,          // bounding box origin (pixels)
    const int bw, const int bh,          // bounding box width, height
    const int n_pix,
    const int ss,                        // sub-rays per pixel side
    const double brightness,
    const int shade,                     // 0: flat disc (geometry tests), 1: Lambertian
    float* __restrict__ out,             // (n_pix, n_pix) image, add into
    unsigned char* __restrict__ mask)    // (n_pix, n_pix) 1 if any sub-ray hit
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= bw || iy >= bh)
        return;
    const int u = u0 + ix, v = v0 + iy;

    // every input is converted to the working precision ONCE, so the float32 variant really is float32
    const real P0 = (real)p[0], P1 = (real)p[1], P2 = (real)p[2];
    const real S0 = (real)s[0], S1 = (real)s[1], S2 = (real)s[2];
    real R[9];
    for (int k = 0; k < 9; ++k)
        R[k] = (real)A[k];
    const real fp = (real)f_pix, cpx = (real)c_pix;
    const real inv_a = real(1) / (real)a, inv_c = real(1) / (real)c;
    const real psx = P0 * inv_a, psy = P1 * inv_a, psz = P2 * inv_c;          // scaled position
    const real C = psx * psx + psy * psy + psz * psz - real(1);
    const real inv_a2 = inv_a * inv_a, inv_c2 = inv_c * inv_c;

    real light = real(0);
    int hits = 0;
    for (int ky = 0; ky < ss; ++ky) {
        const real dv = ((real)ky + real(0.5)) / (real)ss - real(0.5);
        for (int kx = 0; kx < ss; ++kx) {
            const real du = ((real)kx + real(0.5)) / (real)ss - real(0.5);
            // camera-frame ray (unproject), then world frame: d = b @ A
            real bx = ((real)u + du - cpx) / fp;
            real by = ((real)v + dv - cpx) / fp;
            real bz = real(1);
            const real bn = real(1) / sqrt(bx * bx + by * by + real(1));
            bx *= bn; by *= bn; bz *= bn;
            const real dx = bx * R[0] + by * R[3] + bz * R[6];
            const real dy = bx * R[1] + by * R[4] + bz * R[7];
            const real dz = bx * R[2] + by * R[5] + bz * R[8];
            // ray-ellipsoid in scaled space
            const real dsx = dx * inv_a, dsy = dy * inv_a, dsz = dz * inv_c;
            const real Aq = dsx * dsx + dsy * dsy + dsz * dsz;
            const real Bq = real(2) * (dsx * psx + dsy * psy + dsz * psz);
            const real disc = Bq * Bq - real(4) * Aq * C;
            if (disc <= real(0))
                continue;
            const real t = (-Bq - sqrt(disc)) / (real(2) * Aq);
            if (t <= real(0))
                continue;
            ++hits;
            if (shade) {
                const real x = P0 + t * dx, y = P1 + t * dy, z = P2 + t * dz;
                real nx = x * inv_a2, ny = y * inv_a2, nz = z * inv_c2;              // ellipsoid normal
                const real nn = real(1) / sqrt(nx * nx + ny * ny + nz * nz);
                const real cosi = (nx * S0 + ny * S1 + nz * S2) * nn;
                light += cosi > real(0) ? cosi : real(0);
            } else {
                light += real(1);
            }
        }
    }
    const long long idx = (long long)v * n_pix + u;
    out[idx] = (float)((real)brightness * light / (real)(ss * ss));
    mask[idx] = hits > 0 ? 1 : 0;
}
