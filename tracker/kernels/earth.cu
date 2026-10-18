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

    const double inv_a = 1.0 / a, inv_c = 1.0 / c;
    const double psx = p[0] * inv_a, psy = p[1] * inv_a, psz = p[2] * inv_c;          // scaled position
    const double C = psx * psx + psy * psy + psz * psz - 1.0;
    const double inv_a2 = inv_a * inv_a, inv_c2 = inv_c * inv_c;

    double light = 0.0;
    int hits = 0;
    for (int ky = 0; ky < ss; ++ky) {
        const double dv = (ky + 0.5) / ss - 0.5;
        for (int kx = 0; kx < ss; ++kx) {
            const double du = (kx + 0.5) / ss - 0.5;
            // camera-frame ray (unproject), then world frame: d = b @ A
            double bx = ((double)u + du - c_pix) / f_pix;
            double by = ((double)v + dv - c_pix) / f_pix;
            double bz = 1.0;
            const double bn = 1.0 / sqrt(bx * bx + by * by + 1.0);
            bx *= bn; by *= bn; bz *= bn;
            const double dx = bx * A[0] + by * A[3] + bz * A[6];
            const double dy = bx * A[1] + by * A[4] + bz * A[7];
            const double dz = bx * A[2] + by * A[5] + bz * A[8];
            // ray-ellipsoid in scaled space
            const double dsx = dx * inv_a, dsy = dy * inv_a, dsz = dz * inv_c;
            const double Aq = dsx * dsx + dsy * dsy + dsz * dsz;
            const double Bq = 2.0 * (dsx * psx + dsy * psy + dsz * psz);
            const double disc = Bq * Bq - 4.0 * Aq * C;
            if (disc <= 0.0)
                continue;
            const double t = (-Bq - sqrt(disc)) / (2.0 * Aq);
            if (t <= 0.0)
                continue;
            ++hits;
            if (shade) {
                const double x = p[0] + t * dx, y = p[1] + t * dy, z = p[2] + t * dz;
                double nx = x * inv_a2, ny = y * inv_a2, nz = z * inv_c2;              // ellipsoid normal
                const double nn = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
                const double cosi = (nx * s[0] + ny * s[1] + nz * s[2]) * nn;
                light += cosi > 0.0 ? cosi : 0.0;
            } else {
                light += 1.0;
            }
        }
    }
    const long long idx = (long long)v * n_pix + u;
    out[idx] = (float)(brightness * light / (double)(ss * ss));
    mask[idx] = hits > 0 ? 1 : 0;
}
