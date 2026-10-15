// detect.cu: find stars in a batch of images, one thread per pixel per frame.
//
// Launch: block (16, 16), grid (n_pix/16, n_pix/16, n_frames). Mirrors tracker/detect_cpu.detect:
//   candidate = pixel > threshold (float32) AND local max of its 3x3 neighbourhood, with the
//               raster-order tie-break (> the 4 earlier neighbours, >= the 4 later ones);
//   centroid  = background-subtracted, brightness-weighted mean over the 5x5 window.
//
// Each block first copies its 16x16 pixels plus a 2-pixel halo (20x20) into shared memory,
// so the 25 reads per candidate come from fast on-chip memory. Off-sensor halo pixels hold
// OFF_SENSOR: it loses every local-max comparison (like the CPU's -inf padding) and counts
// as zero weight in the centroid (like the CPU's background padding).
//
// Output slots are claimed with atomicAdd on a per-frame counter, so the ORDER of the
// detections is arbitrary; the host sorts them. The counter keeps counting past `capacity`,
// so the host can tell that stars were lost and fail loudly.

#define TILE 16
#define HALO 2
#define SPAN (TILE + 2 * HALO)          // 20
#define OFF_SENSOR (-3.402823466e38f)   // -FLT_MAX: below any real pixel

extern "C" __global__ void detect_kernel(
    const float* __restrict__ images,   // (F, n_pix, n_pix)
    const float background,
    const float threshold,              // float32(background + k_sigma * read_noise), from the host
    const int n_pix,
    const int capacity,                 // output slots per frame
    int* __restrict__ count,            // (F,) zeroed by the host; detections found per frame
    float* __restrict__ out)            // (F, capacity, 5): peak u, peak v, du, dv, flux
{
    __shared__ float tile[SPAN][SPAN];

    const int f = blockIdx.z;
    const int x0 = blockIdx.x * TILE - HALO;                    // image column of tile[.][0]
    const int y0 = blockIdx.y * TILE - HALO;
    const float* img = images + (long long)f * n_pix * n_pix;  // 64-bit offset to frame f

    // 256 threads copy 400 values: each thread takes every 256th one.
    const int tid = threadIdx.y * TILE + threadIdx.x;
    for (int k = tid; k < SPAN * SPAN; k += TILE * TILE) {
        const int ty = k / SPAN, tx = k % SPAN;
        const int gx = x0 + tx, gy = y0 + ty;
        tile[ty][tx] = (gx >= 0 && gx < n_pix && gy >= 0 && gy < n_pix) ? img[gy * n_pix + gx] : OFF_SENSOR;
    }
    __syncthreads();

    const int x = blockIdx.x * TILE + threadIdx.x;               // this thread's pixel
    const int y = blockIdx.y * TILE + threadIdx.y;
    if (x >= n_pix || y >= n_pix)
        return;                                                  // no __syncthreads below this
    const int cx = threadIdx.x + HALO, cy = threadIdx.y + HALO;  // its place in the tile
    const float c = tile[cy][cx];
    if (!(c > threshold))
        return;

    // 3x3 local max, raster-order tie-break: strictly brighter than the neighbours BEFORE it
    // (row above, and left), at least as bright as those AFTER it (right, and row below).
    if (!(c > tile[cy - 1][cx - 1] && c > tile[cy - 1][cx] && c > tile[cy - 1][cx + 1] &&
          c > tile[cy][cx - 1] &&
          c >= tile[cy][cx + 1] &&
          c >= tile[cy + 1][cx - 1] && c >= tile[cy + 1][cx] && c >= tile[cy + 1][cx + 1]))
        return;

    // 5x5 centroid. Negative weights (noise below background) are kept, like the CPU.
    float flux = 0.0f, su = 0.0f, sv = 0.0f;
    for (int dy = -HALO; dy <= HALO; ++dy) {
        for (int dx = -HALO; dx <= HALO; ++dx) {
            const float p = tile[cy + dy][cx + dx];
            const float w = (p == OFF_SENSOR) ? 0.0f : p - background;
            flux += w;
            su += w * (float)dx;
            sv += w * (float)dy;
        }
    }
    if (!(flux > 0.0f))
        return;

    const int slot = atomicAdd(&count[f], 1);
    if (slot < capacity) {
        float* o = out + ((long long)f * capacity + slot) * 5;
        o[0] = (float)x;          // integers: exact in float32
        o[1] = (float)y;
        o[2] = su / flux;         // sub-pixel offsets in [-2, 2]: full float32 precision
        o[3] = sv / flux;
        o[4] = flux;
    }
}
