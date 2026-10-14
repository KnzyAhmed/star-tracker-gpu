// render.cu: draw a batch of star fields, one thread per pixel per frame.
//
// Launch: block (16, 16), grid (n_pix/16, n_pix/16, n_frames). Thread (x, y) of block z
// computes pixel image[z, y, x] = background + sum of star blobs + read_noise * z_noise.
// This mirrors tracker/render_cpu.render (stars in the same order, same integer boxes) and
// tracker/noise.pixel_normals (same hash, same Box-Muller), so CPU and GPU images agree
// to float32 rounding.
//
// Stars of frame f are offsets[f] .. offsets[f+1]-1. They are staged through shared memory
// 64 at a time: all 256 threads of the block read the same stars, so one global read per
// star per block instead of one per thread.

#define TILE_STARS 64

__device__ __forceinline__ unsigned int hash32(unsigned int x)
{
    // lowbias32, identical to tracker.noise.hash32 (unsigned int arithmetic wraps mod 2^32)
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

extern "C" __global__ void render_kernel(
    const int* __restrict__ offsets,              // (F+1,)
    const float* __restrict__ du,                 // (S,) star column minus box u_min
    const float* __restrict__ dv,                 // (S,) star row minus box v_min
    const float* __restrict__ amp,                // (S,) peak counts, flux / (2 pi sigma^2)
    const int* __restrict__ box,                  // (S, 4) u_min, u_max, v_min, v_max
    const unsigned int* __restrict__ frame_ids,   // (F,) global frame number, for the noise
    const float background,
    const float read_noise,
    const float inv_two_sigma2,                   // 1 / (2 sigma_psf^2)
    const unsigned int seed,
    const int n_pix,
    float* __restrict__ out)                      // (F, n_pix, n_pix)
{
    __shared__ float s_du[TILE_STARS], s_dv[TILE_STARS], s_amp[TILE_STARS];
    __shared__ int s_box[TILE_STARS][4];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;      // u, column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;      // v, row
    const int f = blockIdx.z;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..255 within the block
    const int first = offsets[f];
    const int last = offsets[f + 1];

    float acc = background;                                   // same start as the CPU image
    for (int base = first; base < last; base += TILE_STARS) {
        const int n = min(TILE_STARS, last - base);
        if (tid < n) {
            s_du[tid] = du[base + tid];
            s_dv[tid] = dv[base + tid];
            s_amp[tid] = amp[base + tid];
            for (int c = 0; c < 4; ++c)
                s_box[tid][c] = box[4 * (base + tid) + c];
        }
        __syncthreads();                                      // chunk loaded for everyone
        for (int k = 0; k < n; ++k) {
            if (x >= s_box[k][0] && x <= s_box[k][1] && y >= s_box[k][2] && y <= s_box[k][3]) {
                const float dx = (float)(x - s_box[k][0]) - s_du[k];
                const float dy = (float)(y - s_box[k][2]) - s_dv[k];
                acc += s_amp[k] * expf(-(dx * dx + dy * dy) * inv_two_sigma2);
            }
        }
        __syncthreads();                                      // done before the next overwrite
    }
    // Bounds check only AFTER the loop: every thread must reach every __syncthreads().
    if (x >= n_pix || y >= n_pix)
        return;

    if (read_noise > 0.0f) {
        const unsigned int key = hash32(hash32(seed + 0x9E3779B9u) + frame_ids[f]);
        const unsigned int pix = (unsigned int)(y * n_pix + x);
        const unsigned int h1 = hash32(key + 2u * pix);
        const unsigned int h2 = hash32(key + 2u * pix + 1u);
        const float u1 = (float)((h1 >> 8) + 1u) * 5.9604644775390625e-8f;   // 2^-24
        const float u2 = (float)(h2 >> 8) * 5.9604644775390625e-8f;
        const float z = sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
        acc += read_noise * z;
    }
    // 64-bit index: 128 frames x 1024 x 1024 = 1.3e8 fits in 32 bits, but 2,048+ frames would not.
    out[((long long)f * n_pix + y) * n_pix + x] = acc;
}
