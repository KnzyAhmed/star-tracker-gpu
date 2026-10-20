// Lets Apple/LLVM clang SYNTAX-CHECK our CUDA kernels on a machine without the CUDA toolkit.
// Not used for real builds: on the GPU, CuPy compiles the .cu files with NVRTC.
// Usage: clang -fsyntax-only -x cuda --cuda-device-only --cuda-gpu-arch=sm_75 \
//              -nocudainc -nocudalib -include tools/cuda_shim.h tracker/kernels/render.cu
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#include "__clang_cuda_builtin_vars.h"
__device__ void __syncthreads(void);
__device__ float expf(float);
__device__ float logf(float);
__device__ float sqrtf(float);
__device__ double sqrt(double);
__device__ float sqrt(float);
__device__ float cosf(float);
__device__ int min(int, int);
__device__ int max(int, int);
__device__ int atomicAdd(int*, int);
__device__ unsigned int atomicAdd(unsigned int*, unsigned int);
