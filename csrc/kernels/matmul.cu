#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

namespace myops {

template <typename scalar_t>
__global__ void matmulKernel_v1(scalar_t *__restrict__ c,
                             const scalar_t *__restrict__ a,
                             const scalar_t *__restrict__ b,
                             const int m,
                             const int n,
                             const int k) {
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_idx < m && col_idx < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += FloatConverter<scalar_t>::to_float(a[row_idx * k + i]) *
             FloatConverter<scalar_t>::to_float(b[i * n + col_idx]);
    }
    c[row_idx * n + col_idx] = FloatConverter<scalar_t>::from_float(sum);
  }
}

// tiling(16 * 16) + shared_memory
template <typename scalar_t>
__global__ void matmulKernel_v2(scalar_t *__restrict__ c,
                                const scalar_t *__restrict__ a,
                                const scalar_t *__restrict__ b,
                                const int m,
                                const int n,
                                const int k) {
  constexpr int tile = 16;
  __shared__ scalar_t sram_a_tile[tile][tile];
  __shared__ scalar_t sram_b_tile[tile][tile];;
  int global_tid_m = blockIdx.x * blockDim.x + threadIdx.x;
  int global_tid_n = blockIdx.y * blockDim.y + threadIdx.y;
  int block_tid_m = threadIdx.x;
  int block_tid_n = threadIdx.y;
  float sum = 0.f;
  for (int i = 0; i < (k + tile - 1) / tile; ++i) {
    if (block_tid_n + i * tile < k && block_tid_m + i * tile < k) {
      sram_a_tile[block_tid_m][block_tid_n] = a[global_tid_m * k + i * tile + block_tid_n];
      sram_b_tile[block_tid_m][block_tid_n] = b[i * tile * n + global_tid_n + block_tid_m * n];
    } else {
      sram_a_tile[block_tid_m][block_tid_n] = FloatConverter<scalar_t>::from_float(0.f);
      sram_b_tile[block_tid_m][block_tid_n] = FloatConverter<scalar_t>::from_float(0.f);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < tile; ++j) {
      sum += FloatConverter<scalar_t>::to_float(sram_a_tile[block_tid_m][j]) *
              FloatConverter<scalar_t>::to_float(sram_b_tile[j][block_tid_n]);
    }
  }
  __syncthreads();
  if (global_tid_m < m && global_tid_n < n) {
    c[block_tid_m * n + block_tid_n] = FloatConverter<scalar_t>::from_float(sum);
  }
}

template <typename scalar_t>
void launchMatmulKernelImpl(scalar_t *c,
                            const scalar_t *a,
                            const scalar_t *b,
                            int m,
                            int n,
                            int k,
                            cudaStream_t stream) {
  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  matmulKernel_v1<scalar_t><<<blocks, threads, 16*16*2*sizeof(scalar_t), stream>>>(c, a, b, m, n, k);
  MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
}

void launchMatmulKernel(void *c,
                        const void *a,
                        const void *b,
                        int m,
                        int n,
                        int k,
                        cudaStream_t stream,
                        myops::MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      launchMatmulKernelImpl(static_cast<float *>(c), static_cast<const float *>(a),
                                    static_cast<const float *>(b), m, n, k, stream);
      break;
    case MYOPS_DTYPE_HALF:
      launchMatmulKernelImpl(static_cast<__half *>(c), static_cast<const __half *>(a),
                                    static_cast<const __half *>(b), m, n, k, stream);
      break;
    case MYOPS_DTYPE_BFLOAT16:
      launchMatmulKernelImpl(static_cast<__nv_bfloat16 *>(c),
                                    static_cast<const __nv_bfloat16 *>(a),
                                    static_cast<const __nv_bfloat16 *>(b), m, n, k, stream);
      break;
    default:
      MYOPS_CHECK_FAILED("Only support float32, bfloat16 and half.");   
  }
}
}  // namespace myops
