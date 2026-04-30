#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

template <typename scalar_t>
__global__ void matmulKernel(scalar_t *__restrict__ c,
                             const scalar_t *__restrict__ a,
                             const scalar_t *__restrict__ b,
                             const int m,
                             const int n,
                             const int k) {
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx < m && col_idx < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += FloatConverter<scalar_t>::to_float(a[row_idx * k + i]) *
             FloatConverter<scalar_t>::to_float(b[i * n + col_idx]);
    }
    c[row_idx * n + col_idx] = FloatConverter<scalar_t>::from_float(sum);
  }
}

template <typename scalar_t>
cudaError_t launchMatmulKernelImpl(scalar_t *c,
                                   const scalar_t *a,
                                   const scalar_t *b,
                                   int m,
                                   int n,
                                   int k,
                                   cudaStream_t stream) {
  dim3 threads(32, 32);
  dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  matmulKernel<scalar_t><<<blocks, threads, 0, stream>>>(c, a, b, m, n, k);
  return cudaGetLastError();
}

cudaError_t launchMatmulKernel(void *c,
                               const void *a,
                               const void *b,
                               int m,
                               int n,
                               int k,
                               cudaStream_t stream,
                               MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      return launchMatmulKernelImpl(static_cast<float *>(c), static_cast<const float *>(a),
                                    static_cast<const float *>(b), m, n, k, stream);
    case MYOPS_DTYPE_HALF:
      return launchMatmulKernelImpl(static_cast<__half *>(c), static_cast<const __half *>(a),
                                    static_cast<const __half *>(b), m, n, k, stream);
    case MYOPS_DTYPE_BFLOAT16:
      return launchMatmulKernelImpl(static_cast<__nv_bfloat16 *>(c),
                                    static_cast<const __nv_bfloat16 *>(a),
                                    static_cast<const __nv_bfloat16 *>(b), m, n, k, stream);
  }
  return cudaErrorInvalidValue;
}
