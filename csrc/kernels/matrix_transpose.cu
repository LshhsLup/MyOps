#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

template <scalar_t>
__global__ void matTransKernel(scalar_t* __restrict__ out,
                                 const scalar_t* __restrict__ in,
                                 const int rows,
                                 const int cols) {
}

template <typename scalar_t>
void launcmatTransKernelImpl(scalar_t *out,
                            const scalar_t *in,
                            int rows,
                            int cols,
                            cudaStream_t stream) {
  dim3 threads(16, 16);
  dim3 blocks((rows + threads.x - 1) / threads.x, (cols + threads.y - 1) / threads.y);
  matTransKernel<scalar_t><<<blocks, threads, 0, stream>>>(out, in, rows, cols, k);
  MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
}

void launchMatmulKernel(void *out,
                        const void *in,
                        int rows,
                        int cols,
                        cudaStream_t stream,
                        myops::MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      launchMatmulKernelImpl(static_cast<float *>(out), static_cast<const float *>(in),
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