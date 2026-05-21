#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

namespace myops {

template <typename scalar_t>
__global__ void matTransKernel(scalar_t* __restrict__ out,
                                 const scalar_t* __restrict__ in,
                                 const int rows,
                                 const int cols) {
  int tid_m = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_n = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_m < rows && tid_n < cols) {
    out[tid_n * rows + tid_m] = in[tid_m * cols + tid_n];
  }
}

template <typename scalar_t>
void launchMatTransKernelImpl(scalar_t *out,
                            const scalar_t *in,
                            int rows,
                            int cols,
                            cudaStream_t stream) {
  dim3 threads(16, 16);
  dim3 blocks((rows + threads.x - 1) / threads.x, (cols + threads.y - 1) / threads.y);
  matTransKernel<scalar_t><<<blocks, threads, 0, stream>>>(out, in, rows, cols);
  MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
}

void launchMatTransKernel(void *out,
                        const void *in,
                        int rows,
                        int cols,
                        cudaStream_t stream,
                        myops::MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      launchMatTransKernelImpl(static_cast<float *>(out), static_cast<const float *>(in),
                              rows, cols,stream);
      break;
    case MYOPS_DTYPE_HALF:
      launchMatTransKernelImpl(static_cast<__half *>(out), static_cast<const __half *>(in),
                              rows, cols,stream);
      break;
    case MYOPS_DTYPE_BFLOAT16:
      launchMatTransKernelImpl(static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(in),
                              rows, cols,stream);
      break;
    default:
      MYOPS_CHECK_FAILED("Only support float32, bfloat16 and half.");
  }
}

} // namespace myops