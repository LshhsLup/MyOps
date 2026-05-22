#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/binary_ops.cuh"
#include "include/common.h"
#include "include/element_wise.cuh"
#include "include/kernels.h"

namespace myops {

template <typename scalar_t>
__global__ void conv1dKernel_v1(scalar_t *__restrict__ output,
                                const scalar_t *__restrict__ input,
                                const scalar_t *__restrict__ kernel,
                                const int input_size,
                                const int kernel_size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int output_size = input_size - kernel_size + 1;
  float sum{0.f};
  if (tid < output_size) {
    for (int i = 0; i < kernel_size; ++i) {
      sum += FloatConverter<scalar_t>::to_float(input[tid + i]) *
             FloatConverter<scalar_t>::to_float(kernel[i]);
    }
  }
  output[tid] = FloatConverter<scalar_t>::from_float(sum);
}

// todo: use shared_memory
template <typename scalar_t>
__global__ void conv1dKernel_v2(scalar_t *__restrict__ output,
                                const scalar_t *__restrict__ input,
                                const scalar_t *__restrict__ kernel,
                                const int input_size,
                                const int kernel_size) {
  constexpr int tile = threads_per_block();
  __shared__ scalar_t smem_kernel[tile];
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int block_tid = threadIdx.x;
  int output_size = input_size - kernel_size + 1;
  float sum{0.f};
  for (int i = 0; i < (kernel_size + tile - 1) / tile; ++i) {
    if (i * tile + block_tid < kernel_size) {
      smem_kernel[block_tid] = kernel[i * tile + block_tid];
    } else {
      smem_kernel[block_tid] = FloatConverter<scalar_t>::from_float(0.f);
    }
    __syncthreads();
    // one thread work
    for (int j = 0; j < tile; ++j) {
      if (global_tid < output_size && i * tile + j < kernel_size) {
        scalar_t in = input[global_tid + i * tile + j];
        sum += FloatConverter<scalar_t>::to_float(in) *
               FloatConverter<scalar_t>::to_float(smem_kernel[j]);
      }
    }
    __syncthreads();
  }
  if (global_tid < output_size) {
    output[global_tid] = FloatConverter<scalar_t>::from_float(sum);
  }
}

template <typename scalar_t>
void launchConv1dKernelImpl(scalar_t *output,
                            const scalar_t *input,
                            const scalar_t *kernel,
                            const int input_size,
                            const int kernel_size,
                            cudaStream_t stream) {
  constexpr int threads = threads_per_block();
  int output_size = input_size - kernel_size + 1;
  int blocks = (output_size + threads - 1) / threads;
  conv1dKernel_v2<scalar_t>
      <<<blocks, threads, 0, stream>>>(output, input, kernel, input_size, kernel_size);
  MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
}

void launchConv1dKernel(void *output,
                        const void *input,
                        const void *kernel,
                        int input_size,
                        int kernel_size,
                        cudaStream_t stream,
                        MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      launchConv1dKernelImpl(static_cast<float *>(output), static_cast<const float *>(input),
                             static_cast<const float *>(kernel), input_size, kernel_size, stream);
      break;
    case MYOPS_DTYPE_HALF:
      launchConv1dKernelImpl(static_cast<__half *>(output), static_cast<const __half *>(input),
                             static_cast<const __half *>(kernel), input_size, kernel_size, stream);
      break;
    case MYOPS_DTYPE_BFLOAT16:
      launchConv1dKernelImpl(
          static_cast<__nv_bfloat16 *>(output), static_cast<const __nv_bfloat16 *>(input),
          static_cast<const __nv_bfloat16 *>(kernel), input_size, kernel_size, stream);
      break;
    default:
      MYOPS_CHECK_FAILED("Only support float32, bfloat16 and half.");
  }
}

}  // namespace myops
