#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

#define FULL_MASK 0xffffffff

namespace myops {

template <typename scalar_t>
__global__ void reduceSumKernel(scalar_t *__restrict__ output,
                                const scalar_t *__restrict__ input,
                                const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val{0.f};
  if (idx < N) {
    val = FloatConverter<scalar_t>::to_float(input[idx]);
  }
  int warpId = threadIdx.x / WARP_SIZE;
  int laneId = threadIdx.x % WARP_SIZE;
  constexpr int blockSumNum = 128 / WARP_SIZE;
  __shared__ float smem_sum[blockSumNum];
  float warp_sum{val};
#pragma unroll
  for (int i = 16; i > 0; i /= 2) {
    warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, i);
  }
  if (laneId == 0) {
    smem_sum[warpId] = warp_sum;
  }
  __syncthreads();
  float v{0.f};
  if (warpId == 0) {
    if (laneId < blockSumNum) {
      v = smem_sum[laneId];
    }
  }

  float block_sum{v};
#pragma unroll
  for (int i = 2; i > 0; i /= 2) {
    block_sum += __shfl_down_sync(FULL_MASK, block_sum, i);
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, FloatConverter<scalar_t>::from_float(block_sum));
  }
}

template <typename scalar_t>
void launchReduceSumKernelImpl(scalar_t *__restrict__ output,
                               const scalar_t *__restrict__ input,
                               const int N,
                               cudaStream_t stream) {
  dim3 threads(128);
  dim3 blocks((N + threads.x - 1) / threads.x);
  reduceSumKernel<<<blocks, threads, 0, stream>>>(output, input, N);
  MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
}

void launchReduceSumKernel(void *output,
                           const void *input,
                           int N,
                           cudaStream_t stream,
                           MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      launchReduceSumKernelImpl(static_cast<float *>(output), static_cast<const float *>(input), N,
                                stream);
      break;
    case MYOPS_DTYPE_HALF:
      launchReduceSumKernelImpl(static_cast<__half *>(output), static_cast<const __half *>(input),
                                N, stream);
      break;
    case MYOPS_DTYPE_BFLOAT16:
      launchReduceSumKernelImpl(static_cast<__nv_bfloat16 *>(output),
                                static_cast<const __nv_bfloat16 *>(input), N, stream);
      break;
    default:
      MYOPS_CHECK_FAILED("Only support float32, bfloat16 and half.");
  }
}

}  // namespace myops
