#include <cuda_runtime.h>
#include <limits>
#include "include/common.h"

/*
 * Softmax(xi) = exp(xi) / sum(exp(xj))
 *
 * To avoid numerical overflow, subtract max(X) from all elements:
 * Softmax(xi) = exp(xi - max) / sum(exp(xj - max))
 *
 * Online Softmax computes this in a single pass:
 * m0 = -inf, s0 = 0
 * mk = max(mk-1, xk)
 * sk = sk-1 * exp(mk-1 - mk) + exp(xk - mk)
 * Softmax(xi) = exp(xi - mn) / sn
 */

namespace myops {

__global__ void softmaxKernel_stage1(float *output,
                              const float *input,
                              float *workspace,
                              int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  float warp_max = std::numeric_limits<float>::lowest();
  float warp_sum = 0.f;
  int tid = threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float v = -std::numeric_limits<float>::infinity();
  for (int i = idx; i < N; i += stride) {
    v = input[i];
    float new_warp_max = fmaxf(warp_max, v);
    warp_sum = warp_sum * expf(warp_max - new_warp_max) + expf(v - new_warp_max);
    warp_max = new_warp_max;
  }
  #pragma unroll
  for (int i = WARP_SIZE >> 1; i > 0; i >>= 1) {
    float other_max = __shfl_down_sync(0xffffffff, warp_max, i);
    float other_sum = __shfl_down_sync(0xffffffff, warp_sum, i);
    float new_warp_max = fmaxf(other_max, warp_max);
    warp_sum = warp_sum * expf(warp_max - new_warp_max) + other_sum * expf(other_max - new_warp_max);
    warp_max = new_warp_max;
  }
  constexpr int warps_per_block = threads_per_block() / WARP_SIZE;
  __shared__ float smem_max[warps_per_block];
  __shared__ float smem_sum[warps_per_block];
  int warpId = threadIdx.x >> 5;
  int laneId = threadIdx.x & (WARP_SIZE - 1);
  if (laneId == 0) {
    smem_max[warpId] = warp_max;
    smem_sum[warpId] = warp_sum;
  }
  __syncthreads();
  float max = -std::numeric_limits<float>::infinity();
  float sum = 0.f;
  if (warpId == 0 && laneId < warps_per_block) {
    max = smem_max[laneId];
    sum = smem_sum[laneId];
  }
  float block_sum{sum};
  float block_max{max};
  #pragma unroll
  for (int i = warps_per_block >> 1; i > 0; i >>= 1) {
    float other_max = __shfl_down_sync(0x000000ff, block_max, i);
    float other_sum = __shfl_down_sync(0x000000ff, block_sum, i);
    float new_block_max = fmaxf(other_max, block_max);
    block_sum = block_sum * expf(block_max - new_block_max) + other_sum * expf(other_max - new_block_max);
    block_max = new_block_max;
  }
  if (threadIdx.x == 0) {
    workspace[blockIdx.x] = block_max;
    workspace[blockIdx.x + gridDim.x] = block_sum;
  }
}

__global__ void softmaxKernel_stage2(float *output,
                                     const float *input,
                                     float *workspace) {

}

void launchSoftmaxKernel(float *output,
                        const float *input,
                        float *workspace,
                        int N,
                        cudaStream_t stream) {
  constexpr int threads = threads_per_block();
  int blocks = (N + threads - 1) / threads;
  softmaxKernel_stage1<<<dim3(blocks), dim3(threads), 0, stream>>>(output, input, workspace, N);
  int blocks1 = (blocks1 + blocks - 1) / blocks;
  MYOPS_CUDA_KERNEL_LAUNCH_CHECK();
}


} // namespace myops