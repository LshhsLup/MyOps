#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

#define FULL_MASK 0xffffffff

namespace myops {

// naive implementation
template <typename scalar_t>
__global__ void reduceSumKernel_V0(scalar_t *__restrict__ output,
                                   const scalar_t *__restrict__ input,
                                   const int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  __shared__ float smem_val[threads_per_block()];
  if (idx < N) {
    smem_val[tid] = FloatConverter<scalar_t>::to_float(input[idx]);
  } else {
    smem_val[tid] = 0.f;
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      smem_val[tid] += smem_val[tid + i];
    }
    __syncthreads();
  }
  // float block_sum = smem_val[0];
  // if (threadIdx.x == 0) {
  //   atomicAdd(output, FloatConverter<scalar_t>::from_float(block_sum));
  // }
}

template <typename scalar_t>
__global__ void reduceSumKernel_V1(scalar_t *__restrict__ output,
                                   const scalar_t *__restrict__ input,
                                   const int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float grid_stride_sum{0.f};
#pragma unroll
  for (int i = idx; i < N; i += stride) {
    grid_stride_sum += FloatConverter<scalar_t>::to_float(input[i]);
  }
  __shared__ float smem_val[threads_per_block()];
  smem_val[tid] = grid_stride_sum;
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      smem_val[tid] += smem_val[tid + i];
    }
    __syncthreads();
  }
  // float block_sum = smem_val[0];
  // if (threadIdx.x == 0) {
  //   atomicAdd(output, FloatConverter<scalar_t>::from_float(block_sum));
  // }
}

template <typename scalar_t>
__global__ void reduceSumKernel_V2(scalar_t *__restrict__ output,
                                   const scalar_t *__restrict__ input,
                                   const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  float val{0.f};
  for (int i = idx; i < N; i += stride) {
    val += FloatConverter<scalar_t>::to_float(input[i]);
  }
  int warpId = threadIdx.x / WARP_SIZE;
  int laneId = threadIdx.x % WARP_SIZE;
  constexpr int blockSumNum = threads_per_block() / WARP_SIZE;
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

// vectorized(just float)
template <typename scalar_t>
__global__ void reduceSumKernel_V3(scalar_t *__restrict__ output,
                                   const scalar_t *__restrict__ input,
                                   const int N) {
  using vec_t = float4;
  int n4 = N / 4;
  const vec_t *vec_in = reinterpret_cast<const vec_t *>(input);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  vec_t vec_val = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = idx; i < n4; i += stride) {
    vec_t v = vec_in[i];
    vec_val.x += v.x;
    vec_val.y += v.y;
    vec_val.z += v.z;
    vec_val.w += v.w;
  }
  float val = vec_val.x + vec_val.y + vec_val.z + vec_val.w;
  for (int i = idx + n4 * 4; i < N; i += stride) {
    val += FloatConverter<scalar_t>::to_float(input[i]);
  }
  int warpId = threadIdx.x / WARP_SIZE;
  int laneId = threadIdx.x & (WARP_SIZE - 1);
  constexpr int blockSumNum = threads_per_block() / WARP_SIZE;
  __shared__ float smem_sum[blockSumNum];
  float warp_sum{val};
#pragma unroll
  for (int i = (WARP_SIZE >> 1); i > 0; i >>= 1) {
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
  for (int i = (blockSumNum >> 1); i > 0; i >>= 1) {
    block_sum += __shfl_down_sync(FULL_MASK, block_sum, i);
  }
  if (threadIdx.x == 0) {
    if (gridDim.x == 1) {
      *output = FloatConverter<scalar_t>::from_float(block_sum);
    } else {
      atomicAdd(output, FloatConverter<scalar_t>::from_float(block_sum));
    }
  }
}

std::pair<int, int> get_grids(int N, int sms) {
  if (N <= 256) {
    return {1, threads_per_block()};
  }

  if (N < sms * threads_per_block()) {
    constexpr int block_work_size = 4 * threads_per_block();
    return {(N + block_work_size - 1) / block_work_size, threads_per_block()};
  }

  if (N <= sms * threads_per_block() * 4) {
    constexpr int block_work_size = 4 * threads_per_block();
    return {sms, threads_per_block()};
  }

  return {sms * 4, threads_per_block()};
}

template <typename scalar_t>
void launchReduceSumKernelImpl(scalar_t *__restrict__ output,
                               const scalar_t *__restrict__ input,
                               const int N,
                               cudaStream_t stream) {
  auto [blocks, threads] = get_grids(N, get_sm_count());
  reduceSumKernel_V3<<<dim3(blocks), dim3(threads), 0, stream>>>(output, input, N);
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
