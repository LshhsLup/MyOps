#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

using vec_t = uint4;

template <typename T>
__device__ __forceinline__ constexpr int getVecSize() {
  return 16 / sizeof(T);
}

// Type-specific add operations
__device__ __forceinline__ float add_op(float a, float b) {
  return a + b;
}

template <typename T,
          typename std::enable_if_t<std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                                    int> = 0>
__device__ __forceinline__ T add_op(T a, T b) {
  return __hadd(a, b);
}

template <typename scalar_t>
__global__ void addKernel(scalar_t *__restrict__ out,
                          const scalar_t *__restrict__ x,
                          const scalar_t *__restrict__ y,
                          const size_t n) {
  constexpr int VEC_SIZE = getVecSize<scalar_t>();
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  const vec_t *x_vec = reinterpret_cast<const vec_t *>(x);
  const vec_t *y_vec = reinterpret_cast<const vec_t *>(y);
  vec_t *o_vec = reinterpret_cast<vec_t *>(out);
  size_t n_vec = n / VEC_SIZE;
  for (size_t i = idx; i < n_vec; i += stride) {
    const vec_t x_val = x_vec[i];
    const vec_t y_val = y_vec[i];
    vec_t o_val;

    const scalar_t *s_x = reinterpret_cast<const scalar_t *>(&x_val);
    const scalar_t *s_y = reinterpret_cast<const scalar_t *>(&y_val);
    scalar_t *s_o = reinterpret_cast<scalar_t *>(&o_val);

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      s_o[j] = add_op(s_x[j], s_y[j]);
    }

    o_vec[i] = o_val;
  }

  for (size_t i = n_vec * VEC_SIZE + idx; i < n; i += stride) {
    out[i] = add_op(x[i], y[i]);
  }
}

template <typename scalar_t>
cudaError_t launchAddKernelImpl(scalar_t *out,
                                const scalar_t *a,
                                const scalar_t *b,
                                size_t n,
                                cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  int num_sms, max_threads_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
  const int max_warps_per_sm = max_threads_per_sm / 32;
  const int blocks = num_sms * max_warps_per_sm * 32 / threadsPerBlock;
  addKernel<scalar_t><<<blocks, threadsPerBlock, 0, stream>>>(out, a, b, n);
  return cudaGetLastError();
}

cudaError_t launchAddKernel(void *out,
                            const void *a,
                            const void *b,
                            size_t n,
                            cudaStream_t stream,
                            MyOpsDtype dtype) {
  switch (dtype) {
    case MYOPS_DTYPE_FLOAT:
      return launchAddKernelImpl(static_cast<float *>(out), static_cast<const float *>(a),
                                 static_cast<const float *>(b), n, stream);
    case MYOPS_DTYPE_HALF:
      return launchAddKernelImpl(static_cast<__half *>(out), static_cast<const __half *>(a),
                                 static_cast<const __half *>(b), n, stream);
    case MYOPS_DTYPE_BFLOAT16:
      return launchAddKernelImpl(static_cast<__nv_bfloat16 *>(out),
                                 static_cast<const __nv_bfloat16 *>(a),
                                 static_cast<const __nv_bfloat16 *>(b), n, stream);
  }
  return cudaErrorInvalidValue;
}
