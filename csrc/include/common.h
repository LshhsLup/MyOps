#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define MYOPS_HOST_DEVICE __host__ __device__
#define MYOPS_HOST __host__
#define MYOPS_DEVICE __device__
#else
#define MYOPS_HOST_DEVICE
#define MYOPS_HOST 
#define MYOPS_DEVICE 
#endif

namespace myops {

enum MyOpsDtype { MYOPS_DTYPE_FLOAT, MYOPS_DTYPE_HALF, MYOPS_DTYPE_BFLOAT16 };

constexpr int threadsPerBlock = 256;

// Type conversion helpers
template <typename T>
struct FloatConverter;

template <>
struct FloatConverter<float> {
  static __device__ float to_float(float x) { return x; }
  static __device__ float from_float(float x) { return x; }
};

template <>
struct FloatConverter<__half> {
  static __device__ float to_float(__half x) { return __half2float(x); }
  static __device__ __half from_float(float x) { return __float2half(x); }
};

template <>
struct FloatConverter<__nv_bfloat16> {
  static __device__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
  static __device__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

} // namespace myops

