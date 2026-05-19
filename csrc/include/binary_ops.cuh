#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace myops {
// Float operations
template <typename T>
struct BinaryOps {
  static __device__ __forceinline__ T add(T a, T b) { return a + b; }
  static __device__ __forceinline__ T sub(T a, T b) { return a - b; }
  static __device__ __forceinline__ T mul(T a, T b) { return a * b; }
  static __device__ __forceinline__ T div(T a, T b) { return a / b; }
};

// Half operations
template <>
struct BinaryOps<__half> {
  static __device__ __forceinline__ __half add(__half a, __half b) { return __hadd(a, b); }
  static __device__ __forceinline__ __half sub(__half a, __half b) { return __hsub(a, b); }
  static __device__ __forceinline__ __half mul(__half a, __half b) { return __hmul(a, b); }
  static __device__ __forceinline__ __half div(__half a, __half b) { return __hdiv(a, b); }
};

// BFloat16 operations
template <>
struct BinaryOps<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
  }
  static __device__ __forceinline__ __nv_bfloat16 sub(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hsub(a, b);
  }
  static __device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
  }
  static __device__ __forceinline__ __nv_bfloat16 div(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hdiv(a, b);
  }
};

}  // namespace myops
