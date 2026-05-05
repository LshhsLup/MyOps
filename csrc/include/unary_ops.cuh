#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace myops {
// Float operations
template <typename T>
struct UnaryOps {
  static __device__ __forceinline__ T abs(T x) { return ::fabsf(x); }
  static __device__ __forceinline__ T neg(T x) { return -x; }
  static __device__ __forceinline__ T exp(T x) { return ::expf(x); }
  static __device__ __forceinline__ T log(T x) { return ::logf(x); }
  static __device__ __forceinline__ T relu(T x) { return x > T(0) ? x : T(0); }
  static __device__ __forceinline__ T sigmoid(T x) { return T(1) / (T(1) + ::expf(-x)); }
};

// Half operations
template <>
struct UnaryOps<__half> {
  static __device__ __forceinline__ __half abs(__half x) { return __habs(x); }
  static __device__ __forceinline__ __half neg(__half x) { return __hneg(x); }
  static __device__ __forceinline__ __half exp(__half x) { return hexp(x); }
  static __device__ __forceinline__ __half log(__half x) { return hlog(x); }
  static __device__ __forceinline__ __half relu(__half x) {
    const __half zero = __float2half(0.0f);
    return __hgt(x, zero) ? x : zero;
  }
  static __device__ __half sigmoid(__half x) {
    const __half one = __float2half(1.0f);
    return __hdiv(one, __hadd(one, hexp(__hneg(x))));
  }
};

// BFloat16 operations
template <>
struct UnaryOps<__nv_bfloat16> {
  static __device__ __forceinline__ __nv_bfloat16 abs(__nv_bfloat16 x) { return __habs(x); }
  static __device__ __forceinline__ __nv_bfloat16 neg(__nv_bfloat16 x) { return __hneg(x); }
  static __device__ __forceinline__ __nv_bfloat16 exp(__nv_bfloat16 x) { return hexp(x); }
  static __device__ __forceinline__ __nv_bfloat16 log(__nv_bfloat16 x) { return hlog(x); }
  static __device__ __forceinline__ __nv_bfloat16 relu(__nv_bfloat16 x) {
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    return __hgt(x, zero) ? x : zero;
  }
  static __device__ __forceinline__ __nv_bfloat16 sigmoid(__nv_bfloat16 x) {
    const __nv_bfloat16 one = __float2bfloat16(1.0f);
    return __hdiv(one, __hadd(one, hexp(__hneg(x))));
  }
};

}  // namespace myops