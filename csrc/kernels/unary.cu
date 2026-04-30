#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/kernels.h"

template <typename scalar_t, typename UnaryOp>
__global__ void unaryKernel(scalar_t *__restrict__ out,
                            const scalar_t *__restrict__ input,
                            const size_t n,
                            UnaryOp op) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    out[i] = op(input[i]);
  }
}

// Float operations
template <typename T>
struct FloatOps {
  static __device__ __forceinline__ T abs(T x) { return ::fabsf(x); }
  static __device__ __forceinline__ T neg(T x) { return -x; }
  static __device__ __forceinline__ T exp(T x) { return ::expf(x); }
  static __device__ __forceinline__ T log(T x) { return ::logf(x); }
  static __device__ __forceinline__ T relu(T x) { return x > T(0) ? x : T(0); }
  static __device__ __forceinline__ T sigmoid(T x) { return T(1) / (T(1) + ::expf(-x)); }
};

// Half operations
template <>
struct FloatOps<__half> {
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
struct FloatOps<__nv_bfloat16> {
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

// Generic unary op functor
#define DEFINE_UNARY_OP_FUNCTOR(lower, upper)                            \
  template <typename T>                                                  \
  struct upper##Op {                                                     \
    __device__ T operator()(T x) const { return FloatOps<T>::lower(x); } \
  };

FOR_EACH_UNARY_OP(DEFINE_UNARY_OP_FUNCTOR)

template <typename scalar_t, typename Op>
cudaError_t launchUnaryKernelImpl(scalar_t *out,
                                  const scalar_t *input,
                                  size_t n,
                                  cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  const int blocks = num_sms * 4;
  unaryKernel<scalar_t, Op><<<blocks, threadsPerBlock, 0, stream>>>(out, input, n, Op{});
  return cudaGetLastError();
}

#define DEFINE_UNARY_KERNEL(lower, upper)                                                        \
  cudaError_t launch##upper##Kernel(void *out, const void *input, size_t n, cudaStream_t stream, \
                                    MyOpsDtype dtype) {                                          \
    switch (dtype) {                                                                             \
      case MYOPS_DTYPE_FLOAT:                                                                    \
        return launchUnaryKernelImpl<float, upper##Op<float>>(                                   \
            static_cast<float *>(out), static_cast<const float *>(input), n, stream);            \
      case MYOPS_DTYPE_HALF:                                                                     \
        return launchUnaryKernelImpl<__half, upper##Op<__half>>(                                 \
            static_cast<__half *>(out), static_cast<const __half *>(input), n, stream);          \
      case MYOPS_DTYPE_BFLOAT16:                                                                 \
        return launchUnaryKernelImpl<__nv_bfloat16, upper##Op<__nv_bfloat16>>(                   \
            static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(input), n,     \
            stream);                                                                             \
    }                                                                                            \
    return cudaErrorInvalidValue;                                                                \
  }

FOR_EACH_UNARY_OP(DEFINE_UNARY_KERNEL)
