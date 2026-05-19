#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/binary_ops.cuh"
#include "include/common.h"
#include "include/element_wise.cuh"
#include "include/kernels.h"

namespace myops {
// Generic binary op functor
#define DEFINE_BINARY_OP_FUNCTOR(lower, upper) \
  template <typename T>                        \
  struct upper##Op {                           \
    __device__ T operator()(T a, T b) const {  \
      return BinaryOps<T>::lower(a, b);        \
    }                                          \
  };

FOR_EACH_BINARY_OP(DEFINE_BINARY_OP_FUNCTOR)

template <typename scalar_t, typename Op>
void launchBinaryKernelImpl(scalar_t *out, const scalar_t *a, const scalar_t *b, size_t n,
                            cudaStream_t stream) {
  TensorArray<scalar_t, 2> data(out, a, b);
  launch_vectorized_kernel(stream, n, Op{}, data);
}

#define DEFINE_BINARY_KERNEL(lower, upper)                                                    \
  void launch##upper##Kernel(void *out, const void *a, const void *b, size_t n,               \
                             cudaStream_t stream, myops::MyOpsDtype dtype) {                  \
    switch (dtype) {                                                                          \
      case myops::MYOPS_DTYPE_FLOAT:                                                          \
        launchBinaryKernelImpl<float, upper##Op<float>>(                                      \
            static_cast<float *>(out), static_cast<const float *>(a),                         \
            static_cast<const float *>(b), n, stream);                                        \
        break;                                                                                \
      case myops::MYOPS_DTYPE_HALF:                                                           \
        launchBinaryKernelImpl<__half, upper##Op<__half>>(                                    \
            static_cast<__half *>(out), static_cast<const __half *>(a),                       \
            static_cast<const __half *>(b), n, stream);                                       \
        break;                                                                                \
      case myops::MYOPS_DTYPE_BFLOAT16:                                                       \
        launchBinaryKernelImpl<__nv_bfloat16, upper##Op<__nv_bfloat16>>(                      \
            static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(a),         \
            static_cast<const __nv_bfloat16 *>(b), n, stream);                                \
        break;                                                                                \
      default:                                                                                \
        MYOPS_CHECK_FAILED("Only support float32, bfloat16 and half.");                       \
    }                                                                                         \
  }

FOR_EACH_BINARY_OP(DEFINE_BINARY_KERNEL)

}  // namespace myops
