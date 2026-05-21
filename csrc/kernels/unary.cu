#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "include/common.h"
#include "include/element_wise.cuh"
#include "include/kernels.h"
#include "include/unary_ops.cuh"

namespace myops {
// Generic unary op functor
#define DEFINE_UNARY_OP_FUNCTOR(lower, upper)                            \
  template <typename T>                                                  \
  struct upper##Op {                                                     \
    __device__ T operator()(T x) const { return UnaryOps<T>::lower(x); } \
  };

FOR_EACH_UNARY_OP(DEFINE_UNARY_OP_FUNCTOR)

template <typename scalar_t, typename Op>
void launchUnaryKernelImpl(scalar_t *out, const scalar_t *input, size_t n, cudaStream_t stream) {
  TensorArray<scalar_t, 1> data(out, input);
  launch_vectorized_kernel(stream, n, Op{}, data);
}

#define DEFINE_UNARY_KERNEL(lower, upper)                                                    \
  void launch##upper##Kernel(void *out, const void *input, size_t n, cudaStream_t stream,    \
                             myops::MyOpsDtype dtype) {                                      \
    switch (dtype) {                                                                         \
      case myops::MYOPS_DTYPE_FLOAT:                                                         \
        launchUnaryKernelImpl<float, upper##Op<float>>(                                      \
            static_cast<float *>(out), static_cast<const float *>(input), n, stream);        \
        break;                                                                               \
      case myops::MYOPS_DTYPE_HALF:                                                          \
        launchUnaryKernelImpl<__half, upper##Op<__half>>(                                    \
            static_cast<__half *>(out), static_cast<const __half *>(input), n, stream);      \
        break;                                                                               \
      case myops::MYOPS_DTYPE_BFLOAT16:                                                      \
        launchUnaryKernelImpl<__nv_bfloat16, upper##Op<__nv_bfloat16>>(                      \
            static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(input), n, \
            stream);                                                                         \
        break;                                                                               \
      default:                                                                               \
        MYOPS_CHECK_FAILED("Only support float32, bfloat16 and half.");                      \
    }                                                                                        \
  }

FOR_EACH_UNARY_OP(DEFINE_UNARY_KERNEL)

}  // namespace myops
