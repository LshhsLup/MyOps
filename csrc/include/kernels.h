#pragma once

#include "common.h"

namespace myops {
// matmul
void launchMatmulKernel(void *c,
                        const void *a,
                        const void *b,
                        int m,
                        int n,
                        int k,
                        cudaStream_t stream,
                        MyOpsDtype dtype);

void launchMatTransKernel(void *out, void *in, int rows, int cols, cudaStream_t stream, MyOpsDtype dtype);

// unary ops - name(lower), Name(upper)
#define FOR_EACH_UNARY_OP(_) \
  _(abs, Abs)                \
  _(neg, Neg)                \
  _(exp, Exp)                \
  _(log, Log)                \
  _(relu, Relu)              \
  _(sigmoid, Sigmoid)

#define DECLARE_UNARY_KERNEL(lower, upper)                                                \
  void launch##upper##Kernel(void *out, const void *input, size_t n, cudaStream_t stream, \
                             MyOpsDtype dtype);

FOR_EACH_UNARY_OP(DECLARE_UNARY_KERNEL)

// binary ops - name(lower), Name(upper)
#define FOR_EACH_BINARY_OP(_) \
  _(add, Add)                 \
  _(sub, Sub)                 \
  _(mul, Mul)                 \
  _(div, Div)

#define DECLARE_BINARY_KERNEL(lower, upper)                                               \
  void launch##upper##Kernel(void *out, const void *a, const void *b, size_t n,           \
                             cudaStream_t stream, MyOpsDtype dtype);

FOR_EACH_BINARY_OP(DECLARE_BINARY_KERNEL)

}  // namespace myops
