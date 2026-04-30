#pragma once

#include "common.h"

// add
cudaError_t launchAddKernel(void *out,
                            const void *a,
                            const void *b,
                            size_t n,
                            cudaStream_t stream,
                            MyOpsDtype dtype);

// matmul
cudaError_t launchMatmulKernel(void *c,
                               const void *a,
                               const void *b,
                               int m,
                               int n,
                               int k,
                               cudaStream_t stream,
                               MyOpsDtype dtype);

// unary ops - name(lower), Name(upper)
#define FOR_EACH_UNARY_OP(_) \
  _(abs, Abs)                \
  _(neg, Neg)                \
  _(exp, Exp)                \
  _(log, Log)                \
  _(relu, Relu)              \
  _(sigmoid, Sigmoid)

#define DECLARE_UNARY_KERNEL(lower, upper)                                                       \
  cudaError_t launch##upper##Kernel(void *out, const void *input, size_t n, cudaStream_t stream, \
                                    MyOpsDtype dtype);

FOR_EACH_UNARY_OP(DECLARE_UNARY_KERNEL)
