#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "../include/kernels.h"

#define threadsPerBlock 256

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

template <typename scalar_t>
struct AbsOp {
  __device__ scalar_t operator()(scalar_t x) const { return x < 0 ? -x : x; }
};

template <typename scalar_t>
struct NegOp {
  __device__ scalar_t operator()(scalar_t x) const { return -x; }
};

template <typename scalar_t>
struct ExpOp {
  __device__ scalar_t operator()(scalar_t x) const { return exp(x); }
};

template <typename scalar_t>
struct LogOp {
  __device__ scalar_t operator()(scalar_t x) const { return log(x); }
};

template <typename scalar_t>
struct ReluOp {
  __device__ scalar_t operator()(scalar_t x) const { return x > scalar_t(0) ? x : scalar_t(0); }
};

template <typename scalar_t>
struct SigmoidOp {
  __device__ scalar_t operator()(scalar_t x) const { return 1 / (1 + exp(-x)); }
};

#define LAUNCH_UNARY(op_name, op_struct)                                                       \
  void launch##op_name##Kernel(torch::Tensor out, torch::Tensor input) {                       \
    size_t n = input.numel();                                                                  \
    int device;                                                                                \
    cudaGetDevice(&device);                                                                    \
    int num_sms;                                                                               \
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);                  \
    auto stream = at::cuda::getCurrentCUDAStream();                                            \
    const int blocks = num_sms * 4;                                                            \
    AT_DISPATCH_ALL_TYPES_AND2(                                                                \
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), #op_name, ([&] {  \
          unaryKernel<scalar_t, op_struct<scalar_t>><<<blocks, threadsPerBlock, 0, stream>>>(  \
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), n, op_struct<scalar_t>{}); \
        }));                                                                                   \
  }

LAUNCH_UNARY(Abs, AbsOp)
LAUNCH_UNARY(Neg, NegOp)
LAUNCH_UNARY(Exp, ExpOp)
LAUNCH_UNARY(Log, LogOp)
LAUNCH_UNARY(Relu, ReluOp)
LAUNCH_UNARY(Sigmoid, SigmoidOp)