#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define threadsPerBlock 256

using vec_t = uint4;

template <typename T>
__device__ __forceinline__ constexpr int get_vec_size() {
  return 16 / sizeof(T);
}

template <typename scalar_t>
__global__ void addKernel(
           scalar_t* __restrict__ out,
           const scalar_t* __restrict__ x,
           const scalar_t* __restrict__ y,
           const size_t n) {
  constexpr int VEC_SIZE = get_vec_size<scalar_t>();
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);
  const vec_t* y_vec = reinterpret_cast<const vec_t*>(y);
  vec_t* o_vec = reinterpret_cast<vec_t*>(out);
  size_t n_vec = n / VEC_SIZE;
  for(size_t i = idx; i < n_vec; i += stride) {
    const vec_t x_val = x_vec[i];
    const vec_t y_val = y_vec[i];
    vec_t o_val;

    const scalar_t* s_x = reinterpret_cast<const scalar_t*>(&x_val);
    const scalar_t* s_y = reinterpret_cast<const scalar_t*>(&y_val);
    scalar_t* s_o = reinterpret_cast<scalar_t*>(&o_val);

    #pragma unroll
    for(int j = 0; j < VEC_SIZE; ++j) {
      s_o[j] = s_x[j] + s_y[j];
    }

    o_vec[i] = o_val; 
  }

  for (size_t i = n_vec * VEC_SIZE + idx; i < n; i += stride) {
    out[i] = x[i] + y[i];
  }
}

void launchAddKernel(torch::Tensor out, torch::Tensor a, torch::Tensor b) {
  size_t n = a.numel();
  int device;
  cudaGetDevice(&device);
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = num_sms * 4;

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, 
    a.scalar_type(), "add", ([&] {
      addKernel<scalar_t><<<blocks, threadsPerBlock, 0, stream>>>(
          out.data_ptr<scalar_t>(), 
          a.data_ptr<scalar_t>(), 
          b.data_ptr<scalar_t>(), 
          n
      );
    })
  );
}