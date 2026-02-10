#include "../include/kernels.h"
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void matmulKernel(
    scalar_t* __restrict__ c,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const int m, const int n, const int k
) {
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx < m && col_idx < n) {
    float sum = 0.0f;
    for(int i = 0; i < k; ++i) {
      sum += static_cast<float>(a[row_idx * k + i]) *
             static_cast<float>(b[i * n + col_idx]);
    }
    c[row_idx * n + col_idx] = static_cast<scalar_t>(sum);
  }
}

// [M, K] @ [K, N]
void launchMatmulKernel(torch::Tensor C, 
                        torch::Tensor A,
                        torch::Tensor B) {
  const int m = A.size(0);
  const int n = B.size(1);
  const int k = A.size(1);
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid(
    (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (m + threadsPerBlock.y - 1) / threadsPerBlock.y 
  );
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    A.scalar_type(), "matmul", ([&]{
      matmulKernel<scalar_t><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        C.data_ptr<scalar_t>(),
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        m, n, k
      );
    })
  );
}