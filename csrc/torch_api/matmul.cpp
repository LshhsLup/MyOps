#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

void matmul(const at::Tensor &a, const at::Tensor &b, at::Tensor &c) {
  checkTensorSameAttr<TensorAttr::ALL>(a, b);
  checkTensorContiguous("matmul - all tensors must be contiguous", a, b, c);

  TORCH_CHECK(a.dim() == 2, "matmul - a must be 2D");
  TORCH_CHECK(b.dim() == 2, "matmul - b must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "matmul - inner dimensions must agree");

  const auto m = a.size(0);
  const auto k = a.size(1);
  const auto n = b.size(1);
  TORCH_CHECK(c.dim() == 2, "matmul - c must be 2D");
  TORCH_CHECK(c.size(0) == m && c.size(1) == n, "matmul - c shape mismatch");

  const auto dtype = scalarTypeToMyOpsDtype(a.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();

  DeviceGuard guard(a);
  launchMatmulKernel(c.data_ptr(), a.data_ptr(), b.data_ptr(), m, n, k, stream, dtype);
}

}  // namespace torch_api
}  // namespace myops