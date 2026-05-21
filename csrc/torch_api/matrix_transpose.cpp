#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

void matrix_transpose(const at::Tensor &input, at::Tensor &out) {
  checkTensorSameAttr<TensorAttr::ALL>(input, out);
  checkTensorContiguous("matrix_transpose - all tensors must be contiguous", input, out);

  TORCH_CHECK(input.dim() == 2, "matrix_transpose - input must be 2D");
  TORCH_CHECK(out.dim() == 2, "matrix_transpose - out must be 2D");
  TORCH_CHECK(input.size(0) == out.size(1) && input.size(1) == out.size(0),
              "matrix_transpose - out shape must be transposed of input");

  const auto rows = input.size(0);
  const auto cols = input.size(1);
  const auto dtype = scalarTypeToMyOpsDtype(input.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();

  DeviceGuard guard(input);
  launchMatTransKernel(out.data_ptr(), input.data_ptr(), rows, cols, stream, dtype);
}

}  // namespace torch_api
}  // namespace myops