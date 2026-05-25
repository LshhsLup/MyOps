#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

void reduce_sum(const at::Tensor &input, at::Tensor &out) {
  CHECK_TENSOR_CONTIGUOUS(input);
  CHECK_TENSOR_CONTIGUOUS(out);
  checkTensorSameAttr<TensorAttr::ALL>(input, out);

  TORCH_CHECK(input.dim() == 1, "reduce_sum - input must be 1D");
  TORCH_CHECK(out.dim() == 1, "reduce_sum - out must be 1D");
  TORCH_CHECK(out.size(0) == 1, "reduce_sum - out must have size [1]");

  const auto N = input.size(0);
  const auto dtype = scalarTypeToMyOpsDtype(input.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();

  DeviceGuard guard(input);
  launchReduceSumKernel(out.data_ptr(), input.data_ptr(), N, stream, dtype);
}

}  // namespace torch_api
}  // namespace myops
