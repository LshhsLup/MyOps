#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

void add(const at::Tensor &a, const at::Tensor &b, at::Tensor &out) {
  checkTensorSameAttr<TensorAttr::ALL>(a, b);
  checkTensorContiguous("add - all tensors must be contiguous", a, b, out);

  const auto dtype = scalarTypeToMyOpsDtype(a.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();

  DeviceGuard guard(a);
  launchAddKernel(out.data_ptr(), a.data_ptr(), b.data_ptr(), a.numel(), stream, dtype);
}

}  // namespace torch_api
}  // namespace myops