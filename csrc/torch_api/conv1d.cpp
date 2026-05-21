#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

void conv1d(const at::Tensor &input, const at::Tensor &kernel, at::Tensor &output) {
  checkTensorSameAttr<TensorAttr::ALL>(input, kernel, output);
  checkTensorContiguous("conv1d - all tensors must be contiguous", input, kernel, output);

  TORCH_CHECK(input.dim() == 1, "conv1d - input must be 1D");
  TORCH_CHECK(kernel.dim() == 1, "conv1d - kernel must be 1D");
  TORCH_CHECK(kernel.size(0) <= input.size(0), "conv1d - kernel_size must be <= input_size");

  const auto input_size = input.size(0);
  const auto kernel_size = kernel.size(0);
  const auto output_size = input_size - kernel_size + 1;

  TORCH_CHECK(output.dim() == 1, "conv1d - output must be 1D");
  TORCH_CHECK(output.size(0) == output_size, "conv1d - output size mismatch");

  const auto dtype = scalarTypeToMyOpsDtype(input.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();

  DeviceGuard guard(input);
  launchConv1dKernel(output.data_ptr(), input.data_ptr(), kernel.data_ptr(), input_size,
                     kernel_size, stream, dtype);
}

}  // namespace torch_api
}  // namespace myops