#include <unordered_map>
#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

using UnaryKernelFunc = void (*)(void *, const void *, size_t, cudaStream_t, MyOpsDtype);

UnaryKernelFunc getUnaryKernelFunc(const std::string &name) {
#define REGISTER_UNARY_KERNEL(lower, upper) {#lower, launch##upper##Kernel},
  static const std::unordered_map<std::string, UnaryKernelFunc> kernels = {
      FOR_EACH_UNARY_OP(REGISTER_UNARY_KERNEL)};
#undef REGISTER_UNARY_KERNEL
  auto it = kernels.find(name);
  TORCH_CHECK(it != kernels.end(), "Unknown unary op: ", name);
  return it->second;
}

void launchUnaryKernel(const at::Tensor &input, at::Tensor &out, const std::string &name) {
  const auto dtype = scalarTypeToMyOpsDtype(input.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();
  DeviceGuard guard(input);
  getUnaryKernelFunc(name)(out.data_ptr(), input.data_ptr(), input.numel(), stream, dtype);
}

#define DEFINE_UNARY_OP(lower, upper)                                              \
  void lower(const at::Tensor &input, at::Tensor &out) {                           \
    checkTensorContiguous(#lower " - all tensors must be contiguous", input, out); \
    checkTensorSameAttr<TensorAttr::ALL>(input, out);                              \
    launchUnaryKernel(input, out, #lower);                                         \
  }

FOR_EACH_UNARY_OP(DEFINE_UNARY_OP)

}  // namespace torch_api
}  // namespace myops
