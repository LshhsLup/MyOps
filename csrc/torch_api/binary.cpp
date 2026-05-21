#include <unordered_map>
#include "include/kernels.h"
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

using BinaryKernelFunc =
    void (*)(void *, const void *, const void *, size_t, cudaStream_t, MyOpsDtype);

BinaryKernelFunc getBinaryKernelFunc(const std::string &name) {
#define REGISTER_BINARY_KERNEL(lower, upper) {#lower, launch##upper##Kernel},
  static const std::unordered_map<std::string, BinaryKernelFunc> kernels = {
      FOR_EACH_BINARY_OP(REGISTER_BINARY_KERNEL)};
#undef REGISTER_BINARY_KERNEL
  auto it = kernels.find(name);
  TORCH_CHECK(it != kernels.end(), "Unknown binary op: ", name);
  return it->second;
}

void launchBinaryKernel(const at::Tensor &a,
                        const at::Tensor &b,
                        at::Tensor &out,
                        const std::string &name) {
  const auto dtype = scalarTypeToMyOpsDtype(a.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();
  DeviceGuard guard(a);
  getBinaryKernelFunc(name)(out.data_ptr(), a.data_ptr(), b.data_ptr(), a.numel(), stream, dtype);
}

#define DEFINE_BINARY_OP(lower, upper)                                            \
  void lower(const at::Tensor &a, const at::Tensor &b, at::Tensor &out) {         \
    checkTensorContiguous(#lower " - all tensors must be contiguous", a, b, out); \
    checkTensorSameAttr<TensorAttr::ALL>(a, b, out);                              \
    launchBinaryKernel(a, b, out, #lower);                                        \
  }

FOR_EACH_BINARY_OP(DEFINE_BINARY_OP)

}  // namespace torch_api
}  // namespace myops
