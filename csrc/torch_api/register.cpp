#include <pybind11/pybind11.h>
#include <torch/library.h>
#include "torch_api.h"
#include "utils.h"

namespace myops {
namespace torch_api {

TORCH_LIBRARY_FRAGMENT(myops, m) {
  // torch 2.1.0 does not support impl_abstract_pystub, enable it when torch 2.3.0 is released
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
  m.impl_abstract_pystub("myops.abstract");
#endif
  m.def(TORCH_SELECTIVE_SCHEMA("myops::matmul(Tensor a, Tensor b, Tensor(c!) c) -> ()"));
  // unary ops
  m.def(TORCH_SELECTIVE_SCHEMA("myops::abs(Tensor input, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::neg(Tensor input, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::exp(Tensor input, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::log(Tensor input, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::relu(Tensor input, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::sigmoid(Tensor input, Tensor(a!) out) -> ()"));
  // binary ops
  m.def(TORCH_SELECTIVE_SCHEMA("myops::add(Tensor a, Tensor b, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::sub(Tensor a, Tensor b, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::mul(Tensor a, Tensor b, Tensor(a!) out) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA("myops::div(Tensor a, Tensor b, Tensor(a!) out) -> ()"));
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("myops::matmul"), TORCH_FN(matmul));
  // unary ops
  m.impl(TORCH_SELECTIVE_NAME("myops::abs"), TORCH_FN(abs));
  m.impl(TORCH_SELECTIVE_NAME("myops::neg"), TORCH_FN(neg));
  m.impl(TORCH_SELECTIVE_NAME("myops::exp"), TORCH_FN(exp));
  m.impl(TORCH_SELECTIVE_NAME("myops::log"), TORCH_FN(log));
  m.impl(TORCH_SELECTIVE_NAME("myops::relu"), TORCH_FN(relu));
  m.impl(TORCH_SELECTIVE_NAME("myops::sigmoid"), TORCH_FN(sigmoid));
  // binary ops
  m.impl(TORCH_SELECTIVE_NAME("myops::add"), TORCH_FN(add));
  m.impl(TORCH_SELECTIVE_NAME("myops::sub"), TORCH_FN(sub));
  m.impl(TORCH_SELECTIVE_NAME("myops::mul"), TORCH_FN(mul));
  m.impl(TORCH_SELECTIVE_NAME("myops::div"), TORCH_FN(div));
}

}  // namespace torch_api
}  // namespace myops

// Pybind11 module entry point
PYBIND11_MODULE(_core, m) {
  m.doc() = "MyOps CUDA operators";
  // Operators are registered via TORCH_LIBRARY above
}
