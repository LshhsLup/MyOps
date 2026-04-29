#include "include/kernels.h"

PYBIND11_MODULE(_core, m) {
    m.def("add", [](torch::Tensor a, torch::Tensor b) {
        auto out = torch::empty_like(a);
        launchAddKernel(out, a, b);
        return out;
    }, "Add two tensors and return the result");
    m.def("add_", [](torch::Tensor a, torch::Tensor b) {
        launchAddKernel(a, a, b);
        return a;
    }, "Add b to a in-place");
    m.def("matmul", [](torch::Tensor a, torch::Tensor b) {
        auto m = a.size(0);
        auto k = a.size(1);
        auto n = b.size(1);
        TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions must agree");
        auto options = torch::TensorOptions().device(a.device()).dtype(a.dtype());
        auto c = torch::empty({m, n}, options);
        launchMatmulKernel(c, a, b);
        return c;
    }, "Matrix multiplication [M, K] @ [K, N]");

    // Unary element-wise ops
    m.def("abs", [](torch::Tensor x) {
        auto out = torch::empty_like(x);
        launchAbsKernel(out, x);
        return out;
    }, "Absolute value");
    m.def("neg", [](torch::Tensor x) {
        auto out = torch::empty_like(x);
        launchNegKernel(out, x);
        return out;
    }, "Numerical negative");
    m.def("exp", [](torch::Tensor x) {
        auto out = torch::empty_like(x);
        launchExpKernel(out, x);
        return out;
    }, "Exponential");
    m.def("log", [](torch::Tensor x) {
        auto out = torch::empty_like(x);
        launchLogKernel(out, x);
        return out;
    }, "Natural logarithm");
    m.def("relu", [](torch::Tensor x) {
        auto out = torch::empty_like(x);
        launchReluKernel(out, x);
        return out;
    }, "ReLU activation");
    m.def("sigmoid", [](torch::Tensor x) {
        auto out = torch::empty_like(x);
        launchSigmoidKernel(out, x);
        return out;
    }, "Sigmoid activation");
}