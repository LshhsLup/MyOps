#include "include/kernels.h"

PYBIND11_MODULE(_backend, m) {
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
}
