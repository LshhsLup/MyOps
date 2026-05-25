#pragma once

#include <torch/extension.h>

namespace myops {
namespace torch_api {

// ============================================================================
// Operator Declarations (torch_api adapter layer)
// ============================================================================

void matmul(const at::Tensor &a, const at::Tensor &b, at::Tensor &c);
void matrix_transpose(const at::Tensor &input, at::Tensor &out);
void conv1d(const at::Tensor &input, const at::Tensor &kernel, at::Tensor &output);
void reduce_sum(const at::Tensor &input, at::Tensor &out);

// unary ops
void abs(const at::Tensor &input, at::Tensor &out);
void neg(const at::Tensor &input, at::Tensor &out);
void exp(const at::Tensor &input, at::Tensor &out);
void log(const at::Tensor &input, at::Tensor &out);
void relu(const at::Tensor &input, at::Tensor &out);
void sigmoid(const at::Tensor &input, at::Tensor &out);

// binary ops
void add(const at::Tensor &a, const at::Tensor &b, at::Tensor &out);
void sub(const at::Tensor &a, const at::Tensor &b, at::Tensor &out);
void mul(const at::Tensor &a, const at::Tensor &b, at::Tensor &out);
void div(const at::Tensor &a, const at::Tensor &b, at::Tensor &out);

}  // namespace torch_api
}  // namespace myops