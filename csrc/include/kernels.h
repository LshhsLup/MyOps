#pragma once

#include <torch/extension.h>

// add
void launchAddKernel(torch::Tensor out, torch::Tensor a, torch::Tensor b);

// matmul
void launchMatmulKernel(torch::Tensor C,
                        torch::Tensor A,
                        torch::Tensor B);

// unary ops
void launchAbsKernel(torch::Tensor out, torch::Tensor input);
void launchNegKernel(torch::Tensor out, torch::Tensor input);
void launchExpKernel(torch::Tensor out, torch::Tensor input);
void launchLogKernel(torch::Tensor out, torch::Tensor input);
void launchReluKernel(torch::Tensor out, torch::Tensor input);
void launchSigmoidKernel(torch::Tensor out, torch::Tensor input);