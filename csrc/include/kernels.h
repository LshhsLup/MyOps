#pragma once

#include <torch/extension.h>

// add
void launchAddKernel(torch::Tensor out, torch::Tensor a, torch::Tensor b);

// matmul
void launchMatmulKernel(torch::Tensor C, 
                        torch::Tensor A,
                        torch::Tensor B);