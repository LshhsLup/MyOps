# MyOps

A custom CUDA operator library for PyTorch that implements high-performance kernels with benchmarking capabilities against PyTorch's built-in operations. **This is a toy project, mainly for learning how to write a CUDA kernel and facilitating performance testing.**

## Features

- **Custom CUDA Kernels**: Optimized implementations for common operations
- **Easy Registration Pattern**: Simple operator registry system for adding new kernels
- **Performance Benchmarking**: Built-in benchmarking with TFLOPS and bandwidth metrics
- **Accuracy Validation**: Automatic verification against PyTorch reference implementations
- **Multi-Dtype Support**: Supports float32, float16, and bfloat16 (depending on operator)

## Requirements

- CUDA Toolkit (targeting sm_80 architecture)
- PyTorch with CUDA support
- Python 3.7+
- Ninja build system

## Installation

### Build the C++/CUDA Extension

```bash
# Build the extension in-place
python setup.py build_ext --inplace
```

The extension will be compiled with the following optimizations:
- `-O3` optimization level
- `--use_fast_math` for faster math operations
- `-arch=sm_80` for NVIDIA Ampere architecture

### Install as Python Package

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
import myops

# Element-wise addition
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = myops.add(a, b)  # Equivalent to torch.add(a, b)

# Matrix multiplication
a = torch.randn(1024, 512, device='cuda')
b = torch.randn(512, 1024, device='cuda')
c = myops.matmul(a, b)  # Shape: [1024, 1024]
```

### Running Benchmarks

The project includes a comprehensive benchmarking script that compares custom kernels against PyTorch implementations:

```bash
# Run all operator benchmarks
python run.py

# Bench specific operator
python run.py --op VectorAdd

# Benchmark only (skip accuracy checks)
python run.py --bench_only

# Adjust number of benchmark repeats
python run.py --repeats 100
```

### Benchmark Output

The benchmark displays:
- **Mine(ms)**: Average execution time of custom kernel
- **Torch(ms)**: Average execution time of PyTorch reference
- **TFLOPS**: Computational throughput
- **GB/s**: Memory bandwidth utilization (normalized to 1555 GB/s theoretical peak)

```
===================================================================================
                          CUDA OPERATOR PERFORMANCE DASHBOARD
===================================================================================
🚀 Operator: VectorAdd
Shape                     | Dtype        | Mine(ms)    | Torch(ms)   | TFLOPS    | GB/s
-----------------------------------------------------------------------------------------------
(1000,)                   | float32      | 0.0023      | 0.0025      | 0.43      | 0.52
```

## Adding New Operators

### 1. Write the CUDA Kernel

Create a new file in `csrc/kernels/` (e.g., `myop.cu`) with your kernel implementation:
```cpp
template<typename T>
__global__ void myOpKernel(T* out, const T* a, const T* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];  // Example: element-wise multiplication
    }
}

template<typename T>
void launchMyOpKernel(torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b) {
    int n = a.numel();
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    myOpKernel<T><<<grid, block>>>(out.data_ptr<T>(), a.data_ptr<T>(), b.data_ptr<T>(), n);
}
```

### 2. Add Python Bindings

Add bindings in [csrc/binding.cpp](csrc/binding.cpp):
```cpp
m.def("myop", [](torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    DISPATCH_FLOATING_TYPES(a.scalar_type(), "myop", [&] {
        launchMyOpKernel<scalar_t>(out, a, b);
    });
    return out;
}, "Element-wise multiplication");
```

### 3. Register Operator

Add to [myops/registry.py](myops/registry.py):
```python
OPERATOR_REGISTRY.append({
    "name": "MyOp",
    "dtypes": [torch.float32, torch.float16],
    "shapes": [(10**i,) for i in range(3, 10)],
    "my_func": lambda a, b: myops.myop(a, b),
    "torch_func": torch.mul,
    "data_gen": gen_data_elementwise,
    "get_flops": lambda shape: shape[0],
    "get_bytes": lambda shape, dtype_size: shape[0] * 3 * dtype_size
})
```

### 4. Rebuild and Test
```bash
python setup.py build_ext --inplace
python run.py --op MyOp
```

## Project Structure

```
MyOps/
├── csrc/
│   ├── binding.cpp          # Pybind11 bindings
│   ├── include/
│   │   └── kernels.h        # Kernel declarations & dispatch macros
│   └── kernels/
│       ├── add.cu           # Element-wise addition kernel
│       └── matmul.cu        # Matrix multiplication kernel
├── myops/
│   ├── __init__.py          # Module initialization with attribute forwarding
│   └── registry.py          # Operator registry for benchmarks
├── setup.py                 # Build configuration for CUDA extension
├── run.py                   # Benchmarking script
└── README.md                # This file
```

## Implemented Operators

| Operator | Description | Supported Data Types |
|----------|-------------|---------------------|
| `add`    | Element-wise vector addition | float32, float16, bfloat16 |
| `matmul` | Matrix multiplication (M×K @ K×N) | float32 |

## Architecture Notes

- **Target GPU**: NVIDIA Ampere (sm_80 architecture)
- **Thread Block Size**: 256 threads per block (default)
- **Compile Optimization**: `-O3` with fast math enabled
- **Incremental Build**: Relocatable device code can be enabled in [setup.py](setup.py:30) for faster compilation with multiple operators

## License

This project is provided as-is for educational and research purposes.