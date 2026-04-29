# MyOps

A custom CUDA/Triton operator library for PyTorch that implements high-performance kernels with unified benchmarking comparing Triton, CUDA, and PyTorch implementations.

## Features

- **Custom CUDA Kernels**: Hand-written CUDA kernels for common operations
- **Custom Triton Kernels**: Triton implementations for comparison
- **Unified Benchmark**: Side-by-side comparison of Triton vs CUDA vs PyTorch
- **Easy Registration Pattern**: Simple operator registry for adding new kernels
- **Accuracy Validation**: Automatic verification against PyTorch reference

## Requirements

- CUDA Toolkit (targeting sm_80 architecture)
- PyTorch with CUDA support
- Python 3.8+
- Triton
- Ninja build system

## Installation

```bash
# Build the CUDA extension in-place
python setup.py build_ext --inplace

# Or install as a package
pip install -e .
```

## Project Structure

```
myops/
├── __init__.py          # Module init, attribute forwarding to _core
├── registry.py          # Operator registry for unified benchmarking
├── cuda/                # CUDA kernel wrappers
│   └── __init__.py
└── triton/              # Triton kernel implementations
    └── __init__.py
csrc/
├── binding.cpp          # PyBind11 bindings for CUDA kernels
├── include/
│   └── kernels.h
└── kernels/             # CUDA kernel source
    ├── add.cu
    └── matmul.cu
run.py                   # Unified benchmarking script
```

## Usage

### Writing a New Kernel

**1. CUDA Kernel** (`csrc/kernels/your_kernel.cu`):
```cpp
// Implement your kernel
template<typename T>
__global__ void yourKernel(T* out, const T* a, const T* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
```

**2. Bindings** (`csrc/binding.cpp`):
```cpp
m.def("your_kernel", [](torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    launchYourKernel(out, a, b);
    return out;
}, "Your kernel description");
```

**3. Triton Kernel** (`myops/triton/your_kernel.py`):
```python
import triton
import triton.language as tl

@triton.jit
def your_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    # ... kernel implementation
```

**4. Register** (`myops/registry.py`):
```python
{
    "name": "YourOp",
    "dtypes": [torch.float32, torch.float16],
    "shapes": [(10**i,) for i in range(3, 8)],
    "cuda_func": lambda a, b: myops.cuda.your_kernel(a, b),
    "triton_func": lambda a, b: myops.triton.your_kernel(a, b),
    "torch_func": torch.add,
    "gen_data": gen_data_elementwise,
    "get_flops": lambda shape: shape[0],
    "get_bytes": lambda shape, dtype_size: shape[0] * 3 * dtype_size,
}
```

### Running Benchmarks

```bash
# Benchmark all operators
python run.py

# Benchmark specific operator
python run.py --op VectorAdd

# Benchmark only (skip accuracy checks)
python run.py --bench_only

# Adjust number of repeats
python run.py --repeats 100
```

### Benchmark Output

```
==============================================================================================================
                                         KERNEL PERFORMANCE BENCHMARK
==============================================================================================================
Shape                | Dtype      | Triton(ms)   | CUDA(ms)     | Torch(ms)    | Triton Spd | CUDA Spd
--------------------------------------------------------------------------------------------------------------

VectorAdd
(1000,)              | float32    | 0.0381       | 0.0106       | 0.0111       | 0.29      x | 1.04      x
(10000,)             | float32    | 0.0356       | 0.0123       | 0.0113       | 0.32      x | 0.92      x
...
```

Columns:
- **Triton(ms)**: Triton kernel execution time
- **CUDA(ms)**: Custom CUDA kernel execution time
- **Torch(ms)**: PyTorch reference implementation time
- **Triton Spd**: Triton speedup vs PyTorch
- **CUDA Spd**: CUDA kernel speedup vs PyTorch

## Implemented Operators

| Operator | Description | Supported Data Types |
|----------|-------------|---------------------|
| `VectorAdd` | Element-wise vector addition | float32, float16, bfloat16 |

## Architecture Notes

- **Target GPU**: NVIDIA Ampere (sm_80)
- **Thread Block Size**: 256 threads (CUDA)
- **Triton Block Size**: 1024 elements
- **Compile Optimization**: `-O3` with `--use_fast_math`