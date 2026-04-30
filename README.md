# MyOps

> ⚠️ **For Learning & Practice Only** - This is a personal project for studying CUDA/Triton kernel development.

A high-performance CUDA/Triton operator library for PyTorch with unified benchmarking framework.

## Features

- **Dual Kernel Implementation**: Both CUDA and Triton implementations for each operator
- **Unified Benchmarking**: Side-by-side comparison with PyTorch reference using Rich framework
- **Color-coded Results**: Green for speedup, red for slowdown vs PyTorch
- **Automatic Verification**: correctness validation against PyTorch reference
- **Template-based Design**: Reusable CUDA kernel templates for unary operators
- **PyTorch Integration**: Seamless access via `torch.ops.myops`

## Requirements

- CUDA Toolkit (sm_80 architecture)
- PyTorch with CUDA support
- Python 3.8+
- Triton
- Rich (for benchmark output)

## Installation

```bash
pip install -e .
```

Or

```bash
python setup.py build_ext --inplace
```


## Project Structure

```
myops/
├── __init__.py           # Module init, exports ops
├── ops.py                # Python API (add, matmul, abs, neg, exp, log, relu, sigmoid)
├── registry.py           # Operator registry and benchmark config
├── cuda/                 # CUDA kernel wrappers
│   └── __init__.py
└── triton/               # Triton kernel implementations
    └── __init__.py

csrc/
├── include/              # C++ headers
│   ├── kernels.h         # CUDA kernel declarations
│   └── common.h          # Common types and macros
├── kernels/              # CUDA kernel source
│   ├── add.cu            # Element-wise addition
│   ├── matmul.cu         # Matrix multiplication
│   └── unary.cu          # Unary ops (abs, neg, exp, log, relu, sigmoid)
└── torch_api/            # PyTorch C++ extension adapter
    ├── torch_api.h       # PyTorch tensor utilities
    ├── utils.h           # Helper functions
    ├── add.cpp           # add operator binding
    ├── matmul.cpp        # matmul operator binding
    ├── unary.cpp         # unary operators binding
    └── register.cpp      # Operator registration

run.py                    # Benchmark script
setup.py                  # Build configuration
```

## Usage

### Running Benchmarks

```bash
# Benchmark all operators
python run.py

# Benchmark specific operator
python run.py --op add

# Benchmark only (skip accuracy checks)
python run.py --bench_only

# Adjust number of repeats
python run.py --repeats 100
```

### Python API

```python
import myops

# Binary operator
c = myops.add(a, b)

# Unary operators
y = myops.abs(x)
y = myops.neg(x)
y = myops.exp(x)
y = myops.log(x)
y = myops.relu(x)
y = myops.sigmoid(x)

# Matrix multiplication
c = myops.matmul(a, b)
```

### Benchmark Output

```
╭────────────────────────────────────────────────────────────── ADD ───────────────────────────────────────────────────────────────╮
│      Shape         Dtype           Triton               CUDA            Torch (ref)       Triton vs Torch       CUDA vs Torch    │
│     (1000,)       float32       0.038912 ms         0.012800 ms         0.009216 ms            ↓0.24x              ↓0.72x        │
│     (1000,)       float16       0.036659 ms         0.012595 ms         0.009114 ms            ↓0.25x              ↓0.72x        │
│     (1000,)       bfloat16      0.035942 ms         0.012390 ms         0.008806 ms            ↓0.25x              ↓0.71x        │
│    (10000,)       float32       0.035430 ms         0.012493 ms         0.008909 ms            ↓0.25x              ↓0.71x        │
│    (10000,)       float16       0.035328 ms         0.012493 ms         0.009011 ms            ↓0.26x              ↓0.72x        │
│    (10000,)       bfloat16      0.036250 ms         0.012390 ms         0.008704 ms            ↓0.24x              ↓0.70x        │
│    (100000,)      float32       0.034816 ms         0.012390 ms         0.008806 ms            ↓0.25x              ↓0.71x        │
│    (100000,)      float16       0.034918 ms         0.012595 ms         0.008806 ms            ↓0.25x              ↓0.70x        │
│    (100000,)      bfloat16      0.034304 ms         0.012186 ms         0.008704 ms            ↓0.25x              ↓0.71x        │
│   (1000000,)      float32       0.034406 ms         0.012800 ms         0.010035 ms            ↓0.29x              ↓0.78x        │
│   (1000000,)      float16       0.034918 ms         0.013722 ms         0.008909 ms            ↓0.26x              ↓0.65x        │
│   (1000000,)      bfloat16      0.035226 ms         0.012493 ms         0.008602 ms            ↓0.24x              ↓0.69x        │
│   (10000000,)     float32       0.096461 ms         0.095437 ms         0.093082 ms            ↓0.96x              ↓0.98x        │
│   (10000000,)     float16       0.053453 ms         0.051814 ms         0.050688 ms            ↓0.95x              ↓0.98x        │
│   (10000000,)     bfloat16      0.053146 ms         0.051507 ms         0.050790 ms            ↓0.96x              ↓0.99x        │
│  (100000000,)     float32       0.863130 ms         0.879718 ms         0.860160 ms            ↓1.00x              ↓0.98x        │
│  (100000000,)     float16       0.439706 ms         0.446054 ms         0.437862 ms            ↓1.00x              ↓0.98x        │
│  (100000000,)     bfloat16      0.441139 ms         0.444416 ms         0.437453 ms            ↓0.99x              ↓0.98x        │
│  (1000000000,)    float32       8.754585 ms         8.698880 ms         8.752845 ms            ↓1.00x              ↑1.01x        │
│  (1000000000,)    float16       4.415898 ms         4.390912 ms         4.406681 ms            ↓1.00x              ↑1.00x        │
│  (1000000000,)    bfloat16      4.420096 ms         4.361626 ms         4.400947 ms            ↓1.00x              ↑1.01x        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
...

Columns:
- **Triton/CUDA/Torch**: Kernel execution time
- **Speedup**: Ratio vs PyTorch (green=up, red=down)
```

## Implemented Operators

### Binary Operators

| Operator | Description | Dtypes |
|----------|-------------|--------|
| `add` | Element-wise addition | float32, float16, bfloat16 |
| `matmul` | Matrix multiplication | float32, float16, bfloat16 |

### Unary Operators

| Operator | Description | Dtypes |
|----------|-------------|--------|
| `abs` | Absolute value | float32, float16, bfloat16 |
| `neg` | Numerical negative | float32, float16, bfloat16 |
| `exp` | Exponential | float32 |
| `log` | Natural logarithm | float32 |
| `relu` | ReLU activation | float32, float16, bfloat16 |
| `sigmoid` | Sigmoid activation | float32 |

## Adding a New Operator

**1. CUDA Kernel** (`csrc/kernels/your_op.cu`):
```cpp
template <typename scalar_t>
__global__ void yourKernel(scalar_t* out, const scalar_t* input, size_t n) {
    // implementation
}

template <typename scalar_t>
cudaError_t launchYourKernel(scalar_t* out, const scalar_t* input, size_t n, cudaStream_t stream) {
    yourKernel<<<blocks, threadsPerBlock, 0, stream>>>(out, input, n);
    return cudaGetLastError();
}
```

**2. Torch Binding** (`csrc/torch_api/your_op.cpp`):
```cpp
torch::Tensor your_op(torch::Tensor input) {
    auto out = torch::empty_like(input);
    DISPATCH_DTYPE_AND_LAUNCH(input.scalar_type(), input.numel(), your_kernel, out, input);
    return out;
}
```

**3. Triton Kernel** (`myops/triton/your_op.py`):
```python
@triton.jit
def your_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # implementation
```

**4. Register** (`myops/registry.py`):
```python
OPERATOR_REGISTRY.append({
    "name": "your_op",
    "dtypes": [torch.float32, torch.float16],
    "shapes": [(10**i,) for i in range(3, 8)],
    "cuda_func": lambda x: myops.cuda.your_op(x),
    "triton_func": lambda x: myops.triton.your_op(x),
    "torch_func": torch.your_func,
    "gen_data": gen_data_unary,
})
```

## Architecture Notes

- **Target GPU**: NVIDIA Ampere (sm_80)
- **CUDA Thread Block**: 256 threads
- **Triton Block Size**: 1024 elements
- **CUDA Compile**: `-O3 --use_fast_math -arch=sm_80`
- **C++ Standard**: C++17