# MyOps

> For Learning & Practice Only - This is a personal project for studying CUDA kernel development.

A high-performance CUDA operator library for PyTorch — designed for **rapid kernel development, and learning CUDA programming**.

## Why MyOps?

The fun part of CUDA is writing the kernel. The tedious part is everything else: host memory management, dtype dispatch, test harnesses, benchmark scaffolding. MyOps eliminates that overhead by embedding directly into the PyTorch ecosystem — because PyTorch already solved these problems, so you don't have to.

- **Zero Infrastructure** — Host allocations, tensor layouts, dtype routing, tests, and benchmarks are all handled by PyTorch. You write the kernel and a thin Python binding, then call it naturally: `myops.matmul(a, b)`. No C++ operator registration, no manual memory bookkeeping.

- **Instant Verification** — Validate correctness and measure speed against PyTorch's native implementation in one line of Python. No custom test suites or benchmark frameworks to maintain.

- **Learn by Measuring** — Every operator demonstrates the full optimization arc, from naive to advanced (vectorized memory access, shared-memory tiling, warp-level primitives). Each step is benchmarked head-to-head against PyTorch, so you see exactly which technique earns how much performance and why.

## Requirements

- **CUDA Toolkit** 11.0+
- **PyTorch** >= 2.0 (with CUDA support)
- **Python** 3.8+
- **Ninja** build system (`pip install ninja`)
- **GCC** with C++17 support
- **NVIDIA GPU** with compute capability 7.0+ (V100 / RTX 20 / A100 / RTX 40 / H100, etc.)

## Installation

### 1. Configure PyTorch Shared Library Path (CRITICAL)

Custom extensions need to link against PyTorch's internal C++ libraries (like libc10.so and libtorch_cpu.so). Ensure your dynamic linker can find them by setting LD_LIBRARY_PATH:

```bash
# Set for current session
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# (Optional) Persist to ~/.bashrc so you don't have to re-run it in new shells
echo "export LD_LIBRARY_PATH=\$(python -c \"import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\"): \$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install in Editable Mode
Choose Method A for a standard editable pip mapping, or Method B for standalone local kernel hacking without tool interference.

#### Method A: Standard Editable Install (via pip/uv)

Forces the manager to skip the empty environment sandbox and compile in your real system context:
```bash
# Using standard pip
python -m pip install -e . --no-build-isolation

# Using fast modern 'uv' (requires system flag if outside venv)
uv pip install -e . --system
```

#### Method B: Local In-Place Compilation (Best for High-Frequency Kernel Dev)
Bypasses front-end managers completely. Compiles .cu files via Ninja/NVCC and outputs the binary _core.so directly into the myops/ source folder:

```bash
python setup.py build_ext --inplace
```

### 3. Verify Installation

```bash
python -c "import myops; print('MyOps initialized successfully! Core extension:', myops._core)"
```
This compiles all CUDA kernels with `-O3 --use_fast_math --extended-lambda` and installs the `myops` package in editable mode.

## Usage

```python
import torch, myops

a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
# Custom ops
c = myops.matmul(a, b)
```

### Tests

```bash
pytest tests/ -v                # all ops
pytest tests/test_matmul.py -v  # single op
```

### Benchmarks

```bash
python benchmark/benchmark_matmul.py
python benchmark/benchmark_<op>.py
```

Benchmarks compare MyOps kernels against PyTorch native implementations, reporting latency and speedup.

## Adding a New Operator

### Option 1: `/adapt-op` (Recommended)

Write your CUDA kernel in `csrc/kernels/`, then use **Claude Code**:

```
/adapt-op csrc/kernels/your_kernel.cu, please help me adapt this operator.
```

Claude Code will automatically:
1. Analyze the kernel (signature, category, supported dtypes)
2. Update C++ headers and adapters (`kernels.h`, `torch_api.h`, `register.cpp`, adapter `.cpp`)
3. Add Python wrapper to `myops/ops.py`
4. Generate `tests/test_<op>.py` with parametrized correctness tests
5. Generate `benchmark/benchmark_<op>.py` with torch baseline comparison
6. Build and verify

### Option 2: Custom ops

1. **`csrc/include/kernels.h`** — Declare `launchXxxKernel()`
2. **`csrc/torch_api/torch_api.h`** — Declare adapter function
3. **`csrc/torch_api/register.cpp`** — Add schema + impl registration
4. **`csrc/torch_api/<op>.cpp`** — Implement adapter (checks, dtype dispatch, kernel launch)
5. **`myops/ops.py`** — Add Python wrapper + `__all__` entry
6. Write `tests/test_<op>.py` and `benchmark/benchmark_<op>.py`
7. `pip install -e . && pytest`

Follow the `matmul` pattern as a reference.

## Project Structure

```
csrc/
  include/          # Shared headers, op trait definitions, kernel launch helpers
  kernels/          # CUDA kernel implementations (.cu)
  torch_api/        # C++ adapter layer (PyTorch tensor → kernel dispatch)
myops/
  ops.py            # Python API with ergonomic signatures
  cuda/             # Re-exports for myops.cuda.xxx usage
  triton/           # Triton reference implementations
tests/              # Parametrized correctness tests per op
benchmark/          # Latency benchmarks vs PyTorch baseline
```