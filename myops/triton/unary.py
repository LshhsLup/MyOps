import torch
import triton
import triton.language as tl


@triton.jit
def unary_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.abs(x)
    tl.store(output_ptr + offsets, result, mask=mask)


def abs(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    unary_kernel[grid](out, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def neg_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = -x
    tl.store(output_ptr + offsets, result, mask=mask)


def neg(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    neg_kernel[grid](out, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def exp_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.exp(x)
    tl.store(output_ptr + offsets, result, mask=mask)


def exp(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](out, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def log_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.log(x)
    tl.store(output_ptr + offsets, result, mask=mask)


def log(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    log_kernel[grid](out, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def relu_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.where(x > 0, x, 0)
    tl.store(output_ptr + offsets, result, mask=mask)


def relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](out, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def sigmoid_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = 1 / (1 + tl.exp(-x))
    tl.store(output_ptr + offsets, result, mask=mask)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](out, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out