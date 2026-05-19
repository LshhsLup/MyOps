from functools import partial

import torch
import triton
import triton.language as tl

__all__ = ["add", "sub", "mul", "div"]


@triton.jit
def binary_template_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    if OP == "add":
        result = a + b
    elif OP == "sub":
        result = a - b
    elif OP == "mul":
        result = a * b
    elif OP == "div":
        result = a / b
    else:
        result = a

    tl.store(out_ptr + offsets, result, mask=mask)


def apply_binary_op(a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor:
    assert a.is_cuda, "Input must be on CUDA device"
    assert a.shape == b.shape, "Input tensors must have the same shape"
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    binary_template_kernel[grid](a, b, out, n_elements, OP=op, BLOCK_SIZE=BLOCK_SIZE)
    return out


add = partial(apply_binary_op, op="add")
sub = partial(apply_binary_op, op="sub")
mul = partial(apply_binary_op, op="mul")
div = partial(apply_binary_op, op="div")
