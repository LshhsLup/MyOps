import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_val = tl.load(a_ptr + offset, mask=mask)
    b_val = tl.load(b_ptr + offset, mask=mask)
    c_val = a_val + b_val
    tl.store(c_ptr + offset, c_val, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"

    c = torch.empty_like(a) if c is None else c
    assert c.shape == a.shape, "Output tensor must have the same shape as input tensors"
    assert c.dtype == a.dtype, "Output tensor must have the same dtype as input tensors"
    n_elements = a.numel()

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return c