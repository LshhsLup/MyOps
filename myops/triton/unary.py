from functools import partial

import torch
import triton
import triton.language as tl

__all__ = ["abs", "neg", "exp", "log", "relu", "sigmoid"]


@triton.jit
def unary_template_kernel(
    output_ptr, input_ptr, n_elements, OP: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)

    if OP == "abs":
        result = tl.abs(x)
    elif OP == "neg":
        result = -x
    elif OP == "exp":
        result = tl.exp(x)
    elif OP == "log":
        result = tl.log(x)
    elif OP == "relu":
        result = tl.where(x > 0, x, 0.0)
    elif OP == "sigmoid":
        result = 1.0 / (1.0 + tl.exp(-x))
    else:
        result = x  # 默认 / Fallback

    tl.store(output_ptr + offsets, result, mask=mask)


def apply_unary_op(x: torch.Tensor, op: str) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    unary_template_kernel[grid](out, x, n_elements, OP=op, BLOCK_SIZE=BLOCK_SIZE)
    return out


abs = partial(apply_unary_op, op="abs")
neg = partial(apply_unary_op, op="neg")
exp = partial(apply_unary_op, op="exp")
log = partial(apply_unary_op, op="log")
relu = partial(apply_unary_op, op="relu")
sigmoid = partial(apply_unary_op, op="sigmoid")
