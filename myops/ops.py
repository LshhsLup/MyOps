"""Python API for myops operators."""

import torch

__all__ = [
    "matmul",
    "matrix_transpose",
    "conv1d",
    "reduce_sum",
    "abs",
    "neg",
    "exp",
    "log",
    "relu",
    "sigmoid",
    "add",
    "sub",
    "mul",
    "div",
]


def _make_binary_op(name: str):
    def op(
        a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(a)
        getattr(torch.ops.myops, name)(a, b, out)
        return out

    op.__name__ = name
    return op


add = _make_binary_op("add")
sub = _make_binary_op("sub")
mul = _make_binary_op("mul")
div = _make_binary_op("div")


def matmul(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None
) -> torch.Tensor:
    """Matrix multiplication: c = a @ b"""
    if c is None:
        c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    torch.ops.myops.matmul(a, b, c)
    return c


def matrix_transpose(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Matrix transpose: out = input.T"""
    if out is None:
        out = torch.empty((input.size(1), input.size(0)), dtype=input.dtype, device=input.device)
    torch.ops.myops.matrix_transpose(input, out)
    return out


def conv1d(
    input: torch.Tensor, kernel: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """1D convolution (no padding, stride=1): out[i] = sum(input[i:i+K] * kernel)"""
    if out is None:
        out = torch.empty(input.size(0) - kernel.size(0) + 1, dtype=input.dtype, device=input.device)
    torch.ops.myops.conv1d(input, kernel, out)
    return out


def reduce_sum(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Sum reduction: out = sum(input)"""
    if out is None:
        out = torch.zeros(1, dtype=input.dtype, device=input.device)
    torch.ops.myops.reduce_sum(input, out)
    return out


def _make_unary_op(name: str):
    def op(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        getattr(torch.ops.myops, name)(input, out)
        return out

    op.__name__ = name
    return op


abs = _make_unary_op("abs")
neg = _make_unary_op("neg")
exp = _make_unary_op("exp")
log = _make_unary_op("log")
relu = _make_unary_op("relu")
sigmoid = _make_unary_op("sigmoid")
