"""Python API for myops operators."""

import torch

__all__ = [
    "add",
    "matmul",
    "abs",
    "neg",
    "exp",
    "log",
    "relu",
    "sigmoid",
]


def add(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Element-wise addition: out = a + b"""
    if out is None:
        out = torch.empty_like(a)
    torch.ops.myops.add(a, b, out)
    return out


def matmul(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None
) -> torch.Tensor:
    """Matrix multiplication: c = a @ b"""
    if c is None:
        c = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    torch.ops.myops.matmul(a, b, c)
    return c


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
