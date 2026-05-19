import torch

import myops.cuda
import myops.triton


def gen_data_elementwise(n, dtype, device="cuda"):
    return torch.randn(n, dtype=dtype, device=device), torch.randn(
        n, dtype=dtype, device=device
    )


def gen_data_unary(n, dtype, device="cuda"):
    return (torch.randn(n, dtype=dtype, device=device),)


OPERATOR_REGISTRY = []

# Binary ops config: (torch_func, dtypes, max_shape_exp)
BINARY_OPS = {
    "add": (torch.add, [torch.float32, torch.float16, torch.bfloat16], 10),
    "sub": (torch.sub, [torch.float32, torch.float16, torch.bfloat16], 10),
    "mul": (torch.mul, [torch.float32, torch.float16, torch.bfloat16], 10),
    "div": (torch.div, [torch.float32, torch.float16, torch.bfloat16], 10),
}

for name, (torch_fn, dtypes, max_exp) in BINARY_OPS.items():
    OPERATOR_REGISTRY.append(
        {
            "name": name,
            "dtypes": dtypes,
            "shapes": [(10**i,) for i in range(3, max_exp)],
            "cuda_func": lambda a, b, n=name: getattr(myops.cuda, n)(a, b),
            "triton_func": lambda a, b, n=name: getattr(myops.triton, n)(a, b),
            "torch_func": torch_fn,
            "gen_data": gen_data_elementwise,
        }
    )

# Unary ops config: (torch_func, dtypes, max_shape_exp)
UNARY_OPS = {
    "abs": (torch.abs, [torch.float32, torch.float16, torch.bfloat16], 10),
    "neg": (torch.neg, [torch.float32, torch.float16, torch.bfloat16], 10),
    "relu": (
        torch.nn.functional.relu,
        [torch.float32, torch.float16, torch.bfloat16],
        10,
    ),
    "exp": (torch.exp, [torch.float32], 8),
    "log": (torch.log, [torch.float32], 8),
    "sigmoid": (torch.sigmoid, [torch.float32], 8),
}

for name, (torch_fn, dtypes, max_exp) in UNARY_OPS.items():
    OPERATOR_REGISTRY.append(
        {
            "name": name,
            "dtypes": dtypes,
            "shapes": [(10**i,) for i in range(3, max_exp)],
            "cuda_func": lambda a, n=name: getattr(myops.cuda, n)(a),
            "triton_func": lambda a, n=name: getattr(myops.triton, n)(a),
            "torch_func": torch_fn,
            "gen_data": gen_data_unary,
        }
    )
