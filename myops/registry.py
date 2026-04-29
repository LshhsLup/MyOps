import torch
import myops.cuda
import myops.triton


def gen_data_elementwise(n, dtype, device="cuda"):
    return torch.randn(n, dtype=dtype, device=device), torch.randn(n, dtype=dtype, device=device)


def gen_data_unary(n, dtype, device="cuda"):
    return torch.randn(n, dtype=dtype, device=device),


def gen_data_matmul(m, n, k, dtype, device="cuda"):
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    return a, b


OPERATOR_REGISTRY = [
    {
        "name": "VectorAdd",
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
        "shapes": [(10**i,) for i in range(3, 10)],
        "cuda_func": lambda a, b: myops.cuda.add(a, b),
        "triton_func": lambda a, b: myops.triton.add(a, b),
        "torch_func": torch.add,
        "gen_data": gen_data_elementwise,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 3 * dtype_size
    },
]

UNARY_OP_CONFIGS = {
    "Abs":     (torch.abs, [torch.float32, torch.float16, torch.bfloat16], 10),
    "Neg":     (torch.neg, [torch.float32, torch.float16, torch.bfloat16], 10),
    "Relu":    (torch.nn.functional.relu, [torch.float32, torch.float16, torch.bfloat16], 10),
    "Exp":     (torch.exp, [torch.float32], 8),
    "Log":     (torch.log, [torch.float32], 8),
    "Sigmoid": (torch.sigmoid, [torch.float32], 8),
}

for name, (t_func, dtypes, max_exp) in UNARY_OP_CONFIGS.items():
    method_name = name.lower()
    config = {
        "name": f"{method_name}",
        "dtypes": dtypes,
        "shapes": [(10**i,) for i in range(3, max_exp)],
        "cuda_func": lambda a, m=method_name: getattr(myops.cuda, m)(a),
        "triton_func": lambda a, m=method_name: getattr(myops.triton, m)(a),
        "torch_func": t_func,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    }
    OPERATOR_REGISTRY.append(config)