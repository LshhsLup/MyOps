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
    # Binary ops
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
    # Unary ops
    {
        "name": "UnaryAbs",
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
        "shapes": [(10**i,) for i in range(3, 10)],
        "cuda_func": lambda a: myops.cuda.abs(a),
        "triton_func": lambda a: myops.triton.abs(a),
        "torch_func": torch.abs,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    },
    {
        "name": "UnaryNeg",
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
        "shapes": [(10**i,) for i in range(3, 10)],
        "cuda_func": lambda a: myops.cuda.neg(a),
        "triton_func": lambda a: myops.triton.neg(a),
        "torch_func": torch.neg,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    },
    {
        "name": "UnaryExp",
        "dtypes": [torch.float32],
        "shapes": [(10**i,) for i in range(3, 8)],
        "cuda_func": lambda a: myops.cuda.exp(a),
        "triton_func": lambda a: myops.triton.exp(a),
        "torch_func": torch.exp,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    },
    {
        "name": "UnaryLog",
        "dtypes": [torch.float32],
        "shapes": [(10**i,) for i in range(3, 8)],
        "cuda_func": lambda a: myops.cuda.log(a),
        "triton_func": lambda a: myops.triton.log(a),
        "torch_func": torch.log,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    },
    {
        "name": "UnaryRelu",
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
        "shapes": [(10**i,) for i in range(3, 10)],
        "cuda_func": lambda a: myops.cuda.relu(a),
        "triton_func": lambda a: myops.triton.relu(a),
        "torch_func": torch.nn.functional.relu,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    },
    {
        "name": "UnarySigmoid",
        "dtypes": [torch.float32],
        "shapes": [(10**i,) for i in range(3, 8)],
        "cuda_func": lambda a: myops.cuda.sigmoid(a),
        "triton_func": lambda a: myops.triton.sigmoid(a),
        "torch_func": torch.sigmoid,
        "gen_data": gen_data_unary,
        "get_flops": lambda shape: shape[0],
        "get_bytes": lambda shape, dtype_size: shape[0] * 2 * dtype_size
    },
]