import torch
import myops

def gen_data_elementwise(n, dtype, device="cuda"):
    return torch.randn(n, dtype=dtype, device=device), torch.randn(n, dtype=dtype, device=device)

def gen_data_matmul(m, n, k, dtype, device="cuda"):
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    return a, b

OPERATOR_REGISTRY = [
    {
        "name": "VectorAdd",
        "dtypes": [torch.float32, torch.float16, torch.bfloat16],
        "shapes": [(10**i,) for i in range(3, 10)],
        "my_func": lambda a, b: myops.add(a, b),
        "torch_func": torch.add,
        "data_gen": gen_data_elementwise,
        "get_flops": lambda shape: shape[0], 
        "get_bytes": lambda shape, dtype_size: shape[0] * 3 * dtype_size # 2 read, 1 write
    },
    {
        "name": "MatMul",
        "dtypes": [torch.float32],
        "shapes": [(8192, 6144, 4096), (4096, 2048, 1024), (2048, 1024, 512)],
        "my_func": lambda a, b: myops.matmul(a, b),
        "torch_func": torch.matmul,
        "data_gen": gen_data_matmul,
        # MatMul 的标准计算量：2 * M * N * K
        "get_flops": lambda shape: 2 * shape[0] * shape[1] * shape[2], 
        # MatMul 的标准访存量：(M*K + K*N + M*N) * element_size
        "get_bytes": lambda shape, dtype_size: (shape[0]*shape[2] + shape[2]*shape[1] + shape[0]*shape[1]) * dtype_size
    }
]