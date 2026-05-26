import torch
from tabulate import tabulate
from common import benchmark_kernel
from myops import matrix_transpose

params = [
    {"rows": 128, "cols": 128, "dtype": torch.float32},
    {"rows": 1024, "cols": 1024, "dtype": torch.float32},
    {"rows": 4096, "cols": 4096, "dtype": torch.float32},
    {"rows": 1024, "cols": 1024, "dtype": torch.float16},
    {"rows": 1024, "cols": 1024, "dtype": torch.bfloat16},
]

def main():
    all_results = []

    for p in params:
        rows, cols, dtype = p["rows"], p["cols"], p["dtype"]
        input = torch.randn(rows, cols, dtype=dtype, device="cuda")

        def torch_transpose(x):
            return x.T.contiguous()

        torch_time = benchmark_kernel(torch_transpose, (input,), repeats=100)
        myops_time = benchmark_kernel(matrix_transpose, (input,), repeats=100)

        speedup = torch_time / myops_time if myops_time > 0 else float("inf")

        all_results.append({
            "rows": rows,
            "cols": cols,
            "dtype": dtype,
            "torch_us": torch_time,
            "myops_us": myops_time,
            "speedup": speedup,
        })

    table_data = []
    for r in all_results:
        dtype_str = str(r["dtype"]).split(".")[-1]
        table_data.append([
            r["rows"], r["cols"], dtype_str,
            f"{r['torch_us']:.3f}", f"{r['myops_us']:.3f}",
            f"{r['speedup']:.2f}x",
        ])

    headers = ["Rows", "Cols", "Dtype", "torch (ms)", "myops (ms)", "Speedup"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))

if __name__ == "__main__":
    main()
