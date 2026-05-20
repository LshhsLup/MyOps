import torch
from tabulate import tabulate
from common import benchmark_kernel
from myops import matmul

params = [
    {"M": 128, "N": 128, "K": 128, "dtype": torch.float},
    {"M": 4096, "N": 1024, "K": 4096, "dtype": torch.float},
]

def main():
    all_results = []

    for p in params:
        M, N, K, dtype = p["M"], p["N"], p["K"], p["dtype"]
        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")
        c = torch.empty(M, N, dtype=dtype, device="cuda")

        # torch.matmul baseline
        torch_time = benchmark_kernel(torch.matmul, (a, b), repeats=100)

        # myops.matmul
        myops_time = benchmark_kernel(matmul, (a, b, c), repeats=100)

        speedup = torch_time / myops_time if myops_time > 0 else float("inf")

        all_results.append({
            "M": M,
            "N": N,
            "K": K,
            "torch_ms": torch_time,
            "myops_ms": myops_time,
            "speedup": speedup,
        })

    # Build table
    table_data = []
    for r in all_results:
        table_data.append([
            r["M"],
            r["N"],
            r["K"],
            f"{r['torch_ms']:.3f}",
            f"{r['myops_ms']:.3f}",
            f"{r['speedup']:.2f}x",
        ])

    headers = ["M", "N", "K", "torch.matmul (ms)", "myops.matmul (ms)", "Speedup"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))


if __name__ == "__main__":
    main()