import torch
from tabulate import tabulate
from common import benchmark_kernel
from myops import reduce_sum

params = [
    {"N": 4194304, "dtype": torch.float32},
]


def main():
    all_results = []
    for p in params:
        N, dtype = p["N"], p["dtype"]
        input = torch.randn(N, dtype=dtype, device="cuda")

        torch_time = benchmark_kernel(lambda x: torch.sum(x), (input,), repeats=100)
        myops_time = benchmark_kernel(reduce_sum, (input,), repeats=100)

        speedup = torch_time / myops_time if myops_time > 0 else float("inf")
        all_results.append({
            "N": N, "dtype": dtype,
            "torch_ms": torch_time, "myops_ms": myops_time,
            "speedup": speedup,
        })

    table_data = []
    for r in all_results:
        dtype_str = str(r["dtype"]).split(".")[-1]
        table_data.append([
            str(r["N"]), dtype_str,
            f"{r['torch_ms']:.3f}", f"{r['myops_ms']:.3f}",
            f"{r['speedup']:.2f}x",
        ])
    headers = ["N", "Dtype", "torch (ms)", "myops (ms)", "Speedup"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))


if __name__ == "__main__":
    main()
