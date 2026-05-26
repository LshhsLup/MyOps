import torch
from tabulate import tabulate
from common import benchmark_kernel
from myops import reduce_sum

params = [
    {"N": 16, "dtype": torch.float32},
    {"N": 32, "dtype": torch.float32},
    {"N": 64, "dtype": torch.float32},
    {"N": 128, "dtype": torch.float32},
    {"N": 256, "dtype": torch.float32},
    {"N": 512, "dtype": torch.float32},
    {"N": 1024, "dtype": torch.float32},
    {"N": 2048, "dtype": torch.float32},
    {"N": 4096, "dtype": torch.float32},
    {"N": 8192, "dtype": torch.float32},
    {"N": 16384, "dtype": torch.float32},
    {"N": 32768, "dtype": torch.float32},
    {"N": 65536, "dtype": torch.float32},
    {"N": 131072, "dtype": torch.float32},
    {"N": 262144, "dtype": torch.float32},
    {"N": 524288, "dtype": torch.float32},
    {"N": 1048576, "dtype": torch.float32},
    {"N": 2097152, "dtype": torch.float32},
    {"N": 4194304, "dtype": torch.float32},
    {"N": 8388608, "dtype": torch.float32},
    {"N": 16777216, "dtype": torch.float32},
    {"N": 33554432, "dtype": torch.float32},
    {"N": 67108864, "dtype": torch.float32},
    {"N": 134217728, "dtype": torch.float32},
    {"N": 268435456, "dtype": torch.float32},
]

def main():
    all_results = []
    for p in params:
        N, dtype = p["N"], p["dtype"]
        input = torch.randn(N, dtype=dtype, device="cuda")
        torch_out = torch.zeros(1, dtype=dtype, device="cuda")
        # Pre-allocate output tensor and pass it in to avoid including torch.zeros
        # allocation time in the measurement. See benchmark/benchmark_zeros.py.
        myops_out = torch.zeros(1, dtype=dtype, device="cuda")

        torch_time = benchmark_kernel(lambda x, o: torch.sum(x, dim=0, keepdim=True, out=o), (input, torch_out), repeats=100)

        bytes_read = N * input.element_size()
        myops_time = benchmark_kernel(reduce_sum, (input, myops_out), repeats=100)
        myops_bw = bytes_read / (myops_time * 1e-6) / 1e9
        speedup = torch_time / myops_time if myops_time > 0 else float("inf")
        all_results.append({
            "N": N, "dtype": dtype,
            "torch_us": torch_time, "myops_us": myops_time,
            "speedup": speedup, "myops_bw_gb_s": myops_bw,
        })

    table_data = []
    for r in all_results:
        dtype_str = str(r["dtype"]).split(".")[-1]
        table_data.append([
            str(r["N"]), dtype_str,
            f"{r['torch_us']:.3f}", f"{r['myops_us']:.3f}",
            f"{r['speedup']:.2f}x", f"{r['myops_bw_gb_s']:.2f}",
        ])
    headers = ["N", "Dtype", "torch (us)", "myops (us)", "Speedup", "BW (GB/s)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))


if __name__ == "__main__":
    main()
