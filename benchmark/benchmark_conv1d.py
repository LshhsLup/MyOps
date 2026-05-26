import torch
from tabulate import tabulate
from common import benchmark_kernel
from myops import conv1d

def torch_conv1d_ref(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv1d(
        input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()

params = [
    {"input_size": 1500000, "kernel_size": 2047, "dtype": torch.float32},
]

def main():
    all_results = []

    for p in params:
        input_size, kernel_size, dtype = p["input_size"], p["kernel_size"], p["dtype"]
        input = torch.randn(input_size, dtype=dtype, device="cuda")
        kernel = torch.randn(kernel_size, dtype=dtype, device="cuda")

        torch_time = benchmark_kernel(torch_conv1d_ref, (input, kernel), repeats=100)
        myops_time = benchmark_kernel(conv1d, (input, kernel), repeats=100)

        speedup = torch_time / myops_time if myops_time > 0 else float("inf")

        all_results.append({
            "input_size": input_size,
            "kernel_size": kernel_size,
            "dtype": dtype,
            "torch_us": torch_time,
            "myops_us": myops_time,
            "speedup": speedup,
        })

    table_data = []
    for r in all_results:
        dtype_str = str(r["dtype"]).split(".")[-1]
        table_data.append([
            r["input_size"], r["kernel_size"], dtype_str,
            f"{r['torch_us']:.3f}", f"{r['myops_us']:.3f}",
            f"{r['speedup']:.2f}x",
        ])

    headers = ["Input Size", "Kernel Size", "Dtype", "torch (ms)", "myops (ms)", "Speedup"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))

if __name__ == "__main__":
    main()
