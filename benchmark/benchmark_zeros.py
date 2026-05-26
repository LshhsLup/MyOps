import torch
from tabulate import tabulate


sizes = [1, 16, 256, 4096, 65536, 1048576, 16777216]
dtypes = [torch.float32, torch.float16, torch.bfloat16]


def benchmark_zeros_cuda(size: int, dtype: torch.dtype, repeats: int = 1000):
    # Warmup
    for _ in range(10):
        _ = torch.zeros(size, dtype=dtype, device="cuda")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        _ = torch.zeros(size, dtype=dtype, device="cuda")
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    elapsed_us = elapsed_ms * 1000 / repeats

    bytes_alloc = size * torch.tensor([], dtype=dtype).element_size()
    bw = bytes_alloc / (elapsed_us * 1e-6) / 1e9 if elapsed_us > 0 else 0
    return elapsed_us, bw


def main():
    results = []
    for dtype in dtypes:
        for size in sizes:
            us, bw = benchmark_zeros_cuda(size, dtype)
            dtype_str = str(dtype).split(".")[-1]
            results.append([dtype_str, str(size), f"{us:.3f}", f"{bw:.2f}"])

    headers = ["Dtype", "Size", "Time (us)", "BW (GB/s)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
