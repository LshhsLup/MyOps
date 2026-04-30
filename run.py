import argparse
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from myops.registry import OPERATOR_REGISTRY

def benchmark_kernel(func, inputs, repeats=100):
    for _ in range(10):
        func(*inputs)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeats):
        func(*inputs)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / repeats

def format_time(t: float) -> str:
    return f"{t:.6f} ms"

def format_speedup(torch_t: float, kernel_t: float) -> Text:
    ratio = torch_t / kernel_t
    if ratio >= 1.0:
        return Text(f"↑{ratio:.2f}x", style="green")
    else:
        return Text(f"↓{ratio:.2f}x", style="red")

def run_op_test(args):
    console = Console()
    for op in OPERATOR_REGISTRY:
        if args.op != "all" and op["name"].lower() != args.op.lower():
            continue
        op_name = op["name"].upper()
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,            
            show_edge=False,
            pad_edge=False,
            collapse_padding=False, 
            expand=False,
        )
        table.add_column("Shape",           style="white", justify="center",  width=14)
        table.add_column("Dtype",           style="white", justify="center",  width=10)
        table.add_column("Triton",          style="white", justify="center", width=18)
        table.add_column("CUDA",            style="white", justify="center", width=18)
        table.add_column("Torch (ref)",     style="white", justify="center", width=18)
        table.add_column("Triton vs Torch", style="white", justify="center", width=18)
        table.add_column("CUDA vs Torch",   style="white", justify="center", width=18)
        for shape in op["shapes"]:
            for dtype in op["dtypes"]:
                inputs = op["gen_data"](*shape, dtype)
                if not args.bench_only:
                    triton_res = op["triton_func"](*inputs)
                    cuda_res = op["cuda_func"](*inputs)
                    ref_res = op["torch_func"](*inputs)
                    torch.testing.assert_close(triton_res, ref_res, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(cuda_res, ref_res, atol=1e-5, rtol=1e-5)
                t_triton = benchmark_kernel(op["triton_func"], inputs, args.repeats)
                t_cuda   = benchmark_kernel(op["cuda_func"], inputs, args.repeats)
                t_torch  = benchmark_kernel(op["torch_func"], inputs, args.repeats)
                tri_time_text = Text(format_time(t_triton), style="red" if t_triton > t_torch else "green")
                cuda_time_text = Text(format_time(t_cuda), style="red" if t_cuda > t_torch else "green")
                torch_time_text = format_time(t_torch)
                tri_speedup_text = format_speedup(t_torch, t_triton)
                cuda_speedup_text = format_speedup(t_torch, t_cuda)
                dtype_str = str(dtype).split(".")[-1]
                table.add_row(
                    str(shape),
                    dtype_str,
                    tri_time_text,
                    cuda_time_text,
                    torch_time_text,
                    tri_speedup_text,
                    cuda_speedup_text,
                )
        panel = Panel(
            table,
            title=f"[bold cyan]{op_name}[/bold cyan]",
            border_style="bright_blue",
            expand=False,
        )
        console.print(panel)

    if args.op == "all":
        console.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="all")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bench_only", action="store_true")
    args = parser.parse_args()
    run_op_test(args)