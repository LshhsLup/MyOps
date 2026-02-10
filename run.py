import torch
import argparse
import time
from myops.registry import OPERATOR_REGISTRY

def benchmark_kernel(func, inputs, repeats=100):
    # 1. Warmup
    for _ in range(10):
        func(*inputs)
    torch.cuda.synchronize()
    # 2. Actual Benchmark
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(repeats):
        func(*inputs)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / repeats # Average time in ms

def run_op_test(args):
    print(f"\n{'='*95}\n{'CUDA OPERATOR PERFORMANCE DASHBOARD':^95}\n{'='*95}")
    for op in OPERATOR_REGISTRY:
        # Filter by operator name
        if args.op != "all" and op["name"] != args.op:
            continue
        print(f"🚀 Operator: {op['name']}")
        print(f"{'Shape':<25} | {'Dtype':<12} | {'Mine(ms)':<10} | {'Torch(ms)':<10} | {'TFLOPS':<10} | {'GB/s':<10}")
        print("-" * 95)
        for shape in op["shapes"]:
            for dtype in op["dtypes"]:
                inputs = op["data_gen"](*shape, dtype)
                dtype_size = torch.tensor([], dtype=dtype).element_size()
                # Accuracy Check
                if not args.bench_only:
                    my_res = op["my_func"](*inputs)
                    ref_res = op["torch_func"](*inputs)
                    torch.testing.assert_close(my_res, ref_res, atol=1e-2, rtol=1e-2)
                # Benchmarking
                t_my = benchmark_kernel(op["my_func"], inputs, args.repeats)
                t_ref = benchmark_kernel(op["torch_func"], inputs, args.repeats)
                # Calculate metrics
                flops = op["get_flops"](shape)
                bytes_accessed = op["get_bytes"](shape, dtype_size)
                tflops = (flops / (t_my / 1000)) / 1e12 # TFLOPS
                bandwidth = (bytes_accessed / (t_my / 1000)) / 1e9 / 1555# GB/s
                print(f"{str(shape):<20} | {str(dtype).split('.')[-1]:<10} | {t_my:<10.4f} | {t_ref:<10.4f} | {tflops:<10.2f} | {bandwidth:<10.2f}")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="all")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bench_only", action="store_true")
    args = parser.parse_args()
    run_op_test(args)