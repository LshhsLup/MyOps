import torch
import argparse
import time
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


def run_op_test(args):
    print(f"\n{'='*110}")
    print(f"{'KERNEL PERFORMANCE BENCHMARK':^110}")
    print(f"{'='*110}")
    print(f"{'Shape':<20} | {'Dtype':<10} | {'Triton(ms)':<12} | {'CUDA(ms)':<12} | {'Torch(ms)':<12} | {'Triton Spd':<10} | {'CUDA Spd':<10}")
    print("-" * 110)

    for op in OPERATOR_REGISTRY:
        if args.op != "all" and op["name"] != args.op:
            continue
        print(f"\n{op['name']}")

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
                t_cuda = benchmark_kernel(op["cuda_func"], inputs, args.repeats)
                t_torch = benchmark_kernel(op["torch_func"], inputs, args.repeats)

                triton_speedup = t_torch / t_triton
                cuda_speedup = t_torch / t_cuda

                print(f"{str(shape):<20} | {str(dtype).split('.')[-1]:<10} | "
                      f"{t_triton:<12.4f} | {t_cuda:<12.4f} | {t_torch:<12.4f} | "
                      f"{triton_speedup:<10.2f}x | {cuda_speedup:<10.2f}x")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="all")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bench_only", action="store_true")
    args = parser.parse_args()
    run_op_test(args)