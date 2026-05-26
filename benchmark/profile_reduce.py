"""Profiling script for Nsight Compute.

Usage:
    ncu --target-processes all --kernel-name "reduceSum" python profile_reduce.py

Usage (save report for GUI):
    ncu --target-processes all --kernel-name "reduceSum" -o report python profile_reduce.py
"""

import argparse
import torch
from myops import reduce_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda"
    input_t = torch.ones(args.n, dtype=torch.float32, device=device)
    out = torch.zeros(1, dtype=torch.float32, device=device)

    # Warmup: first call may trigger JIT / context init, we don't want ncu to capture that
    reduce_sum(input_t, out=out)
    torch.cuda.synchronize()

    # The actual call ncu will capture (next invocation after warmup)
    reduce_sum(input_t, out=out)
    torch.cuda.synchronize()

    print(f"Done. N={args.n}, result={out.item()}")


if __name__ == "__main__":
    main()
