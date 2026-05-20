import torch

def benchmark_kernel(func, inputs, repeats=1):
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