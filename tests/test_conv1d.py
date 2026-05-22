import torch
import pytest
import random
from myops import conv1d
from utils import allclose

def torch_conv1d_ref(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv1d(
        input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).flatten()

torch.manual_seed(123)
random.seed(123)

@pytest.mark.parametrize("input_size", [16, 1024, 10000, 1500000])
@pytest.mark.parametrize("kernel_size", [1, 7, 16, 128, 2047])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv1d_correctness(input_size, kernel_size, dtype):
    if input_size < kernel_size:
        pytest.skip("input_size must be >= kernel_size")
    input = torch.randn(input_size, dtype=dtype, device="cuda")
    kernel = torch.randn(kernel_size, dtype=dtype, device="cuda")
    result = conv1d(input, kernel)
    expected = torch_conv1d_ref(input, kernel)
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    assert allclose(result, expected, atol=tol, rtol=tol)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
