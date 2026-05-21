import torch
from myops import conv1d
import pytest

def torch_conv1d_ref(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv1d(
        input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()

@pytest.mark.parametrize("input_size,kernel_size", [
    (16, 1), (16, 16), (1024, 7), (1024, 128), (10000, 2047),
    (1500000, 2047), (1500000, 1),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_conv1d_correctness(input_size, kernel_size, dtype):
    input = torch.randn(input_size, dtype=dtype, device="cuda")
    kernel = torch.randn(kernel_size, dtype=dtype, device="cuda")
    result = conv1d(input, kernel)
    expected = torch_conv1d_ref(input, kernel)
    rtol = 1e-2 if dtype != torch.float32 else 1e-3
    atol = 1e-3 if dtype != torch.float32 else 1e-4
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)

def test_conv1d_out_parameter():
    input = torch.randn(1024, dtype=torch.float32, device="cuda")
    kernel = torch.randn(7, dtype=torch.float32, device="cuda")
    out = torch.empty(1024 - 7 + 1, dtype=torch.float32, device="cuda")
    result = conv1d(input, kernel, out=out)
    assert result is out
    expected = torch_conv1d_ref(input, kernel)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
