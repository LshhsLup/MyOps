import torch
from myops import matrix_transpose
import pytest

@pytest.mark.parametrize("rows,cols", [
    (16, 16), (128, 256), (1024, 512), (1024 + 7, 512 + 3),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matrix_transpose_correctness(rows, cols, dtype):
    input = torch.randn(rows, cols, dtype=dtype, device="cuda")
    result = matrix_transpose(input)
    expected = input.T.contiguous()
    rtol = 1e-2 if dtype != torch.float32 else 1e-3
    atol = 1e-3 if dtype != torch.float32 else 1e-4
    assert torch.allclose(result, expected, rtol=rtol, atol=atol)

def test_matrix_transpose_out_parameter():
    input = torch.randn(64, 32, dtype=torch.float32, device="cuda")
    out = torch.empty(32, 64, dtype=torch.float32, device="cuda")
    result = matrix_transpose(input, out=out)
    assert result is out
    assert torch.allclose(result, input.T.contiguous(), rtol=1e-3, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
