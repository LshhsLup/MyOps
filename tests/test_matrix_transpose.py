import torch
from myops import matrix_transpose
import pytest
from utils import allclose

@pytest.mark.parametrize("rows", [16, 128, 1024, 1031])
@pytest.mark.parametrize("cols", [16, 256, 512, 515])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matrix_transpose_correctness(rows, cols, dtype):
    input = torch.randn(rows, cols, dtype=dtype, device="cuda")
    result = matrix_transpose(input)
    expected = input.T.contiguous()
    tol = 1e-5 if dtype == torch.float32 else 1e-3
    assert allclose(result, expected, atol=tol, rtol=tol)

def test_matrix_transpose_out_parameter():
    input = torch.randn(64, 32, dtype=torch.float32, device="cuda")
    out = torch.empty(32, 64, dtype=torch.float32, device="cuda")
    result = matrix_transpose(input, out=out)
    assert result is out
    assert allclose(result, input.T.contiguous(), atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
