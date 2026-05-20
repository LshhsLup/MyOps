import torch
from myops import matmul
import pytest

@pytest.mark.parametrize("M,N,K", [
    (16, 16, 16), (1024+15, 256, 96 + 7)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_matmul_parametrized(M, N, K, dtype):
    """Parametrized test for different matrix sizes."""
    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.randn((K, N), dtype=dtype, device="cuda")
    c = matmul(a, b)

    expected = torch.matmul(a, b)
    assert torch.allclose(c, expected, rtol=1e-3, atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
