import torch
import pytest
import random
from myops import reduce_sum
from utils import allclose

torch.manual_seed(111)
random.seed(111)
n = [random.randint(1, 1000000) for _ in range(20)]
n.append(1)
n.append(3)
n.append(16)
n.append(233)

@pytest.mark.parametrize("N", n)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_reduce_sum_correctness(N, dtype):
    input = torch.randn(N, dtype=dtype, device="cuda")
    result = reduce_sum(input)
    expected = torch.sum(input)
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    assert allclose(result, expected.reshape_as(result), atol=tol, rtol=tol)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
