import torch
import pytest
import random
from myops import matmul
from utils import allclose

m, n, k = [], [], []

torch.manual_seed(123)
random.seed(123)

shapes = []
for _ in range(20):
    shapes.append((
        random.randint(1, 8193),
        random.randint(1, 8193),
        random.randint(1, 8193),
    ))

shapes.extend([
    (1, 1, 1),    
    (1, 1024, 1024),   
    (1024, 1, 1024),     
    (1024, 1024, 1), 
    (8192, 8192, 8192),  
    (1024, 1024, 1024),  
    (2048, 1024, 512),   
])

@pytest.mark.parametrize("M,N,K", shapes)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_matmul_correctness(M, N, K, dtype):
    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.randn((K, N), dtype=dtype, device="cuda")
    c = matmul(a, b)
    expected = torch.matmul(a, b)
    assert allclose(c, expected, atol=0.2, rtol=3e-3)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
