import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["MAX_JOBS"] = str(os.cpu_count())

def get_sources():
    sources = []
    # torch_api adapter layer
    for file in os.listdir("csrc/torch_api"):
        if file.endswith(".cpp"):
            sources.append(os.path.join("csrc/torch_api", file))
    # CUDA kernels
    for root, _, files in os.walk("csrc/kernels"):
        for file in files:
            if file.endswith(".cu") or file.endswith(".cpp"):
                sources.append(os.path.join(root, file))
    return sources

SUPPORTED_ARCHS = [
    (70, 70),   # Volta: V100, Titan V
    (75, 75),   # Turing: RTX 20, T4, Quadro RTX
    (80, 80),   # Ampere: A100, RTX 3090/3080
    (86, 86),   # Ampere: RTX 3060/3070, A40
    (89, 89),   # Ada Lovelace: RTX 4090/4080
    (90, 90),   # Hopper: H100, H200
]

def get_cuda_arch_flags():
    """生成多架构编译标志"""
    flags = []
    for compute, sm in SUPPORTED_ARCHS:
        flags.append(f'-gencode=arch=compute_{compute},code=sm_{sm}')
    latest_compute = max(arch[0] for arch in SUPPORTED_ARCHS)
    flags.append(f'-gencode=arch=compute_{latest_compute},code=compute_{latest_compute}')
    
    return flags

setup(
    name="myops",
    packages=["myops"],
    ext_modules=[
        CUDAExtension(
            name="myops._core",
            sources=get_sources(),
            include_dirs=[
                os.path.abspath("csrc"),
                os.path.abspath("csrc/torch_api"),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    *get_cuda_arch_flags(),
                    '--extended-lambda'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)