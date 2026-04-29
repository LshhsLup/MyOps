import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["MAX_JOBS"] = str(os.cpu_count())

def get_sources():
    sources = ["csrc/binding.cpp"]
    for root, _, files in os.walk("csrc/kernels"):
        for file in files:
            if file.endswith(".cu") or file.endswith(".cpp"):
                sources.append(os.path.join(root, file))
    return sources

setup(
    name="myops",
    ext_modules=[
        CUDAExtension(
            name="myops._core",
            sources=get_sources(),
            include_dirs=[os.path.abspath("csrc/include")],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_80'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)