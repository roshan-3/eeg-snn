"""Build hook for the custom CUDA LIF kernel.

Most users do not need to invoke this directly. The package autoloads the
extension via ``torch.utils.cpp_extension.load`` on import of ``lif_kernel``
(see ``lif_kernel/__init__.py``). This file exists so the extension can also be
built ahead of time with ``python setup.py build_ext --inplace``.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lif_cuda",
    ext_modules=[
        CUDAExtension(
            name="lif_cuda",
            sources=[
                "lif_kernel/lif_binding.cpp",
                "lif_kernel/lif_forward.cu",
                "lif_kernel/lif_backward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
