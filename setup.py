from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lif_cuda',
    ext_modules=[
        CUDAExtension(
            name='lif_cuda'
            sources=[
                'cuda/lif_binding.cpp'
                'cuda/lif_forward.cpp'
                'cuda/lif_backward.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3']
                'nvcc': ['-O3', '--use_fast_math']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
