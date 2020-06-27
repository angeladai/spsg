from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']
nvcc_args = [
    '-gencode', 'arch=compute_75,code=sm_75'
]

setup(
    name='depth_utils_cuda',
    ext_modules=[
        CUDAExtension('depth_utils_cuda', [
            'depth_utils_cuda.cpp',
            'depth_utils_cuda_kernel.cu',
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
