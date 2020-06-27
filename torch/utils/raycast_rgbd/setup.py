from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']
nvcc_args = [
    '-gencode', 'arch=compute_75,code=sm_75'
]

setup(
    name='raycast_rgbd_cuda',
    ext_modules=[
        CUDAExtension('raycast_rgbd_cuda', [
            'raycast_rgbd_cuda.cpp',
            'raycast_rgbd_cuda_kernel.cu',
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
