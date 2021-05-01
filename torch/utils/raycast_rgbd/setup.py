from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='raycast_rgbd_cuda',
    ext_modules=[
        CUDAExtension('raycast_rgbd_cuda', [
            'raycast_rgbd_cuda.cpp',
            'raycast_rgbd_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
