from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depth_utils_cuda',
    ext_modules=[
        CUDAExtension('depth_utils_cuda', [
            'depth_utils_cuda.cpp',
            'depth_utils_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
