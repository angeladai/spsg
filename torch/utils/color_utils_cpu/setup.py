from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='color_utils_cpp',
    ext_modules=[
        CppExtension('color_utils_cpp', ['color_utils_cpp.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
