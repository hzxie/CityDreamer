# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Jiaxiang Tang (@ashawkey)
# @Date:   2023-04-15 10:33:32
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-29 10:47:10
# @Email:  ashawkey1999@gmail.com
# @Ref: https://github.com/ashawkey/torch-ngp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="grid_encoder",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="grid_encoder_ext",
            sources=[
                "grid_encoder_ext.cu",
                "bindings.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
