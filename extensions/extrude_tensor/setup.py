# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Haozhe Xie
# @Date:   2023-03-24 20:35:43
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-29 10:47:30
# @Email:  root@haozhexie.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="extrude_tensor",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            "extrude_tensor_ext",
            [
                "bindings.cpp",
                "extrude_tensor_ext.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
