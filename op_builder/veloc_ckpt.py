# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import distutils.spawn
import subprocess

from .builder import CUDAOpBuilder


class VelocCkptBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_VELOC_CKPT"
    NAME = "veloc_ckpt"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.veloc.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/veloc/deepspeed_py_veloc.cu', 'csrc/veloc/py_ds_veloc.cpp'
        ]

    def include_paths(self):
        return ['csrc/veloc']
    
    def cxx_args(self):
        args = super().cxx_args()
        # return args + self.version_dependent_macros()
        addn_args = ['-std=c++14', '-fopenmp', '-shared', '-fPIC', '-Wno-reorder']
        return args + self.version_dependent_macros() + addn_args

    def nvcc_args(self):
        nvcc_flags = ['-O3', '-std=c++14'] + self.version_dependent_macros()
        return nvcc_flags

    def extra_ldflags(self):
        return ['-lpthread']
    # def cxx_args(self):
    #     # -O0 for improved debugging, since performance is bound by I/O
    #     CPU_ARCH = self.cpu_arch()
    #     SIMD_WIDTH = self.simd_width()
    #     return [
    #         '-g',
    #         '-Wall',
    #         '-O0',
    #         '-std=c++14',
    #         '-shared',
    #         '-fPIC',
    #         '-Wno-reorder',
    #         CPU_ARCH,
    #         '-fopenmp',
    #         SIMD_WIDTH,
    #     ]
