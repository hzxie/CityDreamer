# -*- coding: utf-8 -*-
#
# @File:   distributed.py
# @Author: NVIDIA
# @Date:   2023-04-29 11:50:12
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-29 12:18:02
# @Email:  root@haozhexie.com
# @Ref: https://github.com/NVlabs/imaginaire

import ctypes
import math
import os
import pynvml
import torch
import torch.distributed


pynvml.nvmlInit()


class Device(object):
    r"""Device used for nvml."""
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def getName(self):
        r"""Get obect name"""
        return pynvml.nvmlDeviceGetName(self.handle)

    def getCpuAffinity(self):
        r"""Get CPU affinity"""
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(
            self.handle, Device._nvml_affinity_elements
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        return [i for i, e in enumerate(affinity_list) if e != 0]


def set_affinity(gpu_id=None):
    r"""Set GPU affinity
    Args:
        gpu_id (int): Which gpu device.
    """
    if gpu_id is None:
        gpu_id = int(os.getenv("LOCAL_RANK", 0))

    dev = Device(gpu_id)
    os.sched_setaffinity(0, dev.getCpuAffinity())

    # list of ints
    # representing the logical cores this process is now affinitied with
    return os.sched_getaffinity(0)


def init_dist(local_rank, backend="nccl", **kwargs):
    r"""Initialize distributed training"""
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            return torch.cuda.current_device()
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", **kwargs
        )

    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    # assert pValue.contents.value == 128


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
    return world_size


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


def is_local_master():
    return torch.cuda.current_device() == 0
