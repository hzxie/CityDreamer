# -*- coding: utf-8 -*-
#
# @File:   helper.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:25:10
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 19:42:20
# @Email:  root@haozhexie.com

import numpy as np
import torch

from PIL import Image

count_parameters = lambda n: sum(p.numel() for p in n.parameters())


def var_or_cuda(x, device=None):
    x = x.contiguous()
    if torch.cuda.is_available() and device != torch.device("cpu"):
        if device is None:
            x = x.cuda(non_blocking=True)
        else:
            x = x.cuda(device=device, non_blocking=True)
    return x


def get_seg_map(seg_map):
    PALETTE = np.array([[i, i, i] for i in range(256)])
    # fmt: off
    PALETTE[:7] = np.array(
        [
            [0, 0, 0],       # empty        -> black (ONLY used in voxel)
            [96, 0, 0],      # highway      -> red
            [96, 96, 0],     # building     -> yellow
            [0, 96, 0],      # green lands  -> green
            [0, 96, 96],     # construction -> cyan
            [0, 0, 96],      # water        -> blue
            [128, 128, 128], # ground       -> gray
        ]
    )
    # fmt: on
    seg_map = Image.fromarray(seg_map.astype(np.uint8))
    seg_map.putpalette(PALETTE.reshape(-1).tolist())
    return seg_map


def get_keyframes(tensor):
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    n_ch = tensor.shape[2]
    if n_ch == 3 or n_ch == 1:
        return {None: tensor}
    elif n_ch == 2:
        return {
            "HeightField": tensor[..., 0] / np.max(tensor[..., 0]),
            "SegMap": get_seg_map(tensor[..., 1]).convert("RGB"),
        }
    else:
        raise Exception("Unknown image format with channels=%d" % n_ch)
