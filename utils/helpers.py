# -*- coding: utf-8 -*-
#
# @File:   helper.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:25:10
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-07 20:03:32
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


def mask_to_onehot(mask, n_class, ignored_classes=[]):
    h, w = mask.shape
    n_class_actual = n_class - len(ignored_classes)
    one_hot_masks = np.zeros((h, w, n_class_actual), dtype=np.uint8)

    n_class_cnt = 0
    for i in range(n_class):
        if i not in ignored_classes:
            one_hot_masks[..., n_class_cnt] = mask == i
            n_class_cnt += 1

    return one_hot_masks


def onehot_to_mask(onehot, ignored_classes=[]):
    mask = torch.argmax(onehot, dim=1)
    for ic in ignored_classes:
        mask[mask >= ic] += 1

    return mask


def tensor_to_image(tensor, mode):
    # assert mode in ["HeightField", "SegMap"]
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    if mode == "HeightField":
        return tensor.squeeze() / np.max(tensor)
    elif mode == "SegMap":
        return get_seg_map(tensor.squeeze()).convert("RGB")
    else:
        raise Exception("Unknown mode: %s" % mode)
