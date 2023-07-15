# -*- coding: utf-8 -*-
#
# @File:   helper.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 10:25:10
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-15 13:28:13
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


def requires_grad(model, require=True):
    for p in model.parameters():
        p.requires_grad = require


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def get_seg_map_palette():
    palatte = np.array([[i, i, i] for i in range(256)])
    # fmt: off
    palatte[:7] = np.array(
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
    return palatte


@static_vars(palatte=get_seg_map_palette())
def get_seg_map(seg_map):
    if np.max(seg_map) >= 7:
        return get_ins_seg_map(seg_map)

    seg_map = Image.fromarray(seg_map.astype(np.uint8))
    seg_map.putpalette(get_seg_map.palatte.reshape(-1).tolist())
    return seg_map


def get_ins_seg_map_palette(legacy_palette):
    MAX_N_INSTANCES = 32768
    # Make sure that the roof colors are similar to the corresponding facade colors.
    # The odd and even indexes are reserved for roof and facade, respectively.
    palatte0 = np.random.randint(256, size=(MAX_N_INSTANCES, 3))
    # palatte1 = (255 - palatte0) // 50  + palatte0
    palatte1 = palatte0 - 2
    palatte1[palatte1 < 0] = 0

    palatte = np.concatenate((palatte0, palatte1), axis=1)
    palatte = palatte.reshape(-1, 3)
    palatte[:7] = legacy_palette[:7]
    return palatte


@static_vars(palatte=get_ins_seg_map_palette(get_seg_map_palette()))
def get_ins_seg_map(seg_map):
    h, w = seg_map.shape
    seg_map_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(np.max(seg_map) + 1):
        seg_map_rgb[seg_map == i] = get_ins_seg_map.palatte[i]

    return Image.fromarray(seg_map_rgb)


def get_diffuse_shading_img(seg_map, depth2, raydirs, cam_origin):
    mc_rgb = np.array(seg_map.convert("RGB"))
    # Diffused shading, co-located light.
    first_intersection_depth = depth2[0, :, :, 0, None, :]
    first_intersection_point = (
        raydirs * first_intersection_depth + cam_origin[None, None, None, :]
    )
    fip_local_coords = torch.remainder(first_intersection_point, 1.0)
    fip_wall_proximity = torch.minimum(fip_local_coords, 1.0 - fip_local_coords)
    fip_wall_orientation = torch.argmin(fip_wall_proximity, dim=-1, keepdim=False)
    # 0: [1,0,0]; 1: [0,1,0]; 2: [0,0,1]
    lut = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=torch.float32,
        device=fip_wall_orientation.device,
    )
    fip_normal = lut[fip_wall_orientation]
    diffuse_shade = torch.abs(torch.sum(fip_normal * raydirs, dim=-1))

    mc_rgb = mc_rgb.astype(float) / 255
    mc_rgb = mc_rgb * diffuse_shade.cpu().numpy()
    mc_rgb = (mc_rgb ** (1 / 2.2)) * 255
    return Image.fromarray(mc_rgb.astype(np.uint8))


def masks_to_onehots(masks, n_class, ignored_classes=[]):
    b, h, w = masks.shape
    n_class_actual = n_class - len(ignored_classes)
    one_hot_masks = torch.zeros(
        (b, n_class_actual, h, w), dtype=torch.float32, device=masks.device
    )

    n_class_cnt = 0
    for i in range(n_class):
        if i not in ignored_classes:
            one_hot_masks[:, n_class_cnt] = masks == i
            n_class_cnt += 1

    return one_hot_masks


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
    # assert mode in ["HeightField", "FootprintCtr", "SegMap", "RGB"]
    tensor = tensor.cpu().numpy()
    if mode == "HeightField":
        return tensor.transpose((1, 2, 0)).squeeze() / np.max(tensor)
    elif mode == "FootprintCtr":
        return tensor.transpose((1, 2, 0)).squeeze()
    elif mode == "SegMap":
        return get_seg_map(tensor.squeeze()).convert("RGB")
    elif mode == "RGB":
        return tensor.squeeze().transpose((1, 2, 0)) / 2 + 0.5
    else:
        raise Exception("Unknown mode: %s" % mode)
