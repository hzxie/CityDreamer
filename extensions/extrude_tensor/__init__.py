# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Haozhe Xie
# @Date:   2023-03-24 20:24:38
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-06-16 09:55:58
# @Email:  root@haozhexie.com

import torch

import extrude_tensor_ext


class TensorExtruder(torch.nn.Module):
    def __init__(self, max_height=256):
        super(TensorExtruder, self).__init__()
        self.max_height = max_height

    def forward(self, seg_map, height_field):
        assert torch.max(height_field) < self.max_height, "Max Value %d" % torch.max(height_field)
        return ExtrudeTensorFunction.apply(seg_map, height_field, self.max_height)


class ExtrudeTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, seg_map, height_field, max_height):
        # seg_map.shape: (B, C, H, W)
        # height_field.shape: (B, C, H, W)
        return extrude_tensor_ext.forward(seg_map, height_field, max_height)

    @staticmethod
    def backward(ctx, grad_volume):
        # grad_volume.shape: (B, C, H, W, D)
        # Combine the gradients along the Z-axis.
        grad_seg_map = torch.sum(grad_volume, dim=4)
        grad_height_field = grad_seg_map
        return grad_seg_map, grad_height_field
