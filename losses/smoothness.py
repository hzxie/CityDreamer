# -*- coding: utf-8 -*-
#
# @File:   smoothness.py
# @Author: Haozhe Xie
# @Date:   2023-07-19 10:39:51
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-19 11:59:29
# @Email:  root@haozhexie.com
# @Ref: https://github.com/sczhou/CodeMOVI

import torch
import torch.nn.functional as F


class SmoothnessLoss(torch.nn.Module):
    def __init__(self, use_diag=True, size=None, device="cuda"):
        super(SmoothnessLoss, self).__init__()
        self.use_diag = use_diag
        self.filters = self._get_filters(use_diag, device)
        # Masks would generated for faster training if tensor size is specified
        assert size is None or len(size) == 4, "Size should be (B, C, H, W)"
        self.masks = None if size is None else self._get_masks(size, use_diag, device)

    def forward(self, input, target):
        masks = (
            self.masks
            if self.masks is not None
            else self._get_masks(input.size(), self.use_diag, input.device)
        )
        grad_input = self._get_grads(input)
        grad_target = self._get_grads(target)
        diff = F.smooth_l1_loss(grad_input, grad_target, reduction="none")
        return (diff * masks).mean()

    def _get_filters(self, use_diag, device):
        FILTER_X = torch.tensor([[0, 0, 0.0], [1, -2, 1], [0, 0, 0]], device=device)
        FILTER_Y = torch.tensor([[0, 1, 0.0], [0, -2, 0], [0, 1, 0]], device=device)
        FILTER_DIAG1 = torch.tensor([[1, 0, 0.0], [0, -2, 0], [0, 0, 1]], device=device)
        FILTER_DIAG2 = torch.tensor([[0, 0, 1.0], [0, -2, 0], [1, 0, 0]], device=device)
        if use_diag:
            filters = torch.stack([FILTER_X, FILTER_Y, FILTER_DIAG1, FILTER_DIAG2])
        else:
            filters = torch.stack([FILTER_X, FILTER_Y])

        return filters.unsqueeze(dim=1)

    def _get_grads(self, tensor):
        return F.conv2d(tensor, self.filters, stride=1, padding=1)

    def _get_masks(self, size, use_diag, device):
        MASK_X = self._get_mask(size, [[0, 0], [0, 1]], device)
        MASK_Y = self._get_mask(size, [[0, 1], [0, 0]], device)
        MASK_DIAG = self._get_mask(size, [[1, 1], [1, 1]], device)
        if use_diag:
            return torch.cat((MASK_X, MASK_Y, MASK_DIAG, MASK_DIAG), dim=1)
        else:
            return torch.cat((MASK_X, MASK_Y), dim=1)

    def _get_mask(self, size, paddings, device):
        """
        size: [b, c, h, w]
        paddings: [2 x 2] shape list, the first row indicates up and down paddings
        the second row indicates left and right paddings
        |            |
        |       x    |
        |     x * x  |
        |       x    |
        |            |
        """
        inner_height = size[2] - (paddings[0][0] + paddings[0][1])
        inner_width = size[3] - (paddings[1][0] + paddings[1][1])
        inner = torch.ones([inner_height, inner_width], device=device)
        torch_paddings = [
            paddings[1][0],
            paddings[1][1],
            paddings[0][0],
            paddings[0][1],
        ]  # left, right, up and down
        mask2d = F.pad(inner, pad=torch_paddings)
        return mask2d.unsqueeze(0).repeat(size[0], 1, 1).unsqueeze(1).detach()
