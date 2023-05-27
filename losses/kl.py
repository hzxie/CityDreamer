# -*- coding: utf-8 -*-
#
# @File:   kl.py
# @Author: NVIDIA CORPORATION & AFFILIATES
# @Date:   2023-05-26 20:10:08
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-26 20:11:24
# @Email:  root@haozhexie.com
# @Ref: https://github.com/NVlabs/imaginaire

import torch


class GaussianKLLoss(torch.nn.Module):
    r"""Compute KL loss in VAE for Gaussian distributions"""

    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu, logvar=None):
        r"""Compute loss

        Args:
            mu (tensor): mean
            logvar (tensor): logarithm of variance
        """
        if logvar is None:
            logvar = torch.zeros_like(mu)

        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
