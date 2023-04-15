# -*- coding: utf-8 -*-
#
# @File:   __init__.py
# @Author: Jiaxiang Tang (@ashawkey)
# @Date:   2023-04-15 10:39:28
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-15 13:08:46
# @Email:  ashawkey1999@gmail.com
# @Ref: https://github.com/ashawkey/torch-ngp

import math
import numpy as np
import torch

import grid_encoder_ext


class GridEncoderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        embeddings,
        offsets,
        per_level_scale,
        base_resolution,
        calc_grad_inputs=False,
        gridtype=0,
        align_corners=False,
    ):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float
        inputs = inputs.contiguous()
        # batch size, coord dim
        B, D = inputs.shape
        # level
        L = offsets.shape[0] - 1
        # embedding dim for each level
        C = embeddings.shape[1]
        # resolution multiplier at each level, apply log2 for later CUDA exp2f
        S = math.log2(per_level_scale)
        # base resolution
        H = base_resolution
        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(
                B, L * D * C, device=inputs.device, dtype=embeddings.dtype
            )
        else:
            dy_dx = torch.empty(
                1, device=inputs.device, dtype=embeddings.dtype
            )  # placeholder... TODO: a better way?

        grid_encoder_ext.forward(
            inputs,
            embeddings,
            offsets,
            outputs,
            B,
            D,
            C,
            L,
            S,
            H,
            calc_grad_inputs,
            dy_dx,
            gridtype,
            align_corners,
        )
        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype]
        ctx.calc_grad_inputs = calc_grad_inputs
        ctx.align_corners = align_corners

        return outputs

    @staticmethod
    def backward(ctx, grad):
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)

        if calc_grad_inputs:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = torch.zeros(1, device=inputs.device, dtype=embeddings.dtype)

        grid_encoder_ext.backward(
            grad,
            inputs,
            embeddings,
            offsets,
            grad_embeddings,
            B,
            D,
            C,
            L,
            S,
            H,
            calc_grad_inputs,
            dy_dx,
            grad_inputs,
            gridtype,
            align_corners,
        )

        if calc_grad_inputs:
            grad_inputs = grad_inputs.to(inputs.dtype)
            return grad_inputs, grad_embeddings, None, None, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None, None, None


class GridEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        n_levels,
        lvl_channels,
        desired_resolution,
        per_level_scale=2,
        base_resolution=16,
        log2_hashmap_size=19,
        gridtype="hash",
        align_corners=False,
    ):
        super(GridEncoder, self).__init__()
        self.in_channels = in_channels
        self.n_levels = n_levels  # num levels, each level multiply resolution by 2
        self.lvl_channels = lvl_channels  # encode channels per level
        self.per_level_scale = 2 ** (
            math.log2(desired_resolution / base_resolution) / (n_levels - 1)
        )
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = n_levels * lvl_channels
        self.gridtype = gridtype
        self.gridtype_id = 0 if gridtype == "hash" else 1
        self.align_corners = align_corners

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2**log2_hashmap_size
        for i in range(n_levels):
            resolution = int(math.ceil(base_resolution * per_level_scale**i))
            params_in_level = min(
                self.max_params,
                (resolution if align_corners else resolution + 1) ** in_channels,
            )  # limit max number
            params_in_level = int(math.ceil(params_in_level / 8) * 8)  # make divisible
            offsets.append(offset)
            offset += params_in_level

        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer("offsets", offsets)

        self.n_params = offsets[-1] * lvl_channels
        self.embeddings = torch.nn.Parameter(torch.empty(offset, lvl_channels))
        self._init_weights()

    def _init_weights(self):
        self.embeddings.data.uniform_(-1e-4, 1e-4)

    def forward(self, inputs, bound=1):
        # inputs: [..., in_channels], normalized real world positions in [-bound, bound]
        # return: [..., n_levels * lvl_channels]
        inputs = (inputs + bound) / (2 * bound)  # map to [0, 1]
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.in_channels)
        outputs = GridEncoderFunction.apply(
            inputs,
            self.embeddings,
            self.offsets,
            self.per_level_scale,
            self.base_resolution,
            inputs.requires_grad,
            self.gridtype_id,
            self.align_corners,
        )
        return outputs.view(prefix_shape + [self.output_dim])
