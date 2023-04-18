# -*- coding: utf-8 -*-
#
# @File:   gancraft.py
# @Author: Zhaoxi Chen (@FrozenBurning)
# @Date:   2023-04-12 19:53:21
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-15 15:44:21
# @Email:  root@haozhexie.com
# @Ref: https://github.com/FrozenBurning/SceneDreamer

import numpy as np
import torch
import torch.nn.functional as F

import extensions.grid_encoder
import extensions.voxlib


class GanCraftGenerator(torch.nn.Module):
    def __init__(self, cfg):
        super(GanCraftGenerator, self).__init__()
        self.cfg = cfg
        self.cond_hash_grid = ConditionalHashGrid(cfg)
        self.grid_encoder = extensions.grid_encoder.GridEncoder(
            in_channels=5,
            n_levels=cfg.NETWORK.GANCRAFT.GRID_N_LEVELS,
            lvl_channels=cfg.NETWORK.GANCRAFT.GRID_LEVEL_DIM,
            desired_resolution=cfg.NETWORK.VQGAN.RESOLUTION,
        )
        self.sky_net = SkyMLP(cfg)
        self.render_net = RenderMLP(cfg)
        self.denoiser = RenderCNN(cfg)

    def forward(self, images, voxel_id, depth2, raydirs, cam_ori_t):
        r"""GANcraft Generator forward.

        Args:
            images (N x (1 + M) x H' x W' tensor) : height field + seg map, where M is the number of classes.
            voxel_id (N x H x W x max_samples x 1 tensor): IDs of intersected tensors along each ray.
            depth2 (N x 2 x H x W x max_samples x 1 tensor): Depths of entrance and exit points for each ray-voxel
            intersection.
            raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
            cam_ori_t (N x 3 tensor): Camera origins.
        Returns:
            fake_images (N x 3 x H x W tensor): fake images
        """
        bs, device = images.size(0), images.device
        global_features = self.cond_hash_grid(images)
        z = torch.randn(
            bs,
            self.cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            dtype=torch.float32,
            device=device,
        )
        net_out = self._forward_perpix(
            global_features, voxel_id, depth2, raydirs, cam_ori_t, z
        )
        fake_images = self._forward_global(net_out, z)
        return fake_images

    def _forward_perpix(self, global_features, voxel_id, depth2, raydirs, cam_ori_t, z):
        r"""Sample points along rays, forwarding the per-point MLP and aggregate pixel features

        Args:
            global_features (N x C1 tensor): Global block features determined by the current scene.
            voxel_id (N x H x W x M x 1 tensor): Voxel ids from ray-voxel intersection test. M: num intersected voxels
            depth2 (N x 2 x H x W x M x 1 tensor): Depths of entrance and exit points for each ray-voxel intersection.
            raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
            cam_ori_t (N x 3 tensor): Camera origins.
            z (N x C3 tensor): Intermediate style vectors.
        """
        # Generate sky_mask; PE transform on ray direction.
        with torch.no_grad():
            # sky_mask: when True, ray finally hits sky
            sky_mask = voxel_id[:, :, :, [-1], :] == 0
            # sky_only_mask: when True, ray hits nothing but sky
            sky_only_mask = voxel_id[:, :, :, [0], :] == 0

        with torch.no_grad():
            # Random sample points along the ray
            n_samples = self.cfg.NETWORK.GANCRAFT.N_SAMPLE_POINTS_PER_RAY + 1
            rand_depth, new_dists, new_idx = self._sample_depth_batched(
                depth2,
                n_samples,
                deterministic=True,
                use_box_boundaries=False,
                sample_depth=3,
            )
            worldcoord2 = raydirs * rand_depth + cam_ori_t[:, None, None, None, :]

            # Generate per-sample segmentation label
            mc_masks = torch.gather(voxel_id, -2, new_idx)
            # print(mc_masks.size())  # torch.Size([N, H, W, n_samples, 1])
            mc_masks = mc_masks.long()
            mc_masks_onehot = torch.zeros(
                [
                    mc_masks.size(0),
                    mc_masks.size(1),
                    mc_masks.size(2),
                    mc_masks.size(3),
                    self.cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
                ],
                dtype=torch.float,
                device=voxel_id.device,
            )
            # print(mc_masks_onehot.size())  # torch.Size([N, H, W, n_samples, 1])
            mc_masks_onehot.scatter_(-1, mc_masks, 1.0)

        net_out_s, net_out_c = self._forward_perpix_sub(
            global_features, worldcoord2, z, mc_masks_onehot
        )
        # Handle sky
        sky_raydirs_in = raydirs.expand(-1, -1, -1, 1, -1).contiguous()
        sky_raydirs_in = extensions.voxlib.positional_encoding(
            sky_raydirs_in,
            self.cfg.NETWORK.GANCRAFT.SKY_POS_EMD_DEG,
            -1,
            self.cfg.NETWORK.GANCRAFT.SKY_INC_ORIG_RAYDIR,
        )
        skynet_out_c = self.sky_net(sky_raydirs_in, z)

        # Blending
        weights = self._volum_rendering_relu(
            net_out_s, new_dists * self.cfg.NETWORK.GANCRAFT.DIST_SCALE, dim=-2
        )

        # If a ray exclusively hits the sky (no intersection with the voxels), set its weight to zero.
        weights = weights * torch.logical_not(sky_only_mask).float()
        total_weights_raw = torch.sum(weights, dim=-2, keepdim=True)  # 256 256 1 1
        total_weights = total_weights_raw

        is_gnd = worldcoord2[..., [0]] <= 1.0
        is_gnd = is_gnd.any(dim=-2, keepdim=True)
        nosky_mask = torch.logical_or(torch.logical_not(sky_mask), is_gnd)
        nosky_mask = nosky_mask.float()

        # Avoid sky leakage
        sky_weight = 1.0 - total_weights
        # keep_sky_out_avgpool overrides sky_replace_color
        sky_avg = torch.mean(skynet_out_c, dim=[1, 2], keepdim=True)
        skynet_out_c = skynet_out_c * (1.0 - nosky_mask) + sky_avg * (nosky_mask)

        rgbs = torch.clamp(net_out_c, -1, 1) + 1
        rgbs_sky = torch.clamp(skynet_out_c, -1, 1) + 1
        net_out = (
            torch.sum(weights * rgbs, dim=-2, keepdim=True) + sky_weight * rgbs_sky
        )
        net_out = net_out.squeeze(-2)
        net_out = net_out - 1
        return net_out

    def _sample_depth_batched(
        self,
        depth2,
        n_samples,
        deterministic=False,
        use_box_boundaries=True,
        sample_depth=3,
    ):
        r"""Make best effort to sample points within the same distance for every ray.
        Exception: When there is not enough voxel.

        Args:
            depth2 (N x 2 x H x W x M x 1 tensor):
            - N: Batch.
            - 2: Entrance / exit depth for each intersected box.
            - H, W: Height, Width.
            - M: Number of intersected boxes along the ray.
            - 1: One extra dim for consistent tensor dims.
            depth2 can include NaNs.
            deterministic (bool): Whether to use equal-distance sampling instead of random stratified sampling.
            use_box_boundaries (bool): Whether to add the entrance / exit points into the sample.
            sample_depth (float): Truncate the ray when it travels further than sample_depth inside voxels.
        """

        bs = depth2.size(0)
        dim0 = depth2.size(2)
        dim1 = depth2.size(3)
        dists = depth2[:, 1] - depth2[:, 0]
        dists[torch.isnan(dists)] = 0
        # print(dists.size())  # torch.Size([N, H, W, M, 1])
        accu_depth = torch.cumsum(dists, dim=-2)
        # print(accu_depth.size())  # torch.Size([N, H, W, M, 1])
        total_depth = accu_depth[..., [-1], :]
        # print(total_depth.size())  # torch.Size([N, H, W, 1, 1])
        total_depth = torch.clamp(total_depth, None, sample_depth)

        # Ignore out of range box boundaries. Fill with random samples.
        if use_box_boundaries:
            boundary_samples = accu_depth.clone().detach()
            boundary_samples_filler = torch.rand_like(boundary_samples) * total_depth
            bad_mask = (accu_depth > sample_depth) | (dists == 0)
            boundary_samples[bad_mask] = boundary_samples_filler[bad_mask]

        rand_shape = [bs, dim0, dim1, n_samples, 1]
        if deterministic:
            rand_samples = torch.empty(
                rand_shape, dtype=total_depth.dtype, device=total_depth.device
            )
            rand_samples[..., :, 0] = torch.linspace(0, 1, n_samples + 2)[1:-1]
        else:
            rand_samples = torch.rand(
                rand_shape, dtype=total_depth.dtype, device=total_depth.device
            )
            # Stratified sampling as in NeRF
            rand_samples = rand_samples / n_samples
            rand_samples[..., :, 0] += torch.linspace(
                0, 1, n_samples + 1, device=rand_samples.device
            )[:-1]

        rand_samples = rand_samples * total_depth
        # print(rand_samples.size())  # torch.Size([N, H, W, n_samples, 1])

        # Can also include boundaries
        if use_box_boundaries:
            rand_samples = torch.cat(
                [
                    rand_samples,
                    boundary_samples,
                    torch.zeros(
                        [bs, dim0, dim1, 1, 1],
                        dtype=total_depth.dtype,
                        device=total_depth.device,
                    ),
                ],
                dim=-2,
            )
        rand_samples, _ = torch.sort(rand_samples, dim=-2, descending=False)
        midpoints = (rand_samples[..., 1:, :] + rand_samples[..., :-1, :]) / 2
        # print(midpoints.size())  # torch.Size([N, H, W, n_samples, 1])
        new_dists = rand_samples[..., 1:, :] - rand_samples[..., :-1, :]

        # Scatter the random samples back
        # print(midpoints.unsqueeze(-3).size())   # torch.Size([N, H, W, 1, n_samples, 1])
        # print(accu_depth.unsqueeze(-2).size())  # torch.Size([N, H, W, M, 1, 1])
        idx = torch.sum(midpoints.unsqueeze(-3) > accu_depth.unsqueeze(-2), dim=-3)
        # print(idx.shape, idx.max(), idx.min()) # torch.Size([N, H, W, n_samples, 1]) max 5, min 0

        depth_deltas = (
            depth2[:, 0, :, :, 1:, :] - depth2[:, 1, :, :, :-1, :]
        )  # There might be NaNs!
        # print(depth_deltas.size())  # torch.Size([N, H, W, M, M - 1, 1])
        depth_deltas = torch.cumsum(depth_deltas, dim=-2)
        depth_deltas = torch.cat(
            [depth2[:, 0, :, :, [0], :], depth_deltas + depth2[:, 0, :, :, [0], :]],
            dim=-2,
        )
        heads = torch.gather(depth_deltas, -2, idx)
        # print(heads.size())  # torch.Size([N, H, W, M, 1])
        # print(torch.any(torch.isnan(heads)))
        rand_depth = heads + midpoints
        # print(rand_depth.size())  # torch.Size([N, H, W, M, n_samples, 1])
        return rand_depth, new_dists, idx

    def _volum_rendering_relu(self, sigma, dists, dim=2):
        free_energy = F.relu(sigma) * dists
        a = 1 - torch.exp(-free_energy.float())  # probability of it is not empty here
        b = torch.exp(
            -self._cumsum_exclusive(free_energy, dim=dim)
        )  # probability of everything is empty up to now
        return a * b  # probability of the ray hits something here

    def _cumsum_exclusive(self, tensor, dim):
        cumsum = torch.cumsum(tensor, dim)
        cumsum = torch.roll(cumsum, 1, dim)
        cumsum.index_fill_(
            dim, torch.tensor([0], dtype=torch.long, device=tensor.device), 0
        )
        return cumsum

    def _forward_perpix_sub(self, global_features, worldcoord2, z, mc_masks_onehot):
        r"""Forwarding the MLP.

        Args:
            global_features (N x C1 tensor): Global block features determined by the current scene.
            worldcoord2 (N x H x W x L x 3 tensor): 3D world coordinates of sampled points. L is number of samples; N is batch size, always 1.
            z (N x C3 tensor): Intermediate style vectors.
            mc_masks_onehot (N x H x W x L x C4): One-hot segmentation maps.
        Returns:
            net_out_s (N x H x W x L x 1 tensor): Opacities.
            net_out_c (N x H x W x L x C5 tensor): Color embeddings.
        """
        h, w, d = (
            self.cfg.NETWORK.VQGAN.RESOLUTION,
            self.cfg.NETWORK.VQGAN.RESOLUTION,
            self.cfg.DATASETS.OSM_LAYOUT.MAX_HEIGHT,
        )
        delimeter = torch.tensor([h, w, d], device=worldcoord2.device)
        normalized_cord = worldcoord2 / delimeter * 2 - 1
        # TODO: NAN VALUE FOUND!!
        # print(delimeter, torch.min(normalized_cord), torch.max(normalized_cord))
        global_features = global_features[:, None, None, None, :].repeat(
            1,
            normalized_cord.size(1),
            normalized_cord.size(2),
            normalized_cord.size(3),
            1,
        )
        normalized_cord = torch.cat([normalized_cord, global_features], dim=-1)
        feature_in = self.grid_encoder(normalized_cord)
        net_out_s, net_out_c = self.render_net(feature_in, z, mc_masks_onehot)
        return net_out_s, net_out_c

    def _forward_global(self, net_out, z):
        r"""Forward the CNN

        Args:
            net_out (N x C5 x H x W tensor): Intermediate feature maps.
            z (N x C3 tensor): Intermediate style vectors.

        Returns:
            fake_images (N x 3 x H x W tensor): Output image.
        """
        fake_images = net_out.permute(0, 3, 1, 2).contiguous()
        fake_images_raw = self.denoiser(fake_images, z)
        fake_images = torch.tanh(fake_images_raw)
        return fake_images


class ConditionalHashGrid(torch.nn.Module):
    def __init__(self, cfg):
        super(ConditionalHashGrid, self).__init__()
        n_classes = cfg.DATASETS.OSM_LAYOUT.N_CLASSES
        self.hf_conv = torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.seg_conv = torch.nn.Conv2d(
            n_classes,
            8,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        conv_blocks = []
        cur_hidden_channels = 16
        for _ in range(1, cfg.NETWORK.GANCRAFT.HASH_GRID_N_BLOCKS):
            conv_blocks.append(
                SRTConvBlock(in_channels=cur_hidden_channels, out_channels=None)
            )
            cur_hidden_channels *= 2

        self.conv_blocks = torch.nn.Sequential(*conv_blocks)
        self.fc1 = torch.nn.Linear(cur_hidden_channels, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, images):
        # Skip the 2nd channel for footprint boundaries
        hf = self.act(self.hf_conv(images[:, [0]]))
        seg = self.act(self.seg_conv(images[:, 2:]))
        joint = torch.cat([hf, seg], dim=1)
        for layer in self.conv_blocks:
            out = self.act(layer(joint))
            joint = out

        out = out.permute(0, 2, 3, 1)
        out = torch.mean(out.reshape(out.shape[0], -1, out.shape[-1]), dim=1)
        cond = self.act(self.fc1(out))
        cond = torch.tanh(self.fc2(cond))
        return cond


class RenderMLP(torch.nn.Module):
    r"""MLP with affine modulation."""

    def __init__(self, cfg):
        super(RenderMLP, self).__init__()
        self.fc_m_a = torch.nn.Linear(
            cfg.DATASETS.OSM_LAYOUT.N_CLASSES,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            bias=False,
        )
        self.fc_1 = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.GRID_N_LEVELS * cfg.NETWORK.GANCRAFT.GRID_LEVEL_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.fc_2 = ModLinear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_3 = ModLinear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_4 = ModLinear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )

        self.fc_sigma = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_SIGMA,
        )

        self.fc_5 = ModLinear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_6 = ModLinear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            bias=False,
            mod_bias=True,
            output_mode=True,
        )
        self.fc_out_c = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_COLOR,
        )
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, z, m):
        r"""Forward network

        Args:
            x (N x H x W x M x in_channels tensor): Projected features.
            z (N x cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM tensor): Style codes.
            m (N x H x W x M x mask_channels tensor): One-hot segmentation maps.
        """
        # b, h, w, n, _ = x.size()
        z = z[:, None, None, None, :]
        f = self.fc_1(x)
        f = f + self.fc_m_a(m)
        # Common MLP
        f = self.act(f)
        f = self.act(self.fc_2(f, z))
        f = self.act(self.fc_3(f, z))
        f = self.act(self.fc_4(f, z))
        # Sigma MLP
        sigma = self.fc_sigma(f)
        # Color MLP
        f = self.act(self.fc_5(f, z))
        f = self.act(self.fc_6(f, z))
        c = self.fc_out_c(f)
        return sigma, c


class SkyMLP(torch.nn.Module):
    r"""MLP converting ray directions to sky features."""

    def __init__(self, cfg):
        super(SkyMLP, self).__init__()
        self.fc_z_a = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            bias=False,
        )
        self.fc1 = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.SKY_IN_CHANNEL_BASE
            * cfg.NETWORK.GANCRAFT.SKY_POS_EMD_DEG
            * 2
            + cfg.NETWORK.GANCRAFT.SKY_IN_CHANNEL_BASE,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.fc2 = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.fc3 = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.fc4 = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.fc5 = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.fc_out_c = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_COLOR,
        )
        self.act = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, z):
        r"""Forward network

        Args:
            x (... x in_channels tensor): Ray direction embeddings.
            z (... x cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM tensor): Style codes.
        """
        z = self.fc_z_a(z)
        while z.dim() < x.dim():
            z = z.unsqueeze(1)

        y = self.act(self.fc1(x) + z)
        y = self.act(self.fc2(y))
        y = self.act(self.fc3(y))
        y = self.act(self.fc4(y))
        y = self.act(self.fc5(y))
        c = self.fc_out_c(y)

        return c


class RenderCNN(torch.nn.Module):
    r"""CNN converting intermediate feature map to final image."""

    def __init__(self, cfg):
        super(RenderCNN, self).__init__()
        self.fc_z_cond = torch.nn.Linear(
            cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM,
            2 * 2 * cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
        )
        self.conv1 = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_COLOR,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            1,
            stride=1,
            padding=0,
        )
        self.conv2a = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            3,
            stride=1,
            padding=1,
        )
        self.conv2b = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv3a = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            3,
            stride=1,
            padding=1,
        )
        self.conv3b = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv4a = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            1,
            stride=1,
            padding=0,
        )
        self.conv4b = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM,
            1,
            stride=1,
            padding=0,
        )
        self.conv4 = torch.nn.Conv2d(
            cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM, 3, 1, stride=1, padding=0
        )
        self.act = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def modulate(self, x, w, b):
        w = w[..., None, None]
        b = b[..., None, None]
        return x * (w + 1) + b

    def forward(self, x, z):
        r"""Forward network.

        Args:
            x (N x in_channels x H x W tensor): Intermediate feature map
            z (N x cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM tensor): Style codes.
        """
        z = self.fc_z_cond(z)
        adapt = torch.chunk(z, 2 * 2, dim=-1)

        y = self.act(self.conv1(x))

        y = y + self.conv2b(self.act(self.conv2a(y)))
        y = self.act(self.modulate(y, adapt[0], adapt[1]))

        y = y + self.conv3b(self.act(self.conv3a(y)))
        y = self.act(self.modulate(y, adapt[2], adapt[3]))

        y = y + self.conv4b(self.act(self.conv4a(y)))
        y = self.act(y)

        y = self.conv4(y)

        return y


class SRTConvBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super(SRTConvBlock, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = 2 * hidden_channels

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                hidden_channels,
                stride=1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                hidden_channels,
                out_channels,
                stride=2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ModLinear(torch.nn.Module):
    r"""Linear layer with affine modulation (Based on StyleGAN2 mod demod).
    Equivalent to affine modulation following linear, but faster when the same modulation parameters are shared across
    multiple inputs.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        style_features (int): Number of style features.
        bias (bool): Apply additive bias before the activation function?
        mod_bias (bool): Whether to modulate bias.
        output_mode (bool): If True, modulate output instead of input.
        weight_gain (float): Initialization gain
    """

    def __init__(
        self,
        in_features,
        out_features,
        style_features,
        bias=True,
        mod_bias=True,
        output_mode=False,
        weight_gain=1,
        bias_init=0,
    ):
        super(ModLinear, self).__init__()
        weight_gain = weight_gain / np.sqrt(in_features)
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) * weight_gain
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_alpha = torch.nn.Parameter(
            torch.randn([in_features, style_features]) / np.sqrt(style_features)
        )
        self.bias_alpha = torch.nn.Parameter(
            torch.full([in_features], 1, dtype=torch.float)
        )  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        self.output_mode = output_mode
        if mod_bias:
            if output_mode:
                mod_bias_dims = out_features
            else:
                mod_bias_dims = in_features
            self.weight_beta = torch.nn.Parameter(
                torch.randn([mod_bias_dims, style_features]) / np.sqrt(style_features)
            )
            self.bias_beta = torch.nn.Parameter(
                torch.full([mod_bias_dims], 0, dtype=torch.float)
            )

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, 1, 1, , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        w = self.weight.to(x.dtype)  # [O I]
        w = w.unsqueeze(0) * alpha  # [1 O I] * [B 1 I] = [B O I]

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            if not self.output_mode:
                x = x + beta

        b = self.bias
        if b is not None:
            b = b.to(x.dtype)[None, None, :]
        if self.mod_bias and self.output_mode:
            if b is None:
                b = beta
            else:
                b = b + beta

        # [B ? I] @ [B I O] = [B ? O]
        if b is not None:
            x = torch.baddbmm(b, x, w.transpose(1, 2))
        else:
            x = x.bmm(w.transpose(1, 2))
        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x


class AffineMod(torch.nn.Module):
    r"""Learning affine modulation of activation.

    Args:
        in_features (int): Number of input features.
        style_features (int): Number of style features.
        mod_bias (bool): Whether to modulate bias.
    """

    def __init__(self, in_features, style_features, mod_bias=True):
        super(AffineMod, self).__init__()
        self.weight_alpha = torch.nn.Parameter(
            torch.randn([in_features, style_features]) / np.sqrt(style_features)
        )
        self.bias_alpha = torch.nn.Parameter(
            torch.full([in_features], 1, dtype=torch.float)
        )  # init to 1
        self.weight_beta = None
        self.bias_beta = None
        self.mod_bias = mod_bias
        if mod_bias:
            self.weight_beta = torch.nn.Parameter(
                torch.randn([in_features, style_features]) / np.sqrt(style_features)
            )
            self.bias_beta = torch.nn.Parameter(
                torch.full([in_features], 0, dtype=torch.float)
            )

    @staticmethod
    def _linear_f(x, w, b):
        w = w.to(x.dtype)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        if b is not None:
            b = b.to(x.dtype)
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
        x = x.reshape(*x_shape[:-1], -1)
        return x

    # x: B, ...   , Cin
    # z: B, 1, 1, , Cz
    def forward(self, x, z):
        x_shape = x.shape
        z_shape = z.shape
        x = x.reshape(x_shape[0], -1, x_shape[-1])
        z = z.reshape(z_shape[0], 1, z_shape[-1])

        alpha = self._linear_f(z, self.weight_alpha, self.bias_alpha)  # [B, ..., I]
        x = x * alpha

        if self.mod_bias:
            beta = self._linear_f(z, self.weight_beta, self.bias_beta)  # [B, ..., I]
            x = x + beta

        x = x.reshape(*x_shape[:-1], x.shape[-1])
        return x