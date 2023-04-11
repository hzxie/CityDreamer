# -*- coding: utf-8 -*-
#
# @File:   vqgan.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:09:04
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-10 20:52:20
# @Email:  root@haozhexie.com
# @Ref: https://github.com/CompVis/taming-transformers

import einops
import logging
import numpy as np
import torch


# Helper functions definition
nonlinearity = lambda x: x * torch.sigmoid(x)
normalize = lambda c_in: torch.nn.GroupNorm(
    num_groups=32, num_channels=c_in, eps=1e-6, affine=True
)


class VQAutoEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quantize = VectorQuantizer(cfg)
        self.quant_conv = torch.nn.Conv2d(
            cfg.NETWORK.VQGAN.N_Z_CHANNELS, cfg.NETWORK.VQGAN.EMBED_DIM, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            cfg.NETWORK.VQGAN.EMBED_DIM, cfg.NETWORK.VQGAN.N_Z_CHANNELS, 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_resolutions = len(cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS)
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            cfg.NETWORK.VQGAN.N_IN_CHANNELS,
            cfg.NETWORK.VQGAN.N_CHANNEL_BASE,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        cur_resolution = cfg.NETWORK.VQGAN.RESOLUTION
        in_channel_factor = (1,) + tuple(cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS)
        out_channel_factor = cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS
        self.down = torch.nn.ModuleList()
        for i_level in range(self.n_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = cfg.NETWORK.VQGAN.N_CHANNEL_BASE * in_channel_factor[i_level]
            block_out = cfg.NETWORK.VQGAN.N_CHANNEL_BASE * out_channel_factor[i_level]
            for _ in range(cfg.NETWORK.VQGAN.N_RES_BLOCKS):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=cfg.NETWORK.VQGAN.DROPOUT,
                    )
                )
                block_in = block_out
                if cur_resolution == cfg.NETWORK.VQGAN.ATTN_RESOLUTION:
                    attn.append(AttnBlock(block_in))
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.n_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                cur_resolution = cur_resolution // 2
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=cfg.NETWORK.VQGAN.DROPOUT,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=cfg.NETWORK.VQGAN.DROPOUT,
        )
        # end
        self.norm_out = normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            cfg.NETWORK.VQGAN.N_Z_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], resolution)
        # timestep embedding
        temb = None
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.n_resolutions):
            for i_block in range(self.cfg.NETWORK.VQGAN.N_RES_BLOCKS):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.n_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_resolutions = len(cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS)

        block_in = (
            cfg.NETWORK.VQGAN.N_CHANNEL_BASE
            * cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS[self.n_resolutions - 1]
        )
        cur_resolution = cfg.NETWORK.VQGAN.RESOLUTION // 2 ** (self.n_resolutions - 1)
        self.z_shape = (
            1,
            cfg.NETWORK.VQGAN.N_Z_CHANNELS,
            cur_resolution,
            cur_resolution,
        )
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )
        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            cfg.NETWORK.VQGAN.N_Z_CHANNELS, block_in, kernel_size=3, stride=1, padding=1
        )
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=cfg.NETWORK.VQGAN.DROPOUT,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=cfg.NETWORK.VQGAN.DROPOUT,
        )
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.n_resolutions)):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_out = (
                cfg.NETWORK.VQGAN.N_CHANNEL_BASE
                * cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS[i_level]
            )
            for _ in range(cfg.NETWORK.VQGAN.N_RES_BLOCKS + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=cfg.NETWORK.VQGAN.DROPOUT,
                    )
                )
                block_in = block_out
                if cur_resolution == cfg.NETWORK.VQGAN.ATTN_RESOLUTION:
                    attn.append(AttnBlock(block_in))
            up = torch.nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                cur_resolution = cur_resolution * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            cfg.NETWORK.VQGAN.N_OUT_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape
        # timestep embedding
        temb = None
        # z to block_in
        h = self.conv_in(z)
        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # upsampling
        for i_level in reversed(range(self.n_resolutions)):
            for i_block in range(self.cfg.NETWORK.VQGAN.N_RES_BLOCKS + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # end
        # give_pre_end is always set to False
        # if give_pre_end:
        #     return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(torch.nn.Module):
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, cfg):
        super().__init__()
        self.BETA = 0.25
        self.LEGACY = True
        self.cfg = cfg
        self.embedding = torch.nn.Embedding(
            cfg.NETWORK.VQGAN.N_EMBED, cfg.NETWORK.VQGAN.EMBED_DIM
        )
        self.embedding.weight.data.uniform_(
            -1.0 / cfg.NETWORK.VQGAN.N_EMBED, 1.0 / cfg.NETWORK.VQGAN.N_EMBED
        )

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = einops.rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.cfg.NETWORK.VQGAN.EMBED_DIM)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn",
                z_flattened,
                einops.rearrange(self.embedding.weight, "n d -> d n"),
            )
        )
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        # compute loss for embedding
        if not self.LEGACY:
            loss = self.BETA * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.BETA * torch.mean(
                (z_q - z.detach()) ** 2
            )
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = einops.rearrange(z_q, "b h w c -> b c h w").contiguous()

        return (
            z_q,
            loss,
            {
                "perplexity": perplexity,
                "min_encodings": min_encodings,
                "min_encoding_indices": min_encoding_indices,
            },
        )

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_
