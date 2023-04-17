# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-04-10 10:46:40
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-18 15:49:55
# @Email:  root@haozhexie.com

import logging
import torch
import torch.nn.functional as F

import models.vqgan
import models.sampler
import utils.helpers


def test(cfg, vqae=None, sampler=None):
    torch.backends.cudnn.benchmark = True
    if vqae is None and sampler is None:
        vqae = models.vqgan.VQAutoEncoder(cfg)
        sampler = models.sampler.AbsorbingDiffusionSampler(cfg)
        if torch.cuda.is_available():
            vqae = torch.nn.DataParallel(vqae).cuda()
            sampler = torch.nn.DataParallel(sampler).cuda()
            vqae.device = vqae.output_device
            sampler.device = sampler.output_device
        else:
            vqae.device = torch.device("cpu")
            sampler.device = torch.device("cpu")

        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        vqae.load_state_dict(checkpoint["vqae"])
        sampler.load_state_dict(checkpoint["sampler"])

    # Switch models to evaluation mode
    vqae.eval()
    sampler.eval()

    # Testing loop
    codebook = vqae.module.quantize.get_codebook()
    # print(codebook.size())    # torch.Size([codebook_size, embed_dim])
    key_frames = {}
    for t in cfg.TEST.SAMPLER.TEMPERATURES:
        with torch.no_grad():
            min_encoding_indices = sampler.module.sample(
                cfg.TEST.SAMPLER.N_SAMPLES,
                cfg.NETWORK.SAMPLER.TOTAL_STEPS,
                temperature=t,
                device=sampler.device,
            )
            # print(min_encoding_indices.size())  # torch.Size([bs, att_size**2])
            min_encoding_indices = min_encoding_indices.unsqueeze(dim=2)
            one_hot = torch.zeros(
                (
                    cfg.TEST.SAMPLER.N_SAMPLES,
                    cfg.NETWORK.SAMPLER.BLOCK_SIZE,
                    cfg.NETWORK.VQGAN.N_EMBED,
                ),
                device=sampler.device,
            )
            one_hot.scatter_(2, min_encoding_indices, 1)
            # print(min_encoding_indices, torch.argmax(one_hot, dim=2))
            quant = (
                torch.matmul(one_hot.view(-1, cfg.NETWORK.VQGAN.N_EMBED), codebook)
                .float()
                .view(
                    cfg.TEST.SAMPLER.N_SAMPLES,
                    cfg.NETWORK.VQGAN.ATTN_RESOLUTION,
                    cfg.NETWORK.VQGAN.ATTN_RESOLUTION,
                    cfg.NETWORK.VQGAN.EMBED_DIM,
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # print(quant.size())   # torch.Size([bs, embed_dim, att_size, att_size])
            pred = vqae.module.decode(quant)
            key_frames["Image/T=%d/HeightField" % t] = utils.helpers.tensor_to_image(
                torch.cat([pred[0, 0], pred[1, 0]], dim=1).unsqueeze(dim=0),
                "HeightField",
            )
            key_frames["Image/T=%d/FootprintCtr" % t] = utils.helpers.tensor_to_image(
                torch.cat([torch.sigmoid(pred[0, 1]), pred[1, 1]], dim=1).unsqueeze(
                    dim=0
                ),
                "FootprintCtr",
            )
            key_frames["Image/T=%d/SegMap" % t] = utils.helpers.tensor_to_image(
                utils.helpers.onehot_to_mask(
                    torch.cat(
                        [
                            pred[0, 2:],
                            pred[1, 2:],
                        ],
                        dim=2,
                    ).unsqueeze(dim=0),
                    cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                ),
                "SegMap",
            )
    return key_frames
