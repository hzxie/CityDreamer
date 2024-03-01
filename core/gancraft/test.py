# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-04-21 19:46:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-01 15:29:19
# @Email:  root@haozhexie.com

import logging
import torch

import models.gancraft
import utils.average_meter
import utils.datasets
import utils.distributed
import utils.helpers


def test(cfg, test_data_loader=None, gancraft=None):
    torch.backends.cudnn.benchmark = True
    if gancraft is None:
        gancraft = models.gancraft.GanCraftGenerator(cfg)
        if torch.cuda.is_available():
            gancraft = torch.nn.DataParallel(gancraft).cuda()
            gancraft.device = gancraft.output_device

        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        if cfg.TRAIN.GANCRAFT.EMA_ENABLED:
            gancraft.load_state_dict(checkpoint["gancraft_g_ema"])
        else:
            gancraft.load_state_dict(checkpoint["gancraft_g"])

    if test_data_loader is None:
        test_data_loader = torch.utils.data.DataLoader(
            dataset=utils.datasets.get_dataset(cfg, cfg.TEST.GANCRAFT.DATASET, "test"),
            batch_size=1,
            num_workers=cfg.CONST.N_WORKERS,
            collate_fn=utils.datasets.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    # Switch models to evaluation mode
    gancraft.eval()

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = utils.average_meter.AverageMeter(["RecLoss"])
    key_frames = {}
    for idx, data in enumerate(test_data_loader):
        with torch.no_grad():
            hf_seg = utils.helpers.var_or_cuda(
                torch.cat([data["hf"], data["seg"]], dim=1), gancraft.device
            )
            voxel_id = utils.helpers.var_or_cuda(data["voxel_id"], gancraft.device)
            depth2 = utils.helpers.var_or_cuda(data["depth2"], gancraft.device)
            raydirs = utils.helpers.var_or_cuda(data["raydirs"], gancraft.device)
            cam_origin = utils.helpers.var_or_cuda(data["cam_origin"], gancraft.device)
            footage = utils.helpers.var_or_cuda(data["footage"], gancraft.device)
            building_stats = (
                None if "building_stats" not in data else data["building_stats"]
            )

            fake_imgs = gancraft(
                hf_seg, voxel_id, depth2, raydirs, cam_origin, building_stats
            )
            loss = l1_loss(fake_imgs, footage)
            test_losses.update([loss.item()])

            if utils.distributed.is_master():
                if idx < 3:
                    if cfg.NETWORK.GANCRAFT.BUILDING_MODE:
                        masks = torch.zeros_like(data["mask"], device=gancraft.device)
                        masks[
                            torch.isin(
                                voxel_id[:, None, ..., 0, 0],
                                torch.tensor(
                                    [
                                        cfg.NETWORK.GANCRAFT.FACADE_CLS_ID,
                                        cfg.NETWORK.GANCRAFT.ROOF_CLS_ID,
                                    ],
                                    device=gancraft.device,
                                ),
                            )
                        ] = 1
                        footage = footage * masks

                    key_frames[
                        "GANCraft/Image/%04d" % idx
                    ] = utils.helpers.tensor_to_image(
                        torch.cat([fake_imgs, footage], dim=3), "RGB"
                    )
                logging.info(
                    "Test[%d/%d] Losses = %s"
                    % (idx + 1, n_samples, ["%.4f" % l for l in test_losses.val()])
                )

    return test_losses, key_frames
