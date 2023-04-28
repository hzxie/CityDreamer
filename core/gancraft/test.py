# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-04-21 19:46:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-28 16:05:53
# @Email:  root@haozhexie.com

import logging
import torch

import models.gancraft
import utils.average_meter
import utils.datasets
import utils.helpers


def test(cfg, test_data_loader=None, gancraft=None):
    torch.backends.cudnn.benchmark = True
    local_rank = 0
    if torch.cuda.is_available():
        local_rank = torch.distributed.get_rank()

    if gancraft is None:
        gancraft = models.gancraft.GanCraftGenerator(cfg)
        if torch.cuda.is_available():
            gancraft = torch.nn.DataParallel(gancraft).cuda()
            gancraft.device = gancraft.output_device

        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        gancraft.load_state_dict(checkpoint["gancraft"])

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
            cam_ori_t = utils.helpers.var_or_cuda(data["cam_ori_t"], gancraft.device)
            footage = utils.helpers.var_or_cuda(data["footage"], gancraft.device)

            fake_imgs = gancraft(hf_seg, voxel_id, depth2, raydirs, cam_ori_t)
            loss = l1_loss(fake_imgs, footage)
            test_losses.update([loss.item()])

            if local_rank == 0:
                if idx < 3:
                    key_frames["GANCraft/Image/%04d" % idx] = utils.helpers.tensor_to_image(
                        torch.cat([fake_imgs, footage], dim=3), "RGB"
                    )
                logging.info(
                    "Test[%d/%d] Losses = %s"
                    % (idx + 1, n_samples, ["%.4f" % l for l in test_losses.val()])
                )

    return test_losses, key_frames