# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 09:50:44
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-11 16:06:11
# @Email:  root@haozhexie.com

import logging
import torch

import models.vqgan
import utils.average_meter
import utils.datasets
import utils.helpers


def test(cfg, test_data_loader=None, network=None):
    torch.backends.cudnn.benchmark = True
    if network is None:
        vqae = models.vqgan.VQAutoEncoder(cfg)
        if torch.cuda.is_available():
            vqae = torch.nn.DataParallel(vqae).cuda()

        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        vqae.load_state_dict(checkpoint["vqae"])

    if test_data_loader is None:
        test_data_loader = torch.utils.data.DataLoader(
            dataset=utils.datasets.get_dataset(cfg, "OSM_LAYOUT", "test"),
            batch_size=1,
            num_workers=cfg.CONST.N_WORKERS,
            collate_fn=utils.datasets.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    # Switch models to evaluation mode
    vqae.eval()

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = utils.average_meter.AverageMeter(
        ["RecLoss", "SegLoss", "QuantLoss", "TotalLoss"]
    )
    key_frames = {}
    for idx, data in enumerate(test_data_loader):
        with torch.no_grad():
            input = utils.helpers.var_or_cuda(data["input"])
            output = utils.helpers.var_or_cuda(data["output"])
            pred, quant_loss = vqae(input)
            rec_loss = l1_loss(pred[:, 0], output[:, 0])
            seg_loss = ce_loss(pred[:, 1:], torch.argmax(output[:, 1:], dim=1))
            loss = (
                rec_loss * cfg.TRAIN.VQGAN.REC_LOSS_FACTOR
                + seg_loss * cfg.TRAIN.VQGAN.SEG_LOSS_FACTOR
                + quant_loss
            )
            test_losses.update(
                [rec_loss.item(), seg_loss.item(), quant_loss.item(), loss.item()]
            )

            key_frames["Image/%04d/HeightField" % idx] = utils.helpers.tensor_to_image(
                torch.cat([pred[:, 0], output[:, 0]], dim=2), "HeightField"
            )
            key_frames["Image/%04d/SegMap" % idx] = utils.helpers.tensor_to_image(
                utils.helpers.onehot_to_mask(
                    torch.cat(
                        [
                            pred[:, 1:],
                            output[:, 1:],
                        ],
                        dim=2,
                    ),
                    cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                ),
                "SegMap",
            )
            logging.info(
                "Test[%d/%d] Losses = %s"
                % (idx + 1, n_samples, ["%.4f" % l for l in test_losses.val()])
            )

    return test_losses, key_frames
