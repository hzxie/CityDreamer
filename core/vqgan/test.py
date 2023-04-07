# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 09:50:44
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-07 13:30:03
# @Email:  root@haozhexie.com

import logging
import torch

import utils.average_meter
import utils.datasets
import utils.helpers


def test(cfg, test_data_loader=None, network=None):
    torch.backends.cudnn.benchmark = True
    if network is None:
        # TODO
        network_name = ""
        network = None
        if torch.cuda.is_available():
            network = torch.nn.DataParallel(network).cuda()

        logging.info("Recovering from %s ..." % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        network.load_state_dict(checkpoint[network_name])

    # Switch models to evaluation mode
    network.eval()

    if test_data_loader is None:
        # TODO
        dataset_name = "OSM_LAYOUT"
        test_data_loader = torch.utils.data.DataLoader(
            dataset=utils.datasets.get_dataset(cfg, dataset_name, "val"),
            batch_size=1,
            num_workers=cfg.CONST.N_WORKERS,
            collate_fn=utils.datasets.collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = utils.average_meter.AverageMeter(
        ["RecLoss", "SegLoss", "QuantLoss", "TotalLoss"]
    )
    key_frames = {}

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()
    ce_loss = torch.nn.CrossEntropyLoss()

    # Testing loop
    for idx, data in enumerate(test_data_loader):
        with torch.no_grad():
            input = utils.helpers.var_or_cuda(data["input"], network.device)
            output = utils.helpers.var_or_cuda(data["output"], network.device)
            pred, quant_loss = network(input)
            rec_loss = l1_loss(pred[..., 0], output[..., 0])
            seg_loss = ce_loss(pred[:, 1:], torch.argmax(output[:, 1:], dim=1))
            loss = rec_loss + seg_loss + quant_loss
            test_losses.update(
                [rec_loss.item(), seg_loss.item(), quant_loss.item(), loss.item()]
            )

            key_frames["Image/%04d/HeightField" % idx] = utils.helpers.tensor_to_image(
                torch.cat([pred[:, 0], output[:, 0]], dim=2), "HeightField"
            )
            key_frames["Image/%04d/SegMap" % idx] = utils.helpers.tensor_to_image(
                torch.cat(
                    [
                        utils.helpers.onehot_to_mask(
                            pred[:, 1:],
                            cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                        ),
                        utils.helpers.onehot_to_mask(
                            output[:, 1:],
                            cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES,
                        ),
                    ], dim=2
                ),
                "SegMap",
            )
            logging.info(
                "Test[%d/%d] Losses = %s"
                % (idx + 1, n_samples, ["%.4f" % l for l in test_losses.val()])
            )

    return test_losses, key_frames
