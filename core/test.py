# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 09:50:44
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 21:03:13
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
    test_losses = utils.average_meter.AverageMeter(["RecLoss", "QuantLoss"])
    key_frames = {}

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()

    # Testing loop
    for idx, data in enumerate(test_data_loader):
        with torch.no_grad():
            input = utils.helpers.var_or_cuda(data["input"], network.device)
            output = utils.helpers.var_or_cuda(data["output"], network.device)
            pred = network(input)
            loss = l1_loss(pred["output"], output) + pred["loss"]
            test_losses.update([loss.item(), pred["loss"]])

            key_frame_prefix = "Image/%04d" % idx
            _key_frames = utils.helpers.get_keyframes(
                torch.cat([pred["output"], output], dim=3).squeeze()
            )
            for k, v in _key_frames.items():
                _key = (
                    "%s/%s" % (key_frame_prefix, k)
                    if k is not None
                    else key_frame_prefix
                )
                key_frames[_key] = v

            logging.info(
                "Test[%d/%d] Losses = %s"
                % (idx + 1, n_samples, ["%.4f" % l for l in test_losses.val()])
            )

    return test_losses, key_frames
