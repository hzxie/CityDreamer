# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 09:50:37
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 19:49:41
# @Email:  root@haozhexie.com

import logging
import os
import torch

import core.test
import models.vqvae
import utils.average_meter
import utils.datasets
import utils.helpers
import utils.summary_writer


from time import time


def train(cfg):
    torch.backends.cudnn.benchmark = True
    # Set up networks
    network = None
    network_name = cfg.TRAIN.NETWORK
    dataset_name = cfg.TRAIN[network_name].DATASET
    if cfg.TRAIN.NETWORK == "VQGAN":
        network = models.vqvae.VQVAE(cfg)
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()

    # Current train config
    cfg.TRAIN = cfg.TRAIN[cfg.TRAIN.NETWORK]

    # Set up data loader
    train_data_loader = torch.utils.data.DataLoader(
        dataset=utils.datasets.get_dataset(cfg, dataset_name, "train"),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=utils.datasets.get_dataset(cfg, dataset_name, "val"),
        batch_size=1,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=True,
        shuffle=False,
    )

    # Set up optimizers
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, network.parameters()),
        lr=cfg.TRAIN.BASE_LR * cfg.TRAIN.BATCH_SIZE * torch.cuda.device_count(),
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=cfg.TRAIN.BETAS,
    )
    # TODO: Enable it later
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.N_EPOCHS)

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()

    # Load the pretrained model if exists
    init_epoch = 0
    if "WEIGHTS" in cfg.CONST:
        logging.info("Recovering from %s ..." % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        network.load_state_dict(checkpoint[network_name])
        logging.info("Recover completed. Current epoch = #%d" % (init_epoch,))

    # Set up folders for logs, snapshot and checkpoints
    output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
    cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
    cfg.DIR.LOGS = output_dir % "logs"
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Summary writer
    tb_writer = utils.summary_writer.SummaryWriter(cfg)

    # Training/Testing the network
    n_batches = len(train_data_loader)
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        losses = utils.average_meter.AverageMeter(["RecLoss", "QuantLoss"])

        batch_end_time = time()
        for batch_idx, data in enumerate(train_data_loader):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time() - batch_end_time)

            try:
                input = utils.helpers.var_or_cuda(data["input"])
                output = utils.helpers.var_or_cuda(data["output"])
                pred = network(input)
                loss = l1_loss(pred["output"], output) + pred["loss"]

                losses.update([loss.item(), pred["loss"]])
                network.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as ex:
                logging.exception(ex)
                continue

            tb_writer.add_scalars(
                {
                    "Loss/Batch/Rec": loss.item(),
                    "Loss/Batch/Quant": pred["loss"].item(),
                },
                n_itr,
            )
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info(
                "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s"
                % (
                    epoch_idx,
                    cfg.TRAIN.N_EPOCHS,
                    batch_idx + 1,
                    n_batches,
                    batch_time.val(),
                    data_time.val(),
                    ["%.4f" % l for l in losses.val()],
                )
            )
        # TODO: Enable it later
        # lr_scheduler.step()
        epoch_end_time = time()
        tb_writer.add_scalars(
            {
                "Loss/Epoch/Rec/Train": losses.avg(0),
                "Loss/Epoch/Quant/Train": losses.avg(1),
            },
            epoch_idx,
        )
        logging.info(
            "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s"
            % (
                epoch_idx,
                cfg.TRAIN.N_EPOCHS,
                epoch_end_time - epoch_start_time,
                ["%.4f" % l for l in losses.avg()],
            )
        )

        # Evaluate the current model
        losses, key_frames = core.test(cfg, val_data_loader, network)
        tb_writer.add_scalars(
            {
                "Loss/Epoch/Rec/Test": losses.avg(0),
                "Loss/Epoch/Quant/Test": losses.avg(1),
            },
            epoch_idx,
        )
        tb_writer.add_images(key_frames, epoch_idx)
        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.CKPT_SAVE_FREQ == 0:
            output_path = os.path.join(
                cfg.DIR.CHECKPOINTS, "ckpt-epoch-%06d.pth" % epoch_idx
            )
            torch.save(
                {
                    "cfg": cfg,
                    "epoch_index": epoch_idx,
                    network_name: network.state_dict(),
                },
                output_path,
            )
            logging.info("Saved checkpoint to %s ..." % output_path)

    tb_writer.close()
