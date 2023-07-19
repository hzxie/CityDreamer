# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2023-04-06 09:50:37
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-07-19 13:04:51
# @Email:  root@haozhexie.com

import logging
import os
import torch
import shutil

import core.vqgan.test
import losses.smoothness
import models.vqgan
import utils.average_meter
import utils.datasets
import utils.distributed
import utils.helpers
import utils.summary_writer


from time import time


def train(cfg):
    torch.backends.cudnn.benchmark = True
    # Set up networks
    local_rank = utils.distributed.get_rank()
    vqae = models.vqgan.VQAutoEncoder(cfg)
    if torch.cuda.is_available():
        logging.info("Start running the DDP on rank %d." % local_rank)
        vqae = torch.nn.parallel.DistributedDataParallel(
            vqae.to(local_rank), device_ids=[local_rank]
        )
    else:
        vqae.device = torch.device("cpu")

    # Set up data loader
    train_dataset = utils.datasets.get_dataset(cfg, cfg.TRAIN.VQGAN.DATASET, "train")
    val_dataset = utils.datasets.get_dataset(cfg, cfg.TRAIN.VQGAN.DATASET, "val")
    train_sampler = None
    val_sampler = None
    if torch.cuda.is_available():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=local_rank, shuffle=True, drop_last=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, rank=local_rank, shuffle=False
        )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.VQGAN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=False,
        sampler=train_sampler,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=True,
        sampler=val_sampler,
    )

    # Set up optimizers
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vqae.parameters()),
        lr=cfg.TRAIN.VQGAN.BASE_LR
        * cfg.TRAIN.VQGAN.BATCH_SIZE
        * torch.cuda.device_count(),
        weight_decay=cfg.TRAIN.VQGAN.WEIGHT_DECAY,
        betas=cfg.TRAIN.VQGAN.BETAS,
    )

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()
    ce_loss = torch.nn.CrossEntropyLoss()
    smooth_loss = losses.smoothness.SmoothnessLoss(
        use_diag=cfg.TRAIN.VQGAN.SMOOTH_LOSS_USE_DIAG,
        size=(
            cfg.TRAIN.VQGAN.BATCH_SIZE,
            1,
            cfg.NETWORK.VQGAN.RESOLUTION,
            cfg.NETWORK.VQGAN.RESOLUTION,
        ),
        device=vqae.device,
    )

    # Load the pretrained model if exists
    init_epoch = 0
    if "CKPT" in cfg.CONST:
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        vqae.load_state_dict(checkpoint["vqae"])
        init_epoch = checkpoint["epoch_index"]
        logging.info("Recover completed. Current epoch = #%d" % (init_epoch,))

    # Set up folders for logs, snapshot and checkpoints
    if utils.distributed.is_master():
        output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
        cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
        cfg.DIR.LOGS = output_dir % "logs"
        os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)
        # Summary writer
        tb_writer = utils.summary_writer.SummaryWriter(cfg)

    # Training/Testing the network
    n_batches = len(train_data_loader)
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.VQGAN.N_EPOCHS + 1):
        epoch_start_time = time()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        train_losses = utils.average_meter.AverageMeter(
            ["RecLoss", "SmthLoss", "SegLoss", "QuantLoss", "TotalLoss"]
        )
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        batch_end_time = time()
        for batch_idx, data in enumerate(train_data_loader):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time() - batch_end_time)

            input = utils.helpers.var_or_cuda(data["img"], vqae.device)
            output = utils.helpers.var_or_cuda(data["img"], vqae.device)
            pred, quant_loss = vqae(input)
            rec_loss = l1_loss(pred[:, [0]], output[:, [0]])
            smth_loss = smooth_loss(pred[:, [0]], output[:, [0]])
            seg_loss = ce_loss(pred[:, 1:], torch.argmax(output[:, 1:], dim=1))
            loss = (
                rec_loss * cfg.TRAIN.VQGAN.REC_LOSS_FACTOR
                + smth_loss * cfg.TRAIN.VQGAN.SMOOTH_LOSS_FACTOR
                + seg_loss * cfg.TRAIN.VQGAN.SEG_LOSS_FACTOR
                + quant_loss
            )
            train_losses.update(
                [
                    rec_loss.item(),
                    smth_loss.item(),
                    seg_loss.item(),
                    quant_loss.item(),
                    loss.item(),
                ]
            )
            vqae.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if utils.distributed.is_master():
                tb_writer.add_scalars(
                    {
                        "VQGAN/Loss/Batch/Rec": train_losses.val(0),
                        "VQGAN/Loss/Batch/Smth": train_losses.val(1),
                        "VQGAN/Loss/Batch/Seg": train_losses.val(2),
                        "VQGAN/Loss/Batch/Quant": train_losses.val(3),
                        "VQGAN/Loss/Batch/Total": train_losses.val(4),
                    },
                    n_itr,
                )
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s"
                    % (
                        epoch_idx,
                        cfg.TRAIN.VQGAN.N_EPOCHS,
                        batch_idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ["%.4f" % l for l in train_losses.val()],
                    )
                )

        epoch_end_time = time()
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "VQGAN/Loss/Epoch/Rec/Train": train_losses.avg(0),
                    "VQGAN/Loss/Epoch/Smth/Train": train_losses.avg(1),
                    "VQGAN/Loss/Epoch/Seg/Train": train_losses.avg(2),
                    "VQGAN/Loss/Epoch/Quant/Train": train_losses.avg(3),
                    "VQGAN/Loss/Epoch/Total/Train": train_losses.avg(4),
                },
                epoch_idx,
            )
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s"
                % (
                    epoch_idx,
                    cfg.TRAIN.VQGAN.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    ["%.4f" % l for l in train_losses.avg()],
                )
            )

        # Evaluate the current model
        test_losses, key_frames = core.vqgan.test(cfg, val_data_loader, vqae)
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "VQGAN/Loss/Epoch/Rec/Test": test_losses.avg(0),
                    "VQGAN/Loss/Epoch/Smth/Test": test_losses.avg(1),
                    "VQGAN/Loss/Epoch/Seg/Test": test_losses.avg(2),
                    "VQGAN/Loss/Epoch/Quant/Test": test_losses.avg(3),
                    "VQGAN/Loss/Epoch/Total/Test": test_losses.avg(4),
                },
                epoch_idx,
            )
            tb_writer.add_images(key_frames, epoch_idx)
            # Save ckeckpoints
            logging.info("Saved checkpoint to ckpt-last.pth ...")
            torch.save(
                {
                    "cfg": cfg,
                    "epoch_index": epoch_idx,
                    "vqae": vqae.state_dict(),
                },
                os.path.join(cfg.DIR.CHECKPOINTS, "ckpt-last.pth"),
            )
            if epoch_idx % cfg.TRAIN.VQGAN.CKPT_SAVE_FREQ == 0:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "ckpt-last.pth"),
                    os.path.join(
                        cfg.DIR.CHECKPOINTS, "ckpt-epoch-%03d.pth" % epoch_idx
                    ),
                )

    if utils.distributed.is_master():
        tb_writer.close()
