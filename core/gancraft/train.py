# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2023-04-21 19:45:23
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-05-06 16:04:16
# @Email:  root@haozhexie.com


import logging
import os
import torch
import shutil

import core.gancraft.test
import models.gancraft
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
    gancraft = models.gancraft.GanCraftGenerator(cfg)
    if torch.cuda.is_available():
        logging.info("Start running the DDP on rank %d." % local_rank)
        gancraft = torch.nn.parallel.DistributedDataParallel(
            gancraft.to(local_rank), device_ids=[local_rank]
        )
    else:
        gancraft.device = torch.device("cpu")

    # Set up data loader
    train_dataset = utils.datasets.get_dataset(cfg, cfg.TRAIN.GANCRAFT.DATASET, "train")
    val_dataset = utils.datasets.get_dataset(cfg, cfg.TRAIN.GANCRAFT.DATASET, "val")
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
        batch_size=cfg.TRAIN.GANCRAFT.BATCH_SIZE,
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
        filter(lambda p: p.requires_grad, gancraft.parameters()),
        lr=cfg.TRAIN.GANCRAFT.LR,
        eps=cfg.TRAIN.GANCRAFT.EPS,
        weight_decay=cfg.TRAIN.GANCRAFT.WEIGHT_DECAY,
        betas=cfg.TRAIN.GANCRAFT.BETAS,
    )

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()

    # Load the pretrained model if exists
    init_epoch = 0
    if "CKPT" in cfg.CONST:
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT)
        gancraft.load_state_dict(checkpoint["gancraft"])
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
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.GANCRAFT.N_EPOCHS + 1):
        epoch_start_time = time()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        losses = utils.average_meter.AverageMeter(["RecLoss"])
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        batch_end_time = time()
        for batch_idx, data in enumerate(train_data_loader):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time() - batch_end_time)

            hf_seg = utils.helpers.var_or_cuda(
                torch.cat([data["hf"], data["seg"]], dim=1), gancraft.device
            )
            voxel_id = utils.helpers.var_or_cuda(data["voxel_id"], gancraft.device)
            depth2 = utils.helpers.var_or_cuda(data["depth2"], gancraft.device)
            raydirs = utils.helpers.var_or_cuda(data["raydirs"], gancraft.device)
            cam_ori_t = utils.helpers.var_or_cuda(data["cam_ori_t"], gancraft.device)
            footage = utils.helpers.var_or_cuda(data["footage"], gancraft.device)
            mask = utils.helpers.var_or_cuda(data["mask"], gancraft.device)

            fake_imgs = gancraft(hf_seg, voxel_id, depth2, raydirs, cam_ori_t)
            loss = l1_loss(fake_imgs * mask, footage * mask)
            losses.update([loss.item()])
            gancraft.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if utils.distributed.is_master():
                tb_writer.add_scalars(
                    {
                        "GANCraft/Loss/Batch/RecLoss": losses.val(0),
                    },
                    n_itr,
                )
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s"
                    % (
                        epoch_idx,
                        cfg.TRAIN.GANCRAFT.N_EPOCHS,
                        batch_idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ["%.4f" % l for l in losses.val()],
                    )
                )

        epoch_end_time = time()
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "GANCraft/Loss/Epoch/Rec/Train": losses.avg(0),
                },
                epoch_idx,
            )
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s"
                % (
                    epoch_idx,
                    cfg.TRAIN.GANCRAFT.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    ["%.4f" % l for l in losses.avg()],
                )
            )

        # Evaluate the current model
        losses, key_frames = core.gancraft.test(cfg, val_data_loader, gancraft)
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "GANCraft/Loss/Epoch/Rec/Test": losses.avg(0),
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
                    "gancraft": gancraft.state_dict(),
                },
                os.path.join(cfg.DIR.CHECKPOINTS, "ckpt-last.pth"),
            )
            if epoch_idx % cfg.TRAIN.GANCRAFT.CKPT_SAVE_FREQ == 0:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "ckpt-last.pth"),
                    os.path.join(
                        cfg.DIR.CHECKPOINTS, "ckpt-epoch-%03d.pth" % epoch_idx
                    ),
                )

    if utils.distributed.is_master():
        tb_writer.close()
