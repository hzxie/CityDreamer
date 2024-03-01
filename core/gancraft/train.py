# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2023-04-21 19:45:23
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-01 15:30:24
# @Email:  root@haozhexie.com

import copy
import logging
import os
import torch
import torch.nn.functional as F
import shutil

import core.gancraft.test
import losses.gan
import losses.perceptual
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
    gancraft_g = models.gancraft.GanCraftGenerator(cfg)
    gancraft_d = models.gancraft.GanCraftDiscriminator(cfg)
    if torch.cuda.is_available():
        logging.info("Start running the DDP on rank %d." % local_rank)
        gancraft_g = torch.nn.parallel.DistributedDataParallel(
            gancraft_g.to(local_rank),
            device_ids=[local_rank],
        )
        gancraft_d = torch.nn.parallel.DistributedDataParallel(
            gancraft_d.to(local_rank),
            device_ids=[local_rank],
        )
        if cfg.TRAIN.GANCRAFT.EMA_ENABLED:
            gancraft_g_ema = copy.deepcopy(gancraft_g).requires_grad_(False).eval()
    else:
        gancraft_g.device = torch.device("cpu")
        gancraft_d.device = torch.device("cpu")

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
        persistent_workers=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=cfg.CONST.N_WORKERS,
        collate_fn=utils.datasets.collate_fn,
        pin_memory=False,
        sampler=val_sampler,
        persistent_workers=True,
    )

    # Set up optimizers
    optimizer_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gancraft_g.parameters()),
        lr=cfg.TRAIN.GANCRAFT.LR_GENERATOR,
        eps=cfg.TRAIN.GANCRAFT.EPS,
        weight_decay=cfg.TRAIN.GANCRAFT.WEIGHT_DECAY,
        betas=cfg.TRAIN.GANCRAFT.BETAS,
    )
    optimizer_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gancraft_d.parameters()),
        lr=cfg.TRAIN.GANCRAFT.LR_DISCRIMINATOR,
        eps=cfg.TRAIN.GANCRAFT.EPS,
        weight_decay=cfg.TRAIN.GANCRAFT.WEIGHT_DECAY,
        betas=cfg.TRAIN.GANCRAFT.BETAS,
    )

    # Set up loss functions
    l1_loss = torch.nn.L1Loss()
    gan_loss = losses.gan.GANLoss()
    perceptual_loss = losses.perceptual.PerceptualLoss(
        cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_MODEL,
        cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_LAYERS,
        cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_WEIGHTS,
        device=gancraft_g.device,
    )

    # Load the pretrained model if exists
    init_epoch = 0
    if "CKPT" in cfg.CONST:
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        checkpoint = torch.load(cfg.CONST.CKPT, map_location=gancraft_g.device)
        gancraft_g.load_state_dict(checkpoint["gancraft_g"])
        gancraft_d.load_state_dict(checkpoint["gancraft_d"])
        if cfg.TRAIN.GANCRAFT.EMA_ENABLED:
            gancraft_g_ema.load_state_dict(checkpoint["gancraft_g_ema"])
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
        train_losses = utils.average_meter.AverageMeter(
            [
                "L1Loss",
                "PerceptualLoss",
                "GANLoss",
                "GANLossFake",
                "GANLossReal",
                "GenLoss",
                "DisLoss",
            ]
        )
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        # Switch models to train mode
        gancraft_g.train()
        gancraft_d.train()
        batch_end_time = time()
        for batch_idx, data in enumerate(train_data_loader):
            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            data_time.update(time() - batch_end_time)
            # Warm up the discriminator
            if n_itr <= cfg.TRAIN.GANCRAFT.DISCRIMINATOR_N_WARMUP_ITERS:
                lr = (
                    cfg.TRAIN.GANCRAFT.LR_DISCRIMINATOR
                    * n_itr
                    / cfg.TRAIN.GANCRAFT.DISCRIMINATOR_N_WARMUP_ITERS
                )
                for pg in optimizer_d.param_groups:
                    pg["lr"] = lr

            hf_seg = utils.helpers.var_or_cuda(
                torch.cat([data["hf"], data["seg"]], dim=1), gancraft_g.device
            )
            voxel_id = utils.helpers.var_or_cuda(data["voxel_id"], gancraft_g.device)
            depth2 = utils.helpers.var_or_cuda(data["depth2"], gancraft_g.device)
            raydirs = utils.helpers.var_or_cuda(data["raydirs"], gancraft_g.device)
            cam_origin = utils.helpers.var_or_cuda(
                data["cam_origin"], gancraft_g.device
            )
            footages = utils.helpers.var_or_cuda(data["footage"], gancraft_g.device)
            masks = utils.helpers.var_or_cuda(data["mask"], gancraft_g.device)
            if cfg.NETWORK.GANCRAFT.BUILDING_MODE:
                masks[
                    ~torch.isin(
                        voxel_id[:, None, ..., 0, 0],
                        torch.tensor(
                            [
                                cfg.NETWORK.GANCRAFT.FACADE_CLS_ID,
                                cfg.NETWORK.GANCRAFT.ROOF_CLS_ID,
                            ],
                            device=gancraft_g.device,
                        ),
                    )
                ] = 0
            else:
                masks[
                    voxel_id[:, None, ..., 0, 0] == cfg.NETWORK.GANCRAFT.FACADE_CLS_ID
                ] = 0

            seg_maps = utils.helpers.masks_to_onehots(
                data["voxel_id"][..., 0, 0], cfg.DATASETS.OSM_LAYOUT.N_CLASSES
            )
            building_stats = (
                None if "building_stats" not in data else data["building_stats"]
            )

            # Discriminator Update Step
            utils.helpers.requires_grad(gancraft_g, False)
            utils.helpers.requires_grad(gancraft_d, True)

            with torch.no_grad():
                fake_imgs = gancraft_g(
                    hf_seg, voxel_id, depth2, raydirs, cam_origin, building_stats
                )
                fake_imgs = fake_imgs.detach()

            fake_labels = gancraft_d(fake_imgs, seg_maps, masks)
            real_labels = gancraft_d(footages, seg_maps, masks)

            gan_loss_weights = None
            if cfg.NETWORK.GANCRAFT.BUILDING_MODE:
                gan_loss_weights = F.interpolate(masks, scale_factor=0.25)

            fake_loss = gan_loss(fake_labels, False, gan_loss_weights, dis_update=True)
            real_loss = gan_loss(real_labels, True, gan_loss_weights, dis_update=True)
            loss_d = fake_loss + real_loss
            gancraft_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Generator Update Step
            utils.helpers.requires_grad(gancraft_d, False)
            utils.helpers.requires_grad(gancraft_g, True)

            fake_imgs = gancraft_g(
                hf_seg, voxel_id, depth2, raydirs, cam_origin, building_stats
            )
            fake_labels = gancraft_d(fake_imgs, seg_maps, masks)
            _l1_loss = l1_loss(fake_imgs * masks, footages * masks)
            _perceptual_loss = perceptual_loss(fake_imgs * masks, footages * masks)
            _gan_loss = gan_loss(fake_labels, True, gan_loss_weights, dis_update=False)
            loss_g = (
                _l1_loss * cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR
                + _perceptual_loss * cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR
                + _gan_loss * cfg.TRAIN.GANCRAFT.GAN_LOSS_FACTOR
            )
            gancraft_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Update EMA
            if cfg.TRAIN.GANCRAFT.EMA_ENABLED:
                ema_n_itrs = cfg.TRAIN.GANCRAFT.EMA_N_RAMPUP_ITERS
                if cfg.TRAIN.GANCRAFT.EMA_RAMPUP is not None:
                    ema_n_itrs = min(ema_n_itrs, cfg.TRAIN.GANCRAFT.EMA_RAMPUP * n_itr)

                ema_beta = 0.5 ** (
                    cfg.TRAIN.GANCRAFT.BATCH_SIZE / max(ema_n_itrs, 1e-8)
                )
                for pg, p_gema in zip(
                    gancraft_g.parameters(), gancraft_g_ema.parameters()
                ):
                    p_gema.copy_(pg.lerp(p_gema, ema_beta))
                for bg, b_gema in zip(gancraft_g.buffers(), gancraft_g_ema.buffers()):
                    b_gema.copy_(bg)

            train_losses.update(
                [
                    _l1_loss.item(),
                    _perceptual_loss.item(),
                    _gan_loss.item(),
                    fake_loss.item(),
                    real_loss.item(),
                    loss_g.item(),
                    loss_d.item(),
                ]
            )
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if utils.distributed.is_master():
                tb_writer.add_scalars(
                    {
                        "GANCraft/Loss/Batch/L1": train_losses.val(0),
                        "GANCraft/Loss/Batch/Perceptual": train_losses.val(1),
                        "GANCraft/Loss/Batch/GAN": train_losses.val(2),
                        "GANCraft/Loss/Batch/GANFake": train_losses.val(3),
                        "GANCraft/Loss/Batch/GANReal": train_losses.val(4),
                        "GANCraft/Loss/Batch/GenTotal": train_losses.val(5),
                        "GANCraft/Loss/Batch/DisTotal": train_losses.val(6),
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
                        ["%.4f" % l for l in train_losses.val()],
                    )
                )

        epoch_end_time = time()
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "GANCraft/Loss/Epoch/L1/Train": train_losses.avg(0),
                    "GANCraft/Loss/Epoch/Perceptual/Train": train_losses.avg(1),
                    "GANCraft/Loss/Epoch/GAN/Train": train_losses.avg(2),
                    "GANCraft/Loss/Epoch/GANFake/Train": train_losses.avg(3),
                    "GANCraft/Loss/Epoch/GANReal/Train": train_losses.avg(4),
                    "GANCraft/Loss/Epoch/GenTotal/Train": train_losses.avg(5),
                    "GANCraft/Loss/Epoch/DisTotal/Train": train_losses.avg(6),
                },
                epoch_idx,
            )
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s"
                % (
                    epoch_idx,
                    cfg.TRAIN.GANCRAFT.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    ["%.4f" % l for l in train_losses.avg()],
                )
            )

        # Evaluate the current model
        test_losses, key_frames = core.gancraft.test(
            cfg,
            val_data_loader,
            gancraft_g_ema if cfg.TRAIN.GANCRAFT.EMA_ENABLED else gancraft_g,
        )
        if utils.distributed.is_master():
            tb_writer.add_scalars(
                {
                    "GANCraft/Loss/Epoch/L1/Test": test_losses.avg(0),
                },
                epoch_idx,
            )
            tb_writer.add_images(key_frames, epoch_idx)
            # Save ckeckpoints
            logging.info("Saved checkpoint to ckpt-last.pth ...")
            ckpt = {
                "cfg": cfg,
                "epoch_index": epoch_idx,
                "gancraft_g": gancraft_g.state_dict(),
                "gancraft_d": gancraft_d.state_dict(),
            }
            if cfg.TRAIN.GANCRAFT.EMA_ENABLED:
                ckpt["gancraft_g_ema"] = gancraft_g_ema.state_dict()

            torch.save(
                ckpt,
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
