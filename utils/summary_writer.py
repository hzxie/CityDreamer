# -*- coding: utf-8 -*-
#
# @File:   summary_writer.py
# @Author: Haozhe Xie
# @Date:   2020-04-19 12:52:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 15:50:21
# @Email:  root@haozhexie.com

import logging
import os
import tensorboardX

try:
    import wandb
except Exception as ex:
    logging.warning(ex)


class SummaryWriter(object):
    def __init__(self, cfg):
        os.makedirs(cfg.DIR.OUTPUT, exist_ok=True)
        if cfg.WANDB.ENABLED:
            self.writer = wandb.init(
                entity=cfg.WANDB.ENTITY,
                project=cfg.WANDB.PROJECT,
                name=cfg.CONST.EXP_NAME,
                dir=cfg.DIR.OUTPUT,
                mode=cfg.WANDB.MODE,
            )
        else:
            self.writer = tensorboardX.SummaryWriter(cfg.DIR.LOGS)

    def add_scalars(self, scalars, step=None):
        if type(self.writer) == tensorboardX.writer.SummaryWriter:
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        else:
            self.writer.log(scalars)

    def add_images(self, images, step=None):
        if type(self.writer) == tensorboardX.writer.SummaryWriter:
            for k, v in images.items():
                self.writer.add_image(k, v, step)
        else:
            self.writer.log({k: wandb.Image(v) for k, v in images.items()})

    def close(self):
        if type(self.writer) == tensorboardX.writer.SummaryWriter:
            self.writer.close()
        else:
            self.writer.finish(exit_code=0)
