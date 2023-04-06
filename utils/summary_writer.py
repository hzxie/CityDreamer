# -*- coding: utf-8 -*-
#
# @File:   summary_writer.py
# @Author: Haozhe Xie
# @Date:   2020-04-19 12:52:36
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 19:48:53
# @Email:  root@haozhexie.com

import numpy as np
import logging
import PIL
import os
import torch.utils.tensorboard

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
            self.writer = torch.utils.tensorboard.SummaryWriter(cfg.DIR.LOGS)

    def add_scalars(self, scalars, step=None):
        if type(self.writer) == torch.utils.tensorboard.writer.SummaryWriter:
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, step)
        else:
            self.writer.log(scalars)

    def _get_tb_image(self, image):
        if type(image) == PIL.Image.Image:
            return np.array(image.convert('RGB'))
        elif len(image.shape) == 2:
            return image
        else:
            raise Exception("Unknown image format")

    def _get_tb_image_format(self, image):
        if type(image) == PIL.Image.Image:
            return "HWC"
        elif len(image.shape) == 2:
            return "HW"
        else:
            raise Exception("Unknown image format")

    def add_images(self, images, step=None):
        if type(self.writer) == torch.utils.tensorboard.writer.SummaryWriter:
            for k, v in images.items():
                self.writer.add_image(
                    k, self._get_tb_image(v), step, dataformats=self._get_tb_image_format(v)
                )
        else:
            self.writer.log({k: wandb.Image(v) for k, v in images.items()})

    def close(self):
        if type(self.writer) == torch.utils.tensorboard.writer.SummaryWriter:
            self.writer.close()
        else:
            self.writer.finish(exit_code=0)
