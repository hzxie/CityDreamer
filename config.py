# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 19:32:42
# @Email:  root@haozhexie.com

from easydict import EasyDict
from datetime import datetime

# fmt: off
__C                                              = EasyDict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = EasyDict()
__C.DATASETS.OSM_LAYOUT                          = EasyDict()
__C.DATASETS.OSM_LAYOUT.DIR                      = "./data/osm"

#
# Constants
#
__C.CONST                                        = EasyDict()
__C.CONST.EXP_NAME                               = datetime.now().isoformat()
__C.CONST.DEVICE                                 = '0'
__C.CONST.N_WORKERS                              = 32

#
# Directories
#
__C.DIR                                          = EasyDict()
__C.DIR.OUTPUT                                   = './output'

#
# Memcached
#
__C.MEMCACHED                                    = EasyDict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# WandB
#
__C.WANDB                                        = EasyDict()
__C.WANDB.ENABLED                                = False
__C.WANDB.PROJECT                                = "City-Gen"
__C.WANDB.ENTITY                                 = "haozhexie"
__C.WANDB.MODE                                   = "offline"
__C.WANDB.SYNC_TENSORBOARD                       = True

#
# Network
#
__C.NETWORK                                      = EasyDict()
__C.NETWORK.VQGAN                                = EasyDict()
__C.NETWORK.VQGAN.N_RES_BLOCKS                   = 2
__C.NETWORK.VQGAN.Z_CHANNELS                     = 256
__C.NETWORK.VQGAN.EMBED_DIM                      = 512
__C.NETWORK.VQGAN.N_CHANNEL_BASE                 = 128
__C.NETWORK.VQGAN.N_CHANNEL_FACTORS              = [1, 1, 2, 2, 4, 4]
__C.NETWORK.VQGAN.RESOLUTION                     = 512
__C.NETWORK.VQGAN.ATTN_RESOLUTION                = [16]
__C.NETWORK.VQGAN.N_EMBED                        = 512
__C.NETWORK.VQGAN.DROPOUT                        = 0.0

#
# Train
#
__C.TRAIN                                        = EasyDict()
__C.TRAIN.NETWORK                                = "VQGAN"
__C.TRAIN.VQGAN                                  = EasyDict()
__C.TRAIN.VQGAN.DATASET                          = "OSM_LAYOUT"
__C.TRAIN.VQGAN.N_EPOCHS                         = 200000
__C.TRAIN.VQGAN.CKPT_SAVE_FREQ                   = 1000
__C.TRAIN.VQGAN.BATCH_SIZE                       = 2
__C.TRAIN.VQGAN.BASE_LR                          = 4.5e-6
__C.TRAIN.VQGAN.WEIGHT_DECAY                     = 0
__C.TRAIN.VQGAN.BETAS                            = (0.5, 0.9)
# fmt: on
