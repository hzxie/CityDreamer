# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-05 20:54:57
# @Email:  root@haozhexie.com

from easydict import EasyDict as edict

# fmt: off
__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 32

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './logs'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# WandB
#
__C.WANDB                                        = edict()
__C.WANDB.ENABLED                                = False
__C.WANDB.PROJECT_NAME                           = 'City-Gen'
__C.WANDB.TAGS                                   = []

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.VQGAN                                = edict()
__C.NETWORK.VQGAN.LR                             = 4.5e-6
__C.NETWORK.VQGAN.DROPOUT                        = 0.0
__C.NETWORK.VQGAN.N_RES_BLOCKS                   = 2
__C.NETWORK.VQGAN.N_CHANNEL_BASE                 = 128
__C.NETWORK.VQGAN.N_CHANNEL_FACTORS              = [1, 1, 2, 2, 4, 4]
__C.NETWORK.VQGAN.RESOLUTION                     = 512
__C.NETWORK.VQGAN.ATTN_RESOLUTION                = 16
__C.NETWORK.VQGAN.N_EMBED                        = 512
__C.NETWORK.VQGAN.EMBED_DIM                      = 512
__C.NETWORK.VQGAN.Z_CHANNELS                     = 256
# fmt: on
