# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-14 10:31:58
# @Email:  root@haozhexie.com

from easydict import EasyDict

# fmt: off
__C                                              = EasyDict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = EasyDict()
__C.DATASETS.OSM_LAYOUT                          = EasyDict()
__C.DATASETS.OSM_LAYOUT.PIN_MEMORY               = False
__C.DATASETS.OSM_LAYOUT.N_REPEAT                 = 250
__C.DATASETS.OSM_LAYOUT.DIR                      = "./data/osm"
__C.DATASETS.OSM_LAYOUT.N_CLASSES                = 7
__C.DATASETS.OSM_LAYOUT.IGNORED_CLASSES          = [0]

#
# Constants
#
__C.CONST                                        = EasyDict()
__C.CONST.EXP_NAME                               = ""
__C.CONST.N_WORKERS                              = 32
__C.CONST.NETWORK                                = None

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
__C.WANDB.MODE                                   = "online"
__C.WANDB.RUN_ID                                 = None
__C.WANDB.SYNC_TENSORBOARD                       = False

#
# Network
#
__C.NETWORK                                      = EasyDict()
__C.NETWORK.VQGAN                                = EasyDict()
__C.NETWORK.VQGAN.N_IN_CHANNELS                  = 8
__C.NETWORK.VQGAN.N_OUT_CHANNELS                 = 8
__C.NETWORK.VQGAN.N_Z_CHANNELS                   = 512
__C.NETWORK.VQGAN.N_EMBED                        = 512
__C.NETWORK.VQGAN.EMBED_DIM                      = 512
__C.NETWORK.VQGAN.N_RES_BLOCKS                   = 2
__C.NETWORK.VQGAN.N_CHANNEL_BASE                 = 128
__C.NETWORK.VQGAN.N_CHANNEL_FACTORS              = [1, 1, 2, 2, 4]
__C.NETWORK.VQGAN.RESOLUTION                     = 512
__C.NETWORK.VQGAN.ATTN_RESOLUTION                = 32
__C.NETWORK.VQGAN.DROPOUT                        = 0.0
__C.NETWORK.SAMPLER                              = EasyDict()
__C.NETWORK.SAMPLER.N_EMBED                      = 512
__C.NETWORK.SAMPLER.N_HEAD                       = 8
__C.NETWORK.SAMPLER.N_LAYERS                     = 24
__C.NETWORK.SAMPLER.BLOCK_SIZE                   = cfg.NETWORK.VQGAN.ATTN_RESOLUTION ** 2
__C.NETWORK.SAMPLER.DROPOUT                      = 0.0
__C.NETWORK.SAMPLER.TOTAL_STEPS                  = 256

#
# Train
#
__C.TRAIN                                        = EasyDict()
__C.TRAIN.VQGAN                                  = EasyDict()
__C.TRAIN.VQGAN.DATASET                          = "OSM_LAYOUT"
__C.TRAIN.VQGAN.N_EPOCHS                         = 500
__C.TRAIN.VQGAN.REC_LOSS_FACTOR                  = 10
__C.TRAIN.VQGAN.CTR_LOSS_FACTOR                  = 10
__C.TRAIN.VQGAN.SEG_LOSS_FACTOR                  = 1
__C.TRAIN.VQGAN.CKPT_SAVE_FREQ                   = 25
__C.TRAIN.VQGAN.BATCH_SIZE                       = 2
__C.TRAIN.VQGAN.BASE_LR                          = 4.5e-6
__C.TRAIN.VQGAN.WEIGHT_DECAY                     = 0
__C.TRAIN.VQGAN.BETAS                            = (0.5, 0.9)
__C.TRAIN.SAMPLER                                = EasyDict()
__C.TRAIN.SAMPLER.DATASET                        = "OSM_LAYOUT"
__C.TRAIN.SAMPLER.N_EPOCHS                       = 500
__C.TRAIN.SAMPLER.CKPT_SAVE_FREQ                 = 25
__C.TRAIN.SAMPLER.BATCH_SIZE                     = 10
__C.TRAIN.SAMPLER.N_WARMUP_ITERS                 = 7500
__C.TRAIN.SAMPLER.LR                             = 2e-4
__C.TRAIN.SAMPLER.WEIGHT_DECAY                   = 0
__C.TRAIN.SAMPLER.BETAS                          = (0.9, 0.999)

#
# Test
#
__C.TEST                                         = EasyDict()
__C.TEST.SAMPLER                                 = EasyDict()
__C.TEST.SAMPLER.N_SAMPLES                       = 2
__C.TEST.SAMPLER.TEMPERATURES                    = [0.5, 1.0, 2.0]
# fmt: on
