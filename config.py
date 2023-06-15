# -*- coding: utf-8 -*-
#
# @File:   config.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 20:14:54
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-06-15 15:12:49
# @Email:  root@haozhexie.com

from easydict import EasyDict

# fmt: off
__C                                             = EasyDict()
cfg                                             = __C

#
# Dataset Config
#
cfg.DATASETS                                     = EasyDict()
cfg.DATASETS.OSM_LAYOUT                          = EasyDict()
cfg.DATASETS.OSM_LAYOUT.PIN_MEMORY               = ["hf", "seg"]
cfg.DATASETS.OSM_LAYOUT.N_REPEAT                 = 250
cfg.DATASETS.OSM_LAYOUT.DIR                      = "./data/osm"
cfg.DATASETS.OSM_LAYOUT.IGNORED_CLASSES          = [0]
cfg.DATASETS.OSM_LAYOUT.N_CLASSES                = 7
cfg.DATASETS.OSM_LAYOUT.MAX_HEIGHT               = 640
cfg.DATASETS.GOOGLE_EARTH                        = EasyDict()
cfg.DATASETS.GOOGLE_EARTH.PIN_MEMORY             = ["hf", "seg"]
cfg.DATASETS.GOOGLE_EARTH.N_REPEAT               = 1
cfg.DATASETS.GOOGLE_EARTH.N_VIEWS                = 60
cfg.DATASETS.GOOGLE_EARTH.DIR                    = "./data/ges"
cfg.DATASETS.GOOGLE_EARTH.VOL_SIZE               = 1536
cfg.DATASETS.GOOGLE_EARTH_BUILDING               = EasyDict()
cfg.DATASETS.GOOGLE_EARTH_BUILDING.PIN_MEMORY    = ["hf", "seg"]
cfg.DATASETS.GOOGLE_EARTH_BUILDING.N_REPEAT      = 1
cfg.DATASETS.GOOGLE_EARTH_BUILDING.VOL_SIZE      = 672
cfg.DATASETS.GOOGLE_EARTH_BUILDING.CITY          = "US-NewYork"

#
# Constants
#
cfg.CONST                                        = EasyDict()
cfg.CONST.EXP_NAME                               = ""
cfg.CONST.N_WORKERS                              = 8
cfg.CONST.NETWORK                                = None

#
# Directories
#
cfg.DIR                                          = EasyDict()
cfg.DIR.OUTPUT                                   = "./output"

#
# Memcached
#
cfg.MEMCACHED                                    = EasyDict()
cfg.MEMCACHED.ENABLED                            = False
cfg.MEMCACHED.LIBRARY_PATH                       = "/mnt/lustre/share/pymc/py3"
cfg.MEMCACHED.SERVER_CONFIG                      = "/mnt/lustre/share/memcached_client/server_list.conf"
cfg.MEMCACHED.CLIENT_CONFIG                      = "/mnt/lustre/share/memcached_client/client.conf"

#
# WandB
#
cfg.WANDB                                        = EasyDict()
cfg.WANDB.ENABLED                                = False
cfg.WANDB.PROJECT                                = "City-Gen"
cfg.WANDB.ENTITY                                 = "haozhexie"
cfg.WANDB.MODE                                   = "online"
cfg.WANDB.RUN_ID                                 = None
cfg.WANDB.SYNC_TENSORBOARD                       = False

#
# Network
#
cfg.NETWORK                                      = EasyDict()
# VQGAN
cfg.NETWORK.VQGAN                                = EasyDict()
cfg.NETWORK.VQGAN.N_IN_CHANNELS                  = 7
cfg.NETWORK.VQGAN.N_OUT_CHANNELS                 = 7
cfg.NETWORK.VQGAN.N_Z_CHANNELS                   = 512
cfg.NETWORK.VQGAN.N_EMBED                        = 512
cfg.NETWORK.VQGAN.EMBED_DIM                      = 512
cfg.NETWORK.VQGAN.N_RES_BLOCKS                   = 2
cfg.NETWORK.VQGAN.N_CHANNEL_BASE                 = 128
cfg.NETWORK.VQGAN.N_CHANNEL_FACTORS              = [1, 1, 2, 2, 4]
cfg.NETWORK.VQGAN.RESOLUTION                     = 512
cfg.NETWORK.VQGAN.ATTN_RESOLUTION                = 32
cfg.NETWORK.VQGAN.DROPOUT                        = 0.0
# Sampler
cfg.NETWORK.SAMPLER                              = EasyDict()
cfg.NETWORK.SAMPLER.N_EMBED                      = 512
cfg.NETWORK.SAMPLER.N_HEAD                       = 8
cfg.NETWORK.SAMPLER.N_LAYERS                     = 24
cfg.NETWORK.SAMPLER.BLOCK_SIZE                   = cfg.NETWORK.VQGAN.ATTN_RESOLUTION ** 2
cfg.NETWORK.SAMPLER.DROPOUT                      = 0.0
cfg.NETWORK.SAMPLER.TOTAL_STEPS                  = 256
# GANCraft
cfg.NETWORK.GANCRAFT                             = EasyDict()
cfg.NETWORK.GANCRAFT.BUILDING_MODE               = False
cfg.NETWORK.GANCRAFT.STYLE_DIM                   = 128
cfg.NETWORK.GANCRAFT.N_SAMPLE_POINTS_PER_RAY     = 24
cfg.NETWORK.GANCRAFT.DIST_SCALE                  = 0.25
cfg.NETWORK.GANCRAFT.ENCODER                     = "LOCAL"
cfg.NETWORK.GANCRAFT.ENCODER_OUT_DIM             = 64 if cfg.NETWORK.GANCRAFT.BUILDING_MODE else 32
cfg.NETWORK.GANCRAFT.GLOBAL_ENCODER_N_BLOCKS     = 6
cfg.NETWORK.GANCRAFT.LOCAL_ENCODER_NORM          = "GROUP_NORM"
cfg.NETWORK.GANCRAFT.POS_EMD                     = "SIN_COS"
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_FEATURES     = True
cfg.NETWORK.GANCRAFT.POS_EMD_INCUDE_CORDS        = False
cfg.NETWORK.GANCRAFT.HASH_GRID_N_LEVELS          = 16
cfg.NETWORK.GANCRAFT.HASH_GRID_LEVEL_DIM         = 8
cfg.NETWORK.GANCRAFT.SIN_COS_FREQ_BENDS          = 10
cfg.NETWORK.GANCRAFT.RENDER_HIDDEN_DIM           = 256
cfg.NETWORK.GANCRAFT.RENDER_STYLE_DIM            = 256
cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_SIGMA        = 1
cfg.NETWORK.GANCRAFT.RENDER_OUT_DIM_COLOR        = 64
cfg.NETWORK.GANCRAFT.DIS_N_CHANNEL_BASE          = 128

#
# Train
#
cfg.TRAIN                                        = EasyDict()
# VQGAN
cfg.TRAIN.VQGAN                                  = EasyDict()
cfg.TRAIN.VQGAN.DATASET                          = "OSM_LAYOUT"
cfg.TRAIN.VQGAN.N_EPOCHS                         = 1000
cfg.TRAIN.VQGAN.REC_LOSS_FACTOR                  = 10
cfg.TRAIN.VQGAN.SEG_LOSS_FACTOR                  = 1
cfg.TRAIN.VQGAN.CKPT_SAVE_FREQ                   = 25
cfg.TRAIN.VQGAN.BATCH_SIZE                       = 2
cfg.TRAIN.VQGAN.BASE_LR                          = 4.5e-6
cfg.TRAIN.VQGAN.WEIGHT_DECAY                     = 0
cfg.TRAIN.VQGAN.BETAS                            = (0.5, 0.9)
# Sampler
cfg.TRAIN.SAMPLER                                = EasyDict()
cfg.TRAIN.SAMPLER.DATASET                        = "OSM_LAYOUT"
cfg.TRAIN.SAMPLER.N_EPOCHS                       = 1000
cfg.TRAIN.SAMPLER.CKPT_SAVE_FREQ                 = 25
cfg.TRAIN.SAMPLER.BATCH_SIZE                     = 10
cfg.TRAIN.SAMPLER.N_WARMUP_ITERS                 = 7500
cfg.TRAIN.SAMPLER.LR                             = 2e-4
cfg.TRAIN.SAMPLER.WEIGHT_DECAY                   = 0
cfg.TRAIN.SAMPLER.BETAS                          = (0.9, 0.999)
# GANCraft
cfg.TRAIN.GANCRAFT                               = EasyDict()
cfg.TRAIN.GANCRAFT.DATASET                       = "GOOGLE_EARTH_BUILDING" if cfg.NETWORK.GANCRAFT.BUILDING_MODE else "GOOGLE_EARTH"
cfg.TRAIN.GANCRAFT.N_EPOCHS                      = 500
cfg.TRAIN.GANCRAFT.CKPT_SAVE_FREQ                = 25
cfg.TRAIN.GANCRAFT.BATCH_SIZE                    = 1
cfg.TRAIN.GANCRAFT.LR_GENERATOR                  = 1e-4
cfg.TRAIN.GANCRAFT.LR_DISCRIMINATOR              = 1e-5
cfg.TRAIN.GANCRAFT.DISCRIMINATOR_N_WARMUP_ITERS  = 100000
cfg.TRAIN.GANCRAFT.EPS                           = 1e-7
cfg.TRAIN.GANCRAFT.WEIGHT_DECAY                  = 0
cfg.TRAIN.GANCRAFT.BETAS                         = (0., 0.999)
cfg.TRAIN.GANCRAFT.CROP_SIZE                     = (192, 192)
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_MODEL         = "vgg19"
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_LAYERS        = ["relu_3_1", "relu_4_1", "relu_5_1"]
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_WEIGHTS       = [0.125, 0.25, 1.0]
cfg.TRAIN.GANCRAFT.REC_LOSS_FACTOR               = 10
cfg.TRAIN.GANCRAFT.PERCEPTUAL_LOSS_FACTOR        = 10
cfg.TRAIN.GANCRAFT.GAN_LOSS_FACTOR               = 0.5
cfg.TRAIN.GANCRAFT.EMA_ENABLED                   = False
cfg.TRAIN.GANCRAFT.EMA_RAMPUP                    = 0.05
cfg.TRAIN.GANCRAFT.EMA_N_RAMPUP_ITERS            = 10000

#
# Test
#
cfg.TEST                                         = EasyDict()
cfg.TEST.VQGAN                                   = EasyDict()
cfg.TEST.VQGAN.DATASET                           = "OSM_LAYOUT"
cfg.TEST.SAMPLER                                 = EasyDict()
cfg.TEST.SAMPLER.N_SAMPLES                       = 2
cfg.TEST.SAMPLER.TEMPERATURES                    = [0.5, 1.0, 2.0]
cfg.TEST.GANCRAFT                                = EasyDict()
cfg.TEST.GANCRAFT.DATASET                        = "GOOGLE_EARTH_BUILDING" if cfg.NETWORK.GANCRAFT.BUILDING_MODE else "GOOGLE_EARTH"
cfg.TEST.GANCRAFT.CROP_SIZE                      = (480, 270)
# fmt: on
