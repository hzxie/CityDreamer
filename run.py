# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 21:27:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-10 20:16:46
# @Email:  root@haozhexie.com


import argparse
import importlib
import logging
import torch
import os
import sys

import core.vqgan
import core.sampler

from pprint import pprint
from datetime import datetime


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description="The argument parser of the runner")
    parser.add_argument(
        "-e",
        "--exp",
        dest="exp_name",
        help="The name of the experiment",
        default="%s" % datetime.now(),
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_file",
        help="Path to the config.py file",
        default="config.py",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--network",
        dest="network",
        help="The network name to train or test. ['VQGAN', 'Sampler']",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        help="The GPU device to use (e.g., 0,1,2,3).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--ckpt",
        dest="ckpt",
        help="Initialize the network from a pretrained model.",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--run",
        dest="run_id",
        help="The unique run ID for WandB",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test", dest="test", help="Test the network.", action="store_true"
    )
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Read the experimental config
    exec(compile(open(args.cfg_file, "rb").read(), args.cfg_file, "exec"))
    cfg = locals()["__C"]

    # Parse runtime arguments
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.exp_name is not None:
        cfg.CONST.EXP_NAME = args.exp_name
    if args.network is not None:
        cfg.CONST.NETWORK = args.network
    if args.ckpt is not None:
        cfg.CONST.CKPT_FILE_PATH = args.ckpt
    if args.run_id is not None:
        cfg.WANDB.RUN_ID = args.run_id
    if args.run_id is not None and args.ckpt is None:
        raise Exception("No checkpoints")

    # Print the current config
    local_rank = 0
    if torch.cuda.is_available() and not args.test:
        torch.distributed.init_process_group("nccl")
        local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        pprint(cfg)

    # Start train/test processes
    if not args.test:
        if cfg.CONST.NETWORK == "VQGAN":
            core.vqgan.train(cfg)
        elif cfg.CONST.NETWORK == "Sampler":
            core.sampler.train(cfg)
        else:
            raise Exception("Unknown network: %s" % cfg.CONST.NETWORK)
    else:
        if "CKPT_FILE_PATH" not in cfg.CONST or not os.path.exists(cfg.CONST.CKPT_FILE_PATH):
            logging.error("Please specify the file path of checkpoint.")
            sys.exit(2)

        if cfg.CONST.NETWORK == "VQGAN":
            core.vqgan.test(cfg)
        elif cfg.CONST.NETWORK == "Sampler":
            core.sampler.test(cfg)
        else:
            raise Exception("Unknown network: %s" % cfg.CONST.NETWORK)


if __name__ == "__main__":
    # References: https://stackoverflow.com/a/53553516/1841143
    importlib.reload(logging)
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    main()
