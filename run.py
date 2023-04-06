# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-04-05 21:27:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 21:50:40
# @Email:  root@haozhexie.com


import argparse
import importlib
import logging
import os
import sys

import core

from pprint import pprint


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description="The argument parser of the runner")
    parser.add_argument(
        "-e", "--exp", dest="exp_name", help="Experiment Name", default=None, type=str
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
        "-g", "--gpu", dest="gpu_id", help="GPU device to use", default=None, type=str
    )
    parser.add_argument(
        "--test", dest="test", help="Test neural networks", action="store_true"
    )
    parser.add_argument(
        "--inference",
        dest="inference",
        help="Inference for benchmark",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--weights",
        dest="weights",
        help="Initialize network from the weights file",
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    # Read the experimental config
    exec(compile(open(args.cfg_file, "rb").read(), args.cfg_file, "exec"))
    cfg = locals()["__C"]
    pprint(cfg)

    # Parse runtime arguments
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.exp_name is not None:
        cfg.CONST.EXP_NAME = args.exp_name
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights

    # Start train/test processes
    if not args.test and not args.inference:
        # cfg.CONST.MODE = "train"
        core.train(cfg)
    else:
        if "WEIGHTS" not in cfg.CONST or not os.path.exists(cfg.CONST.WEIGHTS):
            logging.error("Please specify the file path of checkpoint.")
            sys.exit(2)

        if args.test:
            # cfg.CONST.MODE = "test"
            core.test(cfg)
        else:
            # cfg.CONST.MODE = "inference"
            core.test(cfg)


if __name__ == "__main__":
    # References: https://stackoverflow.com/a/53553516/1841143
    importlib.reload(logging)
    logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    main()
