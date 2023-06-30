# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2023-06-30 10:12:55
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-06-30 10:43:36
# @Email:  root@haozhexie.com

import argparse
import logging
import os
import sys
import web

DEMO_HOME_DIR = os.path.abspath(os.path.dirname(__file__))
# Add parent dir to PYTHONPATH
sys.path.append(os.path.join(DEMO_HOME_DIR, os.path.pardir))

import scripts.inference as inference

CONSTANTS = {}
MODELS = {
    "vqae": None,
    "sampler": None,
    "gancraft_bg": None,
    "gancraft_fg": None,
}
URLS = (
    "/", "IndexController",
)


class IndexController:
    def GET(self):
        web.header("Content-Type", "text/html")
        with open(os.path.join(DEMO_HOME_DIR, "index.html")) as f:
            return f.read()


def get_runtime_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampler_ckpt",
        default=os.path.join(inference.PROJECT_HOME, "output", "sampler.pth"),
    )
    parser.add_argument(
        "--gancraft_bg_ckpt",
        default=os.path.join(inference.PROJECT_HOME, "output", "gancraft-bg.pth"),
    )
    parser.add_argument(
        "--gancraft_fg_ckpt",
        default=os.path.join(inference.PROJECT_HOME, "output", "gancraft-fg.pth"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
    )
    args = get_runtime_arguments()
    logging.info("Initialize models ...")
    (
        MODELS["vqae"],
        MODELS["sampler"],
        MODELS["gancraft_bg"],
        MODELS["gancraft_fg"],
    ) = inference.get_models(
        args.sampler_ckpt, args.gancraft_bg_ckpt, args.gancraft_fg_ckpt
    )

    # NOTE: Make sure that the work_dir is set to DEMO_HOME_DIR
    # Otherwise, the static files won't be served.
    os.chdir(DEMO_HOME_DIR)

    # Start HTTP server
    logging.info("Launching demo ...")
    app = web.application(URLS, globals())
    app.run()
